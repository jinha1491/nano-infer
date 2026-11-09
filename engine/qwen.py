import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rope_tables(d_head, max_pos, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
    freqs = torch.outer(torch.arange(max_pos).float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class MLP(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.up_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, d, n_head, n_kv_head, d_head, bias, qk_norm, eps):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.d_head = d_head
        self.n_rep = n_head // n_kv_head
        self.q_proj = nn.Linear(d, n_head * d_head, bias=bias)
        self.k_proj = nn.Linear(d, n_kv_head * d_head, bias=bias)
        self.v_proj = nn.Linear(d, n_kv_head * d_head, bias=bias)
        self.o_proj = nn.Linear(n_head * d_head, d, bias=False)
        self.q_norm = RMSNorm(d_head, eps) if qk_norm else None
        self.k_norm = RMSNorm(d_head, eps) if qk_norm else None

    def forward(self, x, cos, sin, kv=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.d_head)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv is not None:
            k = torch.cat([kv[0], k], dim=2)
            v = torch.cat([kv[1], v], dim=2)
        new_kv = (k, v)
        T_k = k.size(2)

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        mask = torch.tril(torch.ones(T, T_k, dtype=torch.bool, device=x.device),
                          diagonal=T_k - T)
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att.float(), dim=-1).to(q.dtype)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y), new_kv


class Block(nn.Module):
    def __init__(self, d, n_head, n_kv_head, d_head, d_ff, bias, qk_norm, eps):
        super().__init__()
        self.input_layernorm = RMSNorm(d, eps)
        self.self_attn = Attention(d, n_head, n_kv_head, d_head, bias, qk_norm, eps)
        self.post_attention_layernorm = RMSNorm(d, eps)
        self.mlp = MLP(d, d_ff)

    def forward(self, x, cos, sin, kv=None):
        a, new_kv = self.self_attn(self.input_layernorm(x), cos, sin, kv)
        x = x + a
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv


class Qwen(nn.Module):
    def __init__(self, n_layer, n_head, n_kv_head, d, d_ff, vocab, d_head=None,
                 bias=True, qk_norm=False, max_pos=4096, theta=1e6, eps=1e-6, tie=True):
        super().__init__()
        d_head = d_head or d // n_head
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(
            [Block(d, n_head, n_kv_head, d_head, d_ff, bias, qk_norm, eps)
             for _ in range(n_layer)])
        self.norm = RMSNorm(d, eps)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        if tie:
            self.lm_head.weight = self.embed_tokens.weight

        cos, sin = rope_tables(d_head, max_pos, theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx, cache=None):
        B, T = idx.shape
        past = cache[0][0].size(2) if cache is not None else 0
        cos = self.cos[past:past + T]
        sin = self.sin[past:past + T]

        x = self.embed_tokens(idx)
        new_cache = []
        for i, layer in enumerate(self.layers):
            x, kv = layer(x, cos, sin, cache[i] if cache is not None else None)
            new_cache.append(kv)
        return self.lm_head(self.norm(x)), new_cache

    @classmethod
    def from_pretrained(cls, name="Qwen/Qwen3-0.6B", max_pos=4096, dtype=torch.float32):
        import glob
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        from transformers import AutoConfig

        c = AutoConfig.from_pretrained(name)
        theta = getattr(c, "rope_theta", None) or c.rope_parameters["rope_theta"]
        is_qwen3 = c.model_type == "qwen3"
        model = cls(c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads,
                    c.hidden_size, c.intermediate_size, c.vocab_size,
                    d_head=getattr(c, "head_dim", None), bias=not is_qwen3,
                    qk_norm=is_qwen3, max_pos=max_pos, theta=theta,
                    eps=c.rms_norm_eps, tie=c.tie_word_embeddings).to(dtype)

        path = snapshot_download(name, allow_patterns=["*.safetensors"])
        sd = model.state_dict()
        loaded = set()
        with torch.no_grad():
            for f in sorted(glob.glob(f"{path}/*.safetensors")):
                with safe_open(f, framework="pt") as fh:
                    for key in fh.keys():
                        ours = key.removeprefix("model.")
                        if ours in sd:
                            sd[ours].copy_(fh.get_tensor(key))
                            loaded.add(ours)

        missing = set(sd) - loaded
        if c.tie_word_embeddings:
            missing.discard("lm_head.weight")
        assert not missing, f"weights missing from checkpoint: {sorted(missing)[:5]}"
        return model.eval()
