import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.c_fc = nn.Linear(d, 4 * d)
        self.c_proj = nn.Linear(4 * d, d)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Attention(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_head = d // n_head
        self.c_attn = nn.Linear(d, 3 * d)
        self.c_proj = nn.Linear(d, d)

    def forward(self, x, kv=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=-1)

        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        if kv is not None:
            k = torch.cat([kv[0], k], dim=2)
            v = torch.cat([kv[1], v], dim=2)
        new_kv = (k, v)
        T_k = k.size(2)

        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        mask = torch.tril(torch.ones(T, T_k, dtype=torch.bool, device=x.device),
                          diagonal=T_k - T)
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), new_kv


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d)
        self.attn = Attention(d, n_head)
        self.ln_2 = nn.LayerNorm(d)
        self.mlp = MLP(d)

    def forward(self, x, kv=None):
        a, new_kv = self.attn(self.ln_1(x), kv)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv


class GPT2(nn.Module):
    def __init__(self, n_layer=12, n_head=12, d=768, vocab=50257, n_pos=1024):
        super().__init__()
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(n_pos, d)
        self.h = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx, cache=None):
        B, T = idx.shape
        past = cache[0][0].size(2) if cache is not None else 0
        pos = torch.arange(past, past + T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        new_cache = []
        for i, block in enumerate(self.h):
            x, kv = block(x, cache[i] if cache is not None else None)
            new_cache.append(kv)
        x = self.ln_f(x)
        return self.lm_head(x), new_cache

    @classmethod
    def from_pretrained(cls, name="gpt2"):
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(name)
        c = hf.config
        model = cls(c.n_layer, c.n_head, c.n_embd, c.vocab_size, c.n_positions)

        hf_sd = hf.state_dict()
        transposed = ("attn.c_attn.weight", "attn.c_proj.weight",
                      "mlp.c_fc.weight", "mlp.c_proj.weight")
        with torch.no_grad():
            for name, param in model.state_dict().items():
                hf_name = name if name == "lm_head.weight" else "transformer." + name
                w = hf_sd[hf_name]
                if name.endswith(transposed):
                    w = w.t()
                param.copy_(w)
        return model.eval()
