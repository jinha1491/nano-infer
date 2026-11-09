import torch
import torch.nn.functional as F
from engine.qwen import Qwen


class SlotCache:
    def __init__(self, model, n_slots, max_len):
        if isinstance(model, Qwen):
            attn, layers = model.layers[0].self_attn, model.layers
            n_kv = attn.n_kv_head
            weight = model.embed_tokens.weight
        else:
            attn, layers = model.h[0].attn, model.h
            n_kv = attn.n_head
            weight = model.wte.weight
        shape = (n_slots, n_kv, max_len, attn.d_head)
        kw = dict(device=weight.device, dtype=weight.dtype)
        self.k = [torch.zeros(shape, **kw) for _ in layers]
        self.v = [torch.zeros(shape, **kw) for _ in layers]
        self.lens = torch.zeros(n_slots, dtype=torch.long, device=weight.device)

    def load(self, slot, kv):
        T = kv[0][0].size(2)
        for i, (k, v) in enumerate(kv):
            self.k[i][slot, :, :T] = k[0]
            self.v[i][slot, :, :T] = v[0]
        self.lens[slot] = T


def attn_decode(attn, x, kc, vc, lens, L):
    S, _, C = x.shape
    q, k, v = attn.c_attn(x).split(C, dim=-1)
    q = q.view(S, 1, attn.n_head, attn.d_head).transpose(1, 2)
    k = k.view(S, attn.n_head, attn.d_head)
    v = v.view(S, attn.n_head, attn.d_head)

    slots = torch.arange(S, device=x.device)
    kc[slots, :, lens] = k
    vc[slots, :, lens] = v

    att = (q @ kc[:, :, :L].transpose(-2, -1)) / (attn.d_head ** 0.5)
    mask = torch.arange(L, device=x.device)[None, :] <= lens[:, None]
    att = att.masked_fill(~mask[:, None, None, :], float("-inf"))
    att = F.softmax(att, dim=-1)

    y = att @ vc[:, :, :L]
    y = y.transpose(1, 2).contiguous().view(S, 1, C)
    return attn.c_proj(y)


def block_decode(block, x, kc, vc, lens, L):
    x = x + attn_decode(block.attn, block.ln_1(x), kc, vc, lens, L)
    x = x + block.mlp(block.ln_2(x))
    return x


def rope_decode(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


def qwen_attn_decode(attn, x, cos, sin, kc, vc, lens, L):
    S = x.size(0)
    H, KV, D, R = attn.n_head, attn.n_kv_head, attn.d_head, attn.n_rep
    q = attn.q_proj(x).view(S, KV, R, D)
    k = attn.k_proj(x).view(S, KV, D)
    v = attn.v_proj(x).view(S, KV, D)
    if attn.q_norm is not None:
        q, k = attn.q_norm(q), attn.k_norm(k)

    q = rope_decode(q, cos[:, None, None, :], sin[:, None, None, :])
    k = rope_decode(k, cos[:, None, :], sin[:, None, :])

    slots = torch.arange(S, device=x.device)
    kc[slots, :, lens] = k
    vc[slots, :, lens] = v

    att = (q @ kc[:, :, :L].transpose(-2, -1)) / (D ** 0.5)
    mask = torch.arange(L, device=x.device)[None, :] <= lens[:, None]
    att = att.masked_fill(~mask[:, None, None, :], float("-inf"))
    att = F.softmax(att.float(), dim=-1).to(q.dtype)

    y = (att @ vc[:, :, :L]).reshape(S, 1, H * D)
    return attn.o_proj(y)


def qwen_layer_decode(layer, x, cos, sin, kc, vc, lens, L):
    x = x + qwen_attn_decode(layer.self_attn, layer.input_layernorm(x),
                             cos, sin, kc, vc, lens, L)
    x = x + layer.mlp(layer.post_attention_layernorm(x))
    return x


@torch.no_grad()
def model_decode(model, idx, cache, static=False):
    if static:
        L = cache.k[0].size(2)
    else:
        L = int(cache.lens.max()) + 1

    if isinstance(model, Qwen):
        cos, sin = model.cos[cache.lens], model.sin[cache.lens]
        x = model.embed_tokens(idx)
        for i, layer in enumerate(model.layers):
            x = qwen_layer_decode(layer, x, cos, sin, cache.k[i], cache.v[i], cache.lens, L)
        logits = model.lm_head(model.norm(x))
    else:
        x = model.wte(idx) + model.wpe(cache.lens).unsqueeze(1)
        for i, block in enumerate(model.h):
            x = block_decode(block, x, cache.k[i], cache.v[i], cache.lens, L)
        logits = model.lm_head(model.ln_f(x))

    cache.lens.add_(1)
    return logits


import torch._dynamo.config
import torch._inductor.config

torch._dynamo.config.automatic_dynamic_shapes = False
torch._inductor.config.coordinate_descent_tuning = True
compiled_model_decode = torch.compile(model_decode)


class GraphedDecoder:
    def __init__(self, model, cache, compile=False):
        fn = compiled_model_decode if compile else model_decode
        step = lambda idx: fn(model, idx, cache, True)

        n_slots = cache.lens.size(0)
        self.idx = torch.zeros(n_slots, 1, dtype=torch.long, device=cache.lens.device)
        saved_lens = cache.lens.clone()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step(self.idx)
        torch.cuda.current_stream().wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = step(self.idx)

        cache.lens.copy_(saved_lens)

    def __call__(self, idx):
        self.idx.copy_(idx)
        self.graph.replay()
        return self.logits


def make_decoder(model, cache, mode):
    if mode == "graph":
        return GraphedDecoder(model, cache)
    if mode == "compiled":
        return GraphedDecoder(model, cache, compile=True)
    static = mode == "static"
    return lambda idx: model_decode(model, idx, cache, static)


@torch.no_grad()
def generate_static(model, prompts, max_new_tokens, max_len=512, mode="dynamic"):
    cache = SlotCache(model, len(prompts), max_len)
    decode = make_decoder(model, cache, mode)
    first = []
    for s, ids in enumerate(prompts):
        assert ids.size(1) + max_new_tokens <= max_len
        logits, kv = model(ids)
        cache.load(s, kv)
        first.append(logits[:, -1, :].argmax(dim=-1, keepdim=True))

    cur = torch.cat(first, dim=0)
    generated = [cur]
    for _ in range(max_new_tokens - 1):
        logits = decode(cur)
        cur = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(cur)

    gen = torch.cat(generated, dim=1)
    return [torch.cat([p[0], gen[s]]) for s, p in enumerate(prompts)]
