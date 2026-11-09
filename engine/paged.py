import math
import torch


class PagedCache:
    def __init__(self, model, n_slots, n_blocks, block_size=16, max_len=512):
        attn = model.layers[0].self_attn
        w = model.embed_tokens.weight
        kw = dict(device=w.device, dtype=w.dtype)
        shape = (n_blocks, attn.n_kv_head, block_size, attn.d_head)
        self.k = [torch.zeros(shape, **kw) for _ in model.layers]
        self.v = [torch.zeros(shape, **kw) for _ in model.layers]

        self.bs = block_size
        self.n_blocks = n_blocks
        self.max_blocks = math.ceil(max_len / block_size)
        self.max_len = self.max_blocks * block_size

        self.table = torch.zeros(n_slots, self.max_blocks, dtype=torch.long, device=w.device)
        self.lens = torch.zeros(n_slots, dtype=torch.long, device=w.device)

        self.free = list(range(n_blocks - 1, 0, -1))
        self.owned = [[] for _ in range(n_slots)]
        self.reserved = [0] * n_slots
        self.host_lens = [0] * n_slots
        self.peak_used = 0

    def blocks_for(self, n_tokens):
        return math.ceil(n_tokens / self.bs)

    def available(self):
        promised = sum(r - len(o) for r, o in zip(self.reserved, self.owned))
        return len(self.free) - promised

    def can_admit(self, n_tokens):
        return self.blocks_for(n_tokens) <= self.available()

    def _grow(self, slot, n_tokens):
        while len(self.owned[slot]) * self.bs < n_tokens:
            b = self.free.pop()
            self.table[slot, len(self.owned[slot])] = b
            self.owned[slot].append(b)
        used = self.n_blocks - 1 - len(self.free)
        self.peak_used = max(self.peak_used, used)

    def load(self, slot, kv, reserve=None):
        T = kv[0][0].size(2)
        self.reserved[slot] = self.blocks_for(reserve or T)
        self._grow(slot, T)
        for i, (k, v) in enumerate(kv):
            for j, b in enumerate(self.owned[slot]):
                lo, hi = j * self.bs, min((j + 1) * self.bs, T)
                self.k[i][b, :, :hi - lo] = k[0, :, lo:hi]
                self.v[i][b, :, :hi - lo] = v[0, :, lo:hi]
        self.lens[slot] = T
        self.host_lens[slot] = T

    def prepare_step(self, active):
        for s in active:
            self._grow(s, self.host_lens[s] + 1)
            self.host_lens[s] += 1

    def release(self, slot):
        self.free.extend(reversed(self.owned[slot]))
        self.owned[slot] = []
        self.reserved[slot] = 0
        self.host_lens[slot] = 0
        self.table[slot] = 0
        self.lens[slot] = 0

    def write(self, i, k, v):
        slots = torch.arange(k.size(0), device=k.device)
        blk = self.table[slots, self.lens // self.bs]
        off = self.lens % self.bs
        self.k[i][blk, :, off] = k
        self.v[i][blk, :, off] = v

    def read(self, i, L):
        nb = min(math.ceil(L / self.bs), self.max_blocks)
        idx = self.table[:, :nb]
        k, v = self.k[i][idx], self.v[i][idx]
        S, _, KV, bs, D = k.shape
        k = k.transpose(1, 2).reshape(S, KV, nb * bs, D)
        v = v.transpose(1, 2).reshape(S, KV, nb * bs, D)
        return k, v
