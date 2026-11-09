import hashlib
import math
import struct
from collections import OrderedDict
import torch


class PagedCache:
    def __init__(self, model, n_slots, n_blocks, block_size=16, max_len=512,
                 prefix_cache=False):
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
        self.prefix_cache = prefix_cache

        self.table = torch.zeros(n_slots, self.max_blocks, dtype=torch.long, device=w.device)
        self.lens = torch.zeros(n_slots, dtype=torch.long, device=w.device)

        self.free = list(range(n_blocks - 1, 0, -1))
        self.ref = [0] * n_blocks
        self.owned = [[] for _ in range(n_slots)]
        self.reserved = [0] * n_slots
        self.host_lens = [0] * n_slots
        self.peak_used = 0
        self.evictions = 0

        self.block_of = {}
        self.key_of = {}
        self.evictable = OrderedDict()

    def blocks_for(self, n_tokens):
        return math.ceil(n_tokens / self.bs)

    def available(self):
        promised = sum(r - len(o) for r, o in zip(self.reserved, self.owned))
        return len(self.free) + len(self.evictable) - promised

    def can_admit(self, n_tokens):
        return self.blocks_for(n_tokens) <= self.available()

    def _alloc(self):
        if self.free:
            b = self.free.pop()
        else:
            b, _ = self.evictable.popitem(last=False)
            del self.block_of[self.key_of.pop(b)]
            self.evictions += 1
        self.ref[b] = 1
        return b

    def _acquire(self, b):
        if self.ref[b] == 0:
            del self.evictable[b]
        self.ref[b] += 1

    def _append(self, slot, b):
        self.table[slot, len(self.owned[slot])] = b
        self.owned[slot].append(b)

    def _grow(self, slot, n_tokens):
        while len(self.owned[slot]) * self.bs < n_tokens:
            self._append(slot, self._alloc())
        used = self.n_blocks - 1 - len(self.free) - len(self.evictable)
        self.peak_used = max(self.peak_used, used)

    def _keys(self, tokens):
        keys, key = [], b""
        for j in range(len(tokens) // self.bs):
            block = tokens[j * self.bs:(j + 1) * self.bs]
            key = hashlib.sha256(key + struct.pack(f"{len(block)}i", *block)).digest()
            keys.append(key)
        return keys

    def match_prefix(self, tokens):
        if not self.prefix_cache:
            return []
        usable = (len(tokens) - 1) // self.bs * self.bs
        hits = []
        for key in self._keys(tokens[:usable]):
            if key not in self.block_of:
                break
            hits.append(self.block_of[key])
        return hits

    def gather(self, blocks):
        idx = torch.tensor(blocks, device=self.table.device)
        out = []
        for k, v in zip(self.k, self.v):
            n, KV, bs, D = k[idx].shape
            out.append((k[idx].transpose(0, 1).reshape(1, KV, n * bs, D),
                        v[idx].transpose(0, 1).reshape(1, KV, n * bs, D)))
        return out

    def load(self, slot, kv, reserve=None, tokens=None, hits=()):
        T = kv[0][0].size(2)
        self.reserved[slot] = self.blocks_for(reserve or T)
        for b in hits:
            self._acquire(b)
            self._append(slot, b)
        self._grow(slot, T)

        for i, (k, v) in enumerate(kv):
            for j in range(len(hits), len(self.owned[slot])):
                b = self.owned[slot][j]
                lo, hi = j * self.bs, min((j + 1) * self.bs, T)
                self.k[i][b, :, :hi - lo] = k[0, :, lo:hi]
                self.v[i][b, :, :hi - lo] = v[0, :, lo:hi]

        if self.prefix_cache and tokens is not None:
            keys = self._keys(tokens)
            for j in range(len(hits), len(keys)):
                b = self.owned[slot][j]
                if keys[j] not in self.block_of:
                    self.block_of[keys[j]] = b
                    self.key_of[b] = keys[j]

        self.lens[slot] = T
        self.host_lens[slot] = T

    def prepare_step(self, active):
        for s in active:
            self._grow(s, self.host_lens[s] + 1)
            self.host_lens[s] += 1

    def release(self, slot):
        for b in reversed(self.owned[slot]):
            self.ref[b] -= 1
            if self.ref[b] == 0:
                if b in self.key_of:
                    self.evictable[b] = None
                else:
                    self.free.append(b)
        self.owned[slot] = []
        self.reserved[slot] = 0
        self.host_lens[slot] = 0
        self.table[slot] = 0
        self.lens[slot] = 0

    def check_no_leaks(self):
        assert all(r == 0 for r in self.ref), "a block still has references"
        assert len(self.free) + len(self.evictable) == self.n_blocks - 1, "blocks leaked"

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
