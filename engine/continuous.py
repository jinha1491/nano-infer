from dataclasses import dataclass, field
import torch
from engine.batch import SlotCache, make_decoder
from engine.paged import PagedCache


@dataclass
class Request:
    rid: int
    prompt: torch.Tensor
    max_new_tokens: int
    out: list = field(default_factory=list)


@torch.no_grad()
def generate_continuous(model, requests, n_slots, max_len=512, stats=None, mode="dynamic",
                        n_blocks=None, block_size=16):
    if n_blocks is None:
        cache = SlotCache(model, n_slots, max_len)
    else:
        cache = PagedCache(model, n_slots, n_blocks, block_size, max_len)
    decode = make_decoder(model, cache, mode)
    device = cache.lens.device
    queue = list(requests)
    slot_req = [None] * n_slots
    cur = torch.zeros(n_slots, 1, dtype=torch.long, device=device)
    results = {}
    if stats is not None:
        stats.update(decode_steps=0, prefills=0, busy_slot_steps=0)

    def finish(s, req):
        results[req.rid] = torch.cat([req.prompt[0].cpu(), torch.tensor(req.out)])
        slot_req[s] = None
        cache.release(s)

    while queue or any(r is not None for r in slot_req):
        for s in range(n_slots):
            if slot_req[s] is not None or not queue:
                continue
            req = queue[0]
            total = req.prompt.size(1) + req.max_new_tokens
            assert total <= max_len
            if not cache.can_admit(total):
                break
            queue.pop(0)
            logits, kv = model(req.prompt)
            cache.load(s, kv, reserve=total)
            nid = int(logits[0, -1].argmax())
            req.out.append(nid)
            cur[s, 0] = nid
            slot_req[s] = req
            if stats is not None:
                stats["prefills"] += 1
            if len(req.out) >= req.max_new_tokens:
                finish(s, req)

        active = [s for s in range(n_slots) if slot_req[s] is not None]
        if not active:
            if queue and not cache.can_admit(queue[0].prompt.size(1) + queue[0].max_new_tokens):
                raise RuntimeError("request too large for the block pool")
            continue

        cache.prepare_step(active)
        logits = decode(cur)
        if stats is not None:
            stats["decode_steps"] += 1
            stats["busy_slot_steps"] += len(active)
        cur = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        toks = cur[:, 0].tolist()

        for s in range(n_slots):
            req = slot_req[s]
            if req is None:
                cache.lens[s] = 0
                continue
            req.out.append(toks[s])
            if len(req.out) >= req.max_new_tokens:
                finish(s, req)

    if n_blocks is not None:
        assert len(cache.free) == n_blocks - 1, "blocks leaked"
        if stats is not None:
            stats["peak_blocks"] = cache.peak_used
    return results
