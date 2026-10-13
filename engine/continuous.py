from dataclasses import dataclass, field
import torch
from engine.batch import SlotCache, make_decoder


@dataclass
class Request:
    rid: int
    prompt: torch.Tensor
    max_new_tokens: int
    out: list = field(default_factory=list)


@torch.no_grad()
def generate_continuous(model, requests, n_slots, max_len=512, stats=None, mode="dynamic"):
    cache = SlotCache(model, n_slots, max_len)
    decode = make_decoder(model, cache, mode)
    device = cache.lens.device
    queue = list(requests)
    slot_req = [None] * n_slots
    cur = torch.zeros(n_slots, 1, dtype=torch.long, device=device)
    results = {}
    if stats is not None:
        stats.update(decode_steps=0, prefills=0)

    def finish(s, req):
        results[req.rid] = torch.cat([req.prompt[0].cpu(), torch.tensor(req.out)])
        slot_req[s] = None
        cache.lens[s] = 0

    while queue or any(r is not None for r in slot_req):
        for s in range(n_slots):
            if slot_req[s] is None and queue:
                req = queue.pop(0)
                assert req.prompt.size(1) + req.max_new_tokens <= max_len
                logits, kv = model(req.prompt)
                cache.load(s, kv)
                nid = int(logits[0, -1].argmax())
                req.out.append(nid)
                cur[s, 0] = nid
                slot_req[s] = req
                if stats is not None:
                    stats["prefills"] += 1
                if len(req.out) >= req.max_new_tokens:
                    finish(s, req)

        if all(r is None for r in slot_req):
            continue

        logits = decode(cur)
        if stats is not None:
            stats["decode_steps"] += 1
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

    return results
