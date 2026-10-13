import torch
from engine.batch import SlotCache, make_decoder


def truncate(cache, length):
    return [(k[:, :, :length], v[:, :, :length]) for k, v in cache]


class TupleDraft:
    def __init__(self, model):
        self.model = model

    def prefill(self, ids):
        _, self.cache = self.model(ids)

    def step(self, x):
        logits, self.cache = self.model(x, self.cache)
        return logits[:, -1:, :].argmax(dim=-1)

    def rollback(self, length):
        self.cache = truncate(self.cache, length)


class SlotDraft:
    def __init__(self, model, max_len, mode):
        self.model = model
        self.cache = SlotCache(model, 1, max_len)
        self.decode = make_decoder(model, self.cache, mode)

    def prefill(self, ids):
        _, kv = self.model(ids)
        self.cache.load(0, kv)

    def step(self, x):
        return self.decode(x)[:, -1:, :].argmax(dim=-1)

    def rollback(self, length):
        self.cache.lens.fill_(length)


@torch.no_grad()
def generate_speculative(target, draft, input_ids, max_new_tokens, n_draft=4,
                         stats=None, draft_mode="tuple", max_len=512):
    if draft_mode == "tuple":
        drafter = TupleDraft(draft)
    else:
        assert input_ids.size(1) + max_new_tokens + n_draft + 1 <= max_len
        drafter = SlotDraft(draft, max_len, draft_mode)

    t_logits, t_cache = target(input_ids)
    drafter.prefill(input_ids)
    nxt = t_logits[:, -1:, :].argmax(dim=-1)
    out = [nxt]
    n_out = 1
    proposed = accepted = rounds = 0

    while n_out < max_new_tokens:
        base = t_cache[0][0].size(2)

        x = nxt
        guesses = []
        for _ in range(n_draft):
            x = drafter.step(x)
            guesses.append(x)
        guesses = torch.cat(guesses, dim=1)

        t_logits, t_cache = target(torch.cat([nxt, guesses], dim=1), t_cache)
        preds = t_logits.argmax(dim=-1)

        match = (guesses == preds[:, :-1])[0].int()
        n = int(match.cumprod(0).sum())
        new = torch.cat([guesses[:, :n], preds[:, n:n + 1]], dim=1)

        t_cache = truncate(t_cache, base + 1 + n)
        if n == n_draft:
            drafter.step(guesses[:, -1:])
        drafter.rollback(base + 1 + n)

        out.append(new)
        n_out += new.size(1)
        nxt = new[:, -1:]
        proposed += n_draft
        accepted += n
        rounds += 1

    if stats is not None:
        stats.update(proposed=proposed, accepted=accepted, rounds=rounds)
    gen = torch.cat(out, dim=1)[:, :max_new_tokens]
    return torch.cat([input_ids, gen], dim=1)
