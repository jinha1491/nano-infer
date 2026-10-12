import torch


def truncate(cache, length):
    return [(k[:, :, :length], v[:, :, :length]) for k, v in cache]


@torch.no_grad()
def generate_speculative(target, draft, input_ids, max_new_tokens, n_draft=4, stats=None):
    t_logits, t_cache = target(input_ids)
    _, d_cache = draft(input_ids)
    nxt = t_logits[:, -1:, :].argmax(dim=-1)
    out = [nxt]
    n_out = 1
    proposed = accepted = rounds = 0

    while n_out < max_new_tokens:
        base = t_cache[0][0].size(2)

        x = nxt
        guesses = []
        for _ in range(n_draft):
            d_logits, d_cache = draft(x, d_cache)
            x = d_logits[:, -1:, :].argmax(dim=-1)
            guesses.append(x)
        guesses = torch.cat(guesses, dim=1)

        t_logits, t_cache = target(torch.cat([nxt, guesses], dim=1), t_cache)
        preds = t_logits.argmax(dim=-1)

        match = (guesses == preds[:, :-1])[0].int()
        n = int(match.cumprod(0).sum())
        new = torch.cat([guesses[:, :n], preds[:, n:n + 1]], dim=1)

        t_cache = truncate(t_cache, base + 1 + n)
        if n == n_draft:
            _, d_cache = draft(guesses[:, -1:], d_cache)
        d_cache = truncate(d_cache, base + 1 + n)

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
