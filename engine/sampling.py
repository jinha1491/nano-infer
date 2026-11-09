import torch


def sample(logits, temperature, top_p, generator=None):
    logits = logits.float()
    greedy = logits.argmax(-1, keepdim=True)

    t = temperature.clamp(min=1e-5)[:, None]
    probs = torch.softmax(logits / t, dim=-1)
    sorted_p, idx = probs.sort(dim=-1, descending=True)
    cum = sorted_p.cumsum(dim=-1)
    sorted_p = sorted_p.masked_fill(cum - sorted_p > top_p[:, None], 0.0)
    pick = torch.multinomial(sorted_p, 1, generator=generator)
    sampled = idx.gather(-1, pick)

    return torch.where(temperature[:, None] == 0, greedy, sampled)
