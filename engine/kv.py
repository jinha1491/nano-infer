import torch

@torch.no_grad()
def generate_kv(model, input_ids, max_new_tokens):
    logits, cache = model(input_ids)
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    out = torch.cat([input_ids, next_id], dim=1)

    for _ in range(max_new_tokens - 1):
        logits, cache = model(next_id, cache)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_id], dim=1)
    return out
