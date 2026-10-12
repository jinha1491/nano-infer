import torch

@torch.no_grad()
def generate_naive(model, input_ids, max_new_tokens):
    for _ in range(max_new_tokens):
        logits = model(input_ids, use_cache=False).logits
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=1)
    return input_ids
