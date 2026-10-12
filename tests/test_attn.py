import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine.model import Attention

tok = AutoTokenizer.from_pretrained("gpt2")
hf = AutoModelForCausalLM.from_pretrained("gpt2").eval()
hf_attn = hf.transformer.h[0].attn

captured = {}
def hook(module, args, kwargs, output):
    captured["x"] = args[0] if args else kwargs["hidden_states"]
    captured["y"] = output[0] if isinstance(output, tuple) else output

handle = hf_attn.register_forward_hook(hook, with_kwargs=True)
ids = tok("The best poker players know that", return_tensors="pt").input_ids
with torch.no_grad():
    hf(ids)
handle.remove()

attn = Attention(768, 12)
with torch.no_grad():
    attn.c_attn.weight.copy_(hf_attn.c_attn.weight.t())
    attn.c_attn.bias.copy_(hf_attn.c_attn.bias)
    attn.c_proj.weight.copy_(hf_attn.c_proj.weight.t())
    attn.c_proj.bias.copy_(hf_attn.c_proj.bias)
    ours = attn(captured["x"])

print("input shape:", tuple(captured["x"].shape))
print("max diff:", (ours - captured["y"]).abs().max().item())
assert torch.allclose(ours, captured["y"], atol=1e-4), "Mismatch!"
print("PASS")
