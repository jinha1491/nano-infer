import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM
from engine.model import MLP

hf = AutoModelForCausalLM.from_pretrained("gpt2").eval()
hf_mlp = hf.transformer.h[0].mlp

mlp = MLP(768)
with torch.no_grad():
    mlp.c_fc.weight.copy_(hf_mlp.c_fc.weight.t())
    mlp.c_fc.bias.copy_(hf_mlp.c_fc.bias)
    mlp.c_proj.weight.copy_(hf_mlp.c_proj.weight.t())
    mlp.c_proj.bias.copy_(hf_mlp.c_proj.bias)

x = torch.randn(1, 5, 768)
with torch.no_grad():
    ours = mlp(x)
    ref = hf_mlp(x)

print("max diff:", (ours - ref).abs().max().item())
assert torch.allclose(ours, ref, rtol=1e-4, atol=1e-4), "Mismatch!"
print("PASS")
