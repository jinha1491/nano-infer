import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from engine.model import GPT2
from engine.kv import generate_kv
from engine.batch import generate_static
from engine.quant import Int8Linear, quantize_model

torch.manual_seed(0)
lin = torch.nn.Linear(768, 3072)
for per_channel in [True, False]:
    q = Int8Linear(lin, per_channel)
    x = torch.randn(4, 768)
    old = F.linear(x, q.w_q.float() * q.scale.view(-1, 1), q.bias)
    new = q(x)
    assert torch.allclose(old, new, rtol=1e-4, atol=1e-4), f"per_channel={per_channel}: mismatch!"
print("PASS: scale-after-matmul equals dequantize-then-matmul")

if not torch.cuda.is_available():
    print("SKIP: compiled mode needs an NVIDIA GPU")
    sys.exit(0)

tok = AutoTokenizer.from_pretrained("gpt2")
texts = ["The best poker players know that", "Hi"]
for name, model in [("fp32", GPT2.from_pretrained("gpt2").cuda()),
                    ("int8", quantize_model(GPT2.from_pretrained("gpt2")).cuda())]:
    prompts = [tok(t, return_tensors="pt").input_ids.cuda() for t in texts]
    graph = generate_static(model, prompts, 30, max_len=128, mode="graph")
    comp = generate_static(model, prompts, 30, max_len=128, mode="compiled")
    same = all(torch.equal(a, b) for a, b in zip(graph, comp))
    print(f"PASS: {name} compiled mode runs; matches graph mode: {same}")
