import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from engine.model import GPT2
from engine.kv import generate_kv
from engine.batch import generate_static

tok = AutoTokenizer.from_pretrained("gpt2")
model = GPT2.from_pretrained("gpt2")

texts = [
    "The best poker players know that",
    "In a tournament, the most important skill is",
    "Hi",
    "When you are holding pocket aces before the flop, you should usually",
]
prompts = [tok(t, return_tensors="pt").input_ids for t in texts]
print("prompt lengths:", [p.size(1) for p in prompts])

outs = generate_static(model, prompts, 30)
for i, p in enumerate(prompts):
    ref = generate_kv(model, p, 30)[0]
    assert torch.equal(ref, outs[i]), f"prompt {i}: mismatch!"
    print(f"PASS {i}: {tok.decode(outs[i])!r}")
