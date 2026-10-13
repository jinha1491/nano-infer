import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from engine.model import GPT2
from engine.kv import generate_kv
from engine.batch import generate_static
from engine.continuous import Request, generate_continuous

tok = AutoTokenizer.from_pretrained("gpt2")
model = GPT2.from_pretrained("gpt2")

texts = [
    "The best poker players know that",
    "In a tournament, the most important skill is",
    "Hi",
    "When you are holding pocket aces before the flop, you should usually",
]
prompts = [tok(t, return_tensors="pt").input_ids for t in texts]
lengths = [20, 35, 5, 30]
refs = [generate_kv(model, p, n)[0] for p, n in zip(prompts, lengths)]

outs = generate_static(model, prompts, 20, max_len=128, static=True)
for i, p in enumerate(prompts):
    assert torch.equal(generate_kv(model, p, 20)[0], outs[i]), f"static batch {i}: mismatch!"
print("PASS: static batching, static shapes")

reqs = [Request(i, p, n) for i, (p, n) in enumerate(zip(prompts, lengths))]
results = generate_continuous(model, reqs, n_slots=2, max_len=128, static=True)
for i in range(len(prompts)):
    assert torch.equal(refs[i], results[i]), f"continuous {i}: mismatch!"
print("PASS: continuous batching, static shapes")
