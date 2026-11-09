import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from engine.qwen import Qwen
from engine.kv import generate_kv
from engine.batch import generate_static
from engine.continuous import Request, generate_continuous

name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(name)
model = Qwen.from_pretrained(name).to(dev)

texts = [
    "The best poker players know that",
    "In a tournament, the most important skill is",
    "Hi",
    "When you are holding pocket aces before the flop, you should usually",
]
prompts = [tok(t, return_tensors="pt").input_ids.to(dev) for t in texts]
lengths = [20, 35, 5, 30]
refs = [generate_kv(model, p, n)[0].cpu() for p, n in zip(prompts, lengths)]

outs = generate_static(model, prompts, 30, max_len=128)
for i, p in enumerate(prompts):
    assert torch.equal(generate_kv(model, p, 30)[0].cpu(), outs[i].cpu()), f"static {i}: mismatch!"
print("PASS: static batching matches KV cache")

def run(mode, n_slots):
    reqs = [Request(i, p, n) for i, (p, n) in enumerate(zip(prompts, lengths))]
    return generate_continuous(model, reqs, n_slots=n_slots, max_len=128, mode=mode)

for n_slots in [2, 3]:
    res = run("dynamic", n_slots)
    for i in range(len(prompts)):
        assert torch.equal(refs[i], res[i]), f"continuous slots={n_slots} req {i}: mismatch!"
    print(f"PASS: continuous batching, n_slots={n_slots}")

if dev == "cuda":
    static_res, graph_res = run("static", 2), run("graph", 2)
    for i in range(len(prompts)):
        assert torch.equal(static_res[i], graph_res[i]), f"graph {i}: mismatch!"
    print("PASS: CUDA graph matches static")
    print("graph also matches single-sequence KV cache:",
          all(torch.equal(refs[i], graph_res[i]) for i in range(len(prompts))))
