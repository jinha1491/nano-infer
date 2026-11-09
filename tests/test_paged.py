import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from engine.qwen import Qwen
from engine.kv import generate_kv
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
    "Bluffing works best when",
    "The river card changed everything because",
]
lengths = [20, 40, 5, 33, 1, 17]
prompts = [tok(t, return_tensors="pt").input_ids.to(dev) for t in texts]
refs = [generate_kv(model, p, n)[0].cpu() for p, n in zip(prompts, lengths)]

def run(n_slots, mode="dynamic", **kw):
    reqs = [Request(i, p, n) for i, (p, n) in enumerate(zip(prompts, lengths))]
    stats = {}
    res = generate_continuous(model, reqs, n_slots, max_len=128, mode=mode, stats=stats, **kw)
    return res, stats

def check(res, label):
    for i in range(len(prompts)):
        assert torch.equal(refs[i], res[i]), f"{label} req {i}: mismatch!"
    print(f"PASS: {label}")

for bs in [4, 16]:
    for n_slots in [2, 3]:
        res, st = run(n_slots, n_blocks=64, block_size=bs)
        check(res, f"paged block_size={bs}, n_slots={n_slots} (peak {st['peak_blocks']} blocks)")

res, st = run(4, n_blocks=8, block_size=8)
check(res, f"tight pool of 7 blocks, requests wait for free blocks (peak {st['peak_blocks']})")

if dev == "cuda":
    static_res, _ = run(3, "static", n_blocks=64, block_size=16)
    graph_res, _ = run(3, "graph", n_blocks=64, block_size=16)
    for i in range(len(prompts)):
        assert torch.equal(static_res[i], graph_res[i]), f"paged graph {i}: mismatch!"
    print("PASS: paged CUDA graph matches paged static")
    print("paged graph also matches single-sequence KV cache:",
          all(torch.equal(refs[i], graph_res[i]) for i in range(len(prompts))))
