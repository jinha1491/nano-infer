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

system = ("You are a poker coach. Give short, practical answers. Explain the reasoning "
          "behind each decision in terms of position, stack sizes, ranges, and pot odds. "
          "Avoid jargon unless you define it first.\n\nQuestion: ")
questions = [
    "When should I three-bet light?",
    "How do I play top pair on a wet board?",
    "Is limping ever correct?",
    "How big should my river bluffs be?",
    "When should I three-bet light?",
]
lengths = [25, 15, 30, 10, 25]
prompts = [tok(system + q, return_tensors="pt").input_ids.to(dev) for q in questions]
print("prompt lengths:", [p.size(1) for p in prompts])
refs = [generate_kv(model, p, n)[0].cpu() for p, n in zip(prompts, lengths)]

with torch.no_grad():
    p = prompts[0]
    full, _ = model(p)
    _, past = model(p[:, :32])
    part, _ = model(p[:, 32:], past)
    print(f"max logit diff, cached-prefix prefill vs full prefill: "
          f"{(full[:, 32:] - part).abs().max().item():.2e}")

def run(n_slots, n_blocks, prefix, mode="dynamic"):
    reqs = [Request(i, p, n) for i, (p, n) in enumerate(zip(prompts, lengths))]
    stats = {}
    res = generate_continuous(model, reqs, n_slots, max_len=256, mode=mode, stats=stats,
                              n_blocks=n_blocks, block_size=16, prefix_cache=prefix)
    for i in range(len(prompts)):
        assert torch.equal(refs[i], res[i]), f"slots={n_slots} prefix={prefix} req {i}: mismatch!"
    return res, stats

def report(label, st):
    print(f"PASS: {label}: computed {st['prefill_tokens']} prompt tokens, "
          f"reused {st['cached_tokens']} from cache")

_, st = run(2, 128, prefix=False)
report("prefix cache off, 2 slots", st)
_, st = run(2, 128, prefix=True)
report("prefix cache on, 2 slots", st)
_, st = run(1, 128, prefix=True)
report("prefix cache on, 1 slot (blocks reused after the owner finished)", st)
_, st = run(2, 14, prefix=True)
report("prefix cache on, small pool (cached blocks get evicted)", st)

if dev == "cuda":
    run(2, 128, prefix=True, mode="static")
    run(2, 128, prefix=True, mode="graph")
    print("PASS: prefix cache with static shapes and CUDA graphs")
