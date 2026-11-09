import sys, os, json, time, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from transformers import AutoTokenizer
from utils import sync
from engine.qwen import Qwen
from engine.continuous import Request, generate_continuous

assert torch.cuda.is_available(), "run this on an NVIDIA GPU"
device = torch.device("cuda")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = Qwen.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16).to(device)

system = ("You are a poker coach. Give short, practical answers. Explain each decision in "
          "terms of position, stack sizes, ranges, and pot odds. Define jargon the first "
          "time you use it. Prefer concrete examples over general advice. ") * 30 + "\n\nQuestion: "
questions = [f"How should I play hand number {i} from the button?" for i in range(32)]
prompts = [tok(system + q, return_tensors="pt").input_ids.to(device) for q in questions]
sys_len = len(tok(system).input_ids)
NEW = 16

results = {}
for on in [False, True]:
    stats, times = {}, []
    for rep in range(4):
        reqs = [Request(i, p, NEW) for i, p in enumerate(prompts)]
        sync(device)
        start = time.perf_counter()
        generate_continuous(model, reqs, 8, max_len=2048, stats=stats, n_blocks=1024,
                            block_size=16, prefix_cache=on)
        sync(device)
        if rep > 0:
            times.append(time.perf_counter() - start)
    t = statistics.median(times)
    results["on" if on else "off"] = {
        "seconds": t, "tok_per_s": len(prompts) * NEW / t,
        "prefill_tokens": stats["prefill_tokens"], "cached_tokens": stats["cached_tokens"],
    }

print(f"\n### Prefix caching, 32 requests sharing a {sys_len}-token system prompt, "
      f"{NEW} new tokens each, Qwen3-0.6B fp16\n")
print("| prefix cache | total time (s) | useful tok/s | prompt tokens computed | reused |")
print("|---|---|---|---|---|")
for name, r in results.items():
    print(f"| {name} | {r['seconds']:.2f} | {r['tok_per_s']:.0f} | "
          f"{r['prefill_tokens']} | {r['cached_tokens']} |")
print(f"speedup: {results['off']['seconds'] / results['on']['seconds']:.2f}x")

with open(os.path.join(ROOT, "results", "prefix_long.json"), "w") as f:
    json.dump(results, f, indent=2)
