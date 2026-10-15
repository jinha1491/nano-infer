import sys, os, json, time, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import sync

assert torch.cuda.is_available(), "run this on an NVIDIA GPU"
device = torch.device("cuda")
N = 200
TEXTS = [
    "The best poker players know that",
    "In a tournament, the most important skill is",
    "The history of the Roman Empire began when",
    "To train a neural network, you first need",
]

tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
tok.padding_side = "left"
hf = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()

single = tok(TEXTS[:1], return_tensors="pt").to(device)
batch = tok(TEXTS * 4, return_tensors="pt", padding=True).to(device)


def bench(fn, n_tokens, repeats=5):
    fn()
    rates = []
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        rates.append(n_tokens / (time.perf_counter() - start))
    return {"median": statistics.median(rates), "min": min(rates), "max": max(rates)}


@torch.no_grad()
def generate(enc, **kwargs):
    hf.generate(**enc, max_new_tokens=N, min_new_tokens=N, do_sample=False,
                pad_token_id=tok.eos_token_id, **kwargs)


results = {}
for name, kwargs in [("hf_default", {}), ("hf_static_cache", {"cache_implementation": "static"})]:
    for label, enc, B in [("single", single, 1), ("batch16", batch, 16)]:
        try:
            results[f"{name}_{label}"] = bench(lambda: generate(enc, **kwargs), B * N)
        except Exception as e:
            results[f"{name}_{label}"] = {"error": f"{type(e).__name__}: {e}"[:200]}

with open(os.path.join(ROOT, "results", "results.json")) as f:
    ours = json.load(f)

def show(r):
    if "error" in r:
        return "failed: " + r["error"]
    return f"{r['median']:.0f} ({r['min']:.0f}-{r['max']:.0f})"

print("\n### HuggingFace generate() vs this engine, GPT-2 small, T4, fp32, 200 new tokens\n")
print("| engine | batch 1 (tok/s) | batch 16 (tok/s) |")
print("|---|---|---|")
for name in ["hf_default", "hf_static_cache"]:
    print(f"| {name} | {show(results[name + '_single'])} | {show(results[name + '_batch16'])} |")
for name in ["cuda_graph", "compiled", "compiled_int8"]:
    print(f"| ours: {name} | {show(ours['single'][name])} | {show(ours['batch']['16'][name])} |")

with open(os.path.join(ROOT, "results", "compare_hf.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results/compare_hf.json")
