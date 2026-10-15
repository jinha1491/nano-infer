import sys, os, json, time, random, math, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
import torch._dynamo.config as dynamo_config
from datasets import load_dataset
from transformers import AutoTokenizer
from utils import sync
from engine.model import GPT2
from engine.naive import generate_naive
from engine.kv import generate_kv
from engine.batch import generate_static
from engine.continuous import Request, generate_continuous
from engine.speculative import generate_speculative
from engine.quant import quantize_model, model_bytes

assert torch.cuda.is_available(), "run this on an NVIDIA GPU"
for key in ("recompile_limit", "cache_size_limit"):
    if hasattr(dynamo_config, key):
        setattr(dynamo_config, key, 32)

device = torch.device("cuda")
tok = AutoTokenizer.from_pretrained("gpt2")
TEXTS = [
    "The best poker players know that",
    "In a tournament, the most important skill is",
    "The history of the Roman Empire began when",
    "To train a neural network, you first need",
]
pool = [tok(t, return_tensors="pt").input_ids.to(device) for t in TEXTS]
results = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}


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


def fmt(r):
    return f"{r['median']:.0f} ({r['min']:.0f}-{r['max']:.0f})"


def table(title, header, rows):
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")


small = GPT2.from_pretrained("gpt2").to(device)
small_int8 = quantize_model(GPT2.from_pretrained("gpt2")).to(device)
N = 200

# A. Single sequence
p = pool[0]
single = {
    "naive": lambda: generate_naive(small, p, N),
    "kv_cache": lambda: generate_kv(small, p, N),
    "cuda_graph": lambda: generate_static(small, [p], N, mode="graph"),
    "compiled": lambda: generate_static(small, [p], N, mode="compiled"),
    "compiled_int8": lambda: generate_static(small_int8, [p], N, mode="compiled"),
}
results["single"] = {name: bench(fn, N) for name, fn in single.items()}
base = results["single"]["naive"]["median"]
table("Single sequence, GPT-2 small, 200 new tokens (tok/s, median of 5, range)",
      ["engine", "tok/s", "vs naive"],
      [(n, fmt(r), f"{r['median'] / base:.1f}x") for n, r in results["single"].items()])

# B. Batching
configs = [("dynamic", small, "dynamic"), ("cuda_graph", small, "graph"),
           ("compiled", small, "compiled"), ("compiled_int8", small_int8, "compiled")]
results["batch"] = {}
for B in [1, 8, 16]:
    prompts = [pool[i % len(pool)] for i in range(B)]
    results["batch"][str(B)] = {
        name: bench(lambda: generate_static(m, prompts, N, mode=mode), B * N)
        for name, m, mode in configs
    }
table("Batched decode, GPT-2 small, 200 new tokens per sequence (total tok/s)",
      ["batch"] + [c[0] for c in configs],
      [(B, *(fmt(results["batch"][B][c[0]]) for c in configs)) for B in results["batch"]])

# C. Serving workload: mixed output lengths
N_REQ, N_SLOTS = 32, 8
rng = random.Random(0)
lengths = [rng.randint(10, 200) for _ in range(N_REQ)]
req_prompts = [pool[i % len(pool)] for i in range(N_REQ)]
useful = sum(lengths)
groups = [range(i, min(i + N_SLOTS, N_REQ)) for i in range(0, N_REQ, N_SLOTS)]


def run_static_batching():
    for g in groups:
        generate_static(small, [req_prompts[j] for j in g], max(lengths[j] for j in g))


def run_continuous(mode):
    reqs = [Request(i, req_prompts[i], lengths[i]) for i in range(N_REQ)]
    generate_continuous(small, reqs, N_SLOTS, mode=mode)


results["serving"] = {"static_batching": bench(run_static_batching, useful, 3)}
for mode in ["dynamic", "graph", "compiled"]:
    results["serving"][f"continuous_{mode}"] = bench(lambda: run_continuous(mode), useful, 3)
base = results["serving"]["static_batching"]["median"]
table(f"Serving: {N_REQ} requests, 10-200 tokens each, {N_SLOTS} slots (useful tok/s)",
      ["scheduler", "tok/s", "vs static batching"],
      [(n, fmt(r), f"{r['median'] / base:.2f}x") for n, r in results["serving"].items()])

# D. Speculative decoding
large = GPT2.from_pretrained("gpt2-large").to(device)
NS, N_DRAFT = 100, 3


def run_target():
    for q in pool:
        generate_kv(large, q, NS)


results["speculative"] = {"target_alone": bench(run_target, len(pool) * NS, 3)}
for draft_mode in ["tuple", "graph", "compiled"]:
    totals = {}

    def run():
        totals.update(proposed=0, accepted=0, rounds=0)
        for q in pool:
            s = {}
            generate_speculative(large, small, q, NS, n_draft=N_DRAFT,
                                 stats=s, draft_mode=draft_mode)
            for k in totals:
                totals[k] += s[k]

    r = bench(run, len(pool) * NS, 3)
    r["acceptance"] = totals["accepted"] / totals["proposed"]
    r["tokens_per_round"] = len(pool) * (NS - 1) / totals["rounds"]
    results["speculative"][f"draft_{draft_mode}"] = r
base = results["speculative"]["target_alone"]["median"]
table(f"Speculative decoding, GPT-2 large target, GPT-2 small draft, n_draft={N_DRAFT}",
      ["config", "tok/s", "speedup", "acceptance", "tokens/round"],
      [(n, fmt(r), f"{r['median'] / base:.2f}x",
        f"{r['acceptance']:.0%}" if "acceptance" in r else "-",
        f"{r['tokens_per_round']:.2f}" if "tokens_per_round" in r else "-")
       for n, r in results["speculative"].items()])

# E. Quality and memory
text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
eval_ids = tok(text, return_tensors="pt").input_ids[:, :20480].to(device)


@torch.no_grad()
def perplexity(model, ids, ctx=1024):
    nll, count = 0.0, 0
    for i in range(0, ids.size(1) - 1, ctx):
        chunk = ids[:, i:i + ctx + 1]
        logits, _ = model(chunk[:, :-1])
        nll += F.cross_entropy(logits[0], chunk[0, 1:], reduction="sum").item()
        count += chunk.size(1) - 1
    return math.exp(nll / count)


per_tensor = quantize_model(GPT2.from_pretrained("gpt2"), per_channel=False).to(device)
results["quality"] = {
    name: {"memory_mb": model_bytes(m) / 1e6, "perplexity": perplexity(m, eval_ids)}
    for name, m in [("fp32", small), ("int8_per_tensor", per_tensor),
                    ("int8_per_channel", small_int8)]
}
table("Quality and memory, GPT-2 small, WikiText-2 (20,480 tokens)",
      ["weights", "memory (MB)", "perplexity"],
      [(n, f"{r['memory_mb']:.0f}", f"{r['perplexity']:.2f}")
       for n, r in results["quality"].items()])

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
with open(os.path.join(ROOT, "results", "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results/results.json")
