import sys, os, json, time, random, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch._dynamo.config as dynamo_config
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import sync
from engine.qwen import Qwen
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
FP16 = torch.float16
SMALL, LARGE = "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"
tok = AutoTokenizer.from_pretrained(SMALL)
TEXTS = [
    "The best poker players know that",
    "In a tournament, the most important skill is",
    "The history of the Roman Empire began when",
    "To train a neural network, you first need",
]
pool = [tok(t, return_tensors="pt").input_ids.to(device) for t in TEXTS]
results = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "dtype": "fp16"}


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
    if "error" in r:
        return "failed"
    return f"{r['median']:.0f} ({r['min']:.0f}-{r['max']:.0f})"


def table(title, header, rows):
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")


# A. Single sequence vs HuggingFace
N = 200
model = Qwen.from_pretrained(SMALL, dtype=FP16).to(device)
model_int8 = quantize_model(Qwen.from_pretrained(SMALL, dtype=FP16)).to(device)
p = pool[0]
single = {
    "kv_cache": lambda: generate_kv(model, p, N),
    "cuda_graph": lambda: generate_static(model, [p], N, mode="graph"),
    "compiled": lambda: generate_static(model, [p], N, mode="compiled"),
    "compiled_int8": lambda: generate_static(model_int8, [p], N, mode="compiled"),
}
results["single"] = {name: bench(fn, N) for name, fn in single.items()}

hf = AutoModelForCausalLM.from_pretrained(SMALL, dtype=FP16).to(device).eval()
for name, kw in [("hf_default", {}), ("hf_static_cache", {"cache_implementation": "static"})]:
    def run_hf(kw=kw):
        with torch.no_grad():
            hf.generate(p, max_new_tokens=N, min_new_tokens=N, do_sample=False,
                        pad_token_id=tok.eos_token_id, **kw)
    try:
        results["single"][name] = bench(run_hf, N)
    except Exception as e:
        results["single"][name] = {"error": f"{type(e).__name__}: {e}"[:200]}
del hf
torch.cuda.empty_cache()

base = results["single"]["hf_default"].get("median")
table("Single sequence, Qwen3-0.6B fp16, 200 new tokens (tok/s, median of 5, range)",
      ["engine", "tok/s", "vs HF default"],
      [(n, fmt(r), f"{r['median'] / base:.2f}x" if base and "median" in r else "-")
       for n, r in results["single"].items()])
results["weights_mb"] = {"fp16": model_bytes(model) / 2**20, "int8": model_bytes(model_int8) / 2**20}

# B. Batched decode
configs = [("cuda_graph", model, "graph"), ("compiled", model, "compiled"),
           ("compiled_int8", model_int8, "compiled")]
results["batch"] = {}
for B in [1, 8, 16]:
    prompts = [pool[i % len(pool)] for i in range(B)]
    results["batch"][str(B)] = {
        name: bench(lambda: generate_static(m, prompts, N, mode=mode), B * N, 3)
        for name, m, mode in configs
    }
table("Batched decode, Qwen3-0.6B fp16, 200 new tokens per sequence (total tok/s)",
      ["batch"] + [c[0] for c in configs],
      [(B, *(fmt(results["batch"][B][c[0]]) for c in configs)) for B in results["batch"]])
del configs, single, model_int8
torch.cuda.empty_cache()

# C. Paged vs slot cache at the same KV memory budget
N_REQ, MAX_LEN, SLOT_SLOTS, BLOCK = 64, 512, 8, 16
budget = SLOT_SLOTS * MAX_LEN
rng = random.Random(0)
lengths = [rng.randint(16, 256) for _ in range(N_REQ)]
req_prompts = [pool[i % len(pool)] for i in range(N_REQ)]
useful = sum(lengths)
serving = {}


def serve(label, **kw):
    stats = {}
    def run():
        reqs = [Request(i, req_prompts[i], lengths[i]) for i in range(N_REQ)]
        generate_continuous(model, reqs, max_len=MAX_LEN, stats=stats, **kw)
    r = bench(run, useful, 3)
    r["avg_concurrency"] = stats["busy_slot_steps"] / stats["decode_steps"]
    r["peak_blocks"] = stats.get("peak_blocks")
    serving[label] = r


serve("slot_cache", n_slots=SLOT_SLOTS)
serve("paged", n_slots=32, n_blocks=budget // BLOCK + 1, block_size=BLOCK)
results["serving"] = serving
base = serving["slot_cache"]["median"]
table(f"Serving {N_REQ} requests (16-256 new tokens), same KV budget of {budget} tokens",
      ["cache", "useful tok/s", "avg concurrent requests", "vs slot cache"],
      [(n, fmt(r), f"{r['avg_concurrency']:.1f}", f"{r['median'] / base:.2f}x")
       for n, r in serving.items()])

# D. Prefix caching with a shared system prompt
system = ("You are a poker coach. Give short, practical answers. Explain each decision in "
          "terms of position, stack sizes, ranges, and pot odds. Define jargon the first "
          "time you use it. Prefer concrete examples over general advice. ") * 6 + "\n\nQuestion: "
questions = [f"How should I play hand number {i} from the button?" for i in range(32)]
sp_prompts = [tok(system + q, return_tensors="pt").input_ids.to(device) for q in questions]
SP_NEW = 16
prefix = {}
for on in [False, True]:
    stats = {}
    def run(on=on):
        reqs = [Request(i, sp_prompts[i], SP_NEW) for i in range(len(sp_prompts))]
        generate_continuous(model, reqs, 8, max_len=512, stats=stats, n_blocks=512,
                            block_size=BLOCK, prefix_cache=on)
    r = bench(run, len(sp_prompts) * SP_NEW, 3)
    r["prefill_tokens"] = stats["prefill_tokens"]
    r["cached_tokens"] = stats["cached_tokens"]
    prefix["on" if on else "off"] = r
results["prefix"] = prefix
sys_len = len(tok(system).input_ids)
table(f"Prefix caching, 32 requests sharing a {sys_len}-token system prompt, 16 new tokens each",
      ["prefix cache", "useful tok/s", "prompt tokens computed", "reused"],
      [(n, fmt(r), r["prefill_tokens"], r["cached_tokens"]) for n, r in prefix.items()])
print(f"speedup: {prefix['on']['median'] / prefix['off']['median']:.2f}x")
del model
torch.cuda.empty_cache()

# E. Speculative decoding
target = Qwen.from_pretrained(LARGE, dtype=FP16).to(device)
draft = Qwen.from_pretrained(SMALL, dtype=FP16).to(device)
spec = {"target_alone": bench(lambda: generate_kv(target, p, N), N, 3)}
for mode in ["tuple", "graph"]:
    for k in [3, 5]:
        stats = {}
        r = bench(lambda: generate_speculative(target, draft, p, N, n_draft=k, stats=stats,
                                               draft_mode=mode, max_len=512), N, 3)
        r["acceptance"] = stats["accepted"] / stats["proposed"]
        spec[f"{mode}_k{k}"] = r
results["speculative"] = spec
base = spec["target_alone"]["median"]
table("Speculative decoding, Qwen3-1.7B target + Qwen3-0.6B draft, fp16, 200 tokens",
      ["config", "tok/s", "acceptance", "vs target alone"],
      [(n, fmt(r), f"{r['acceptance']:.0%}" if "acceptance" in r else "-",
        f"{r['median'] / base:.2f}x") for n, r in spec.items()])

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
with open(os.path.join(ROOT, "results", "qwen_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results/qwen_results.json")
