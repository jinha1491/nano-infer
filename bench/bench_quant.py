import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from utils import get_device, sync
from engine.model import GPT2
from engine.kv import generate_kv
from engine.quant import quantize_model, model_bytes

device = get_device()
print("device:", device)
tok = AutoTokenizer.from_pretrained("gpt2")

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


def tok_per_sec(model):
    p = tok("The best poker players know that", return_tensors="pt").input_ids.to(device)
    generate_kv(model, p, 10)
    sync(device)
    start = time.perf_counter()
    generate_kv(model, p, 100)
    sync(device)
    return 100 / (time.perf_counter() - start)


configs = {
    "fp32": lambda m: m,
    "int8 per-tensor": lambda m: quantize_model(m, per_channel=False),
    "int8 per-channel": lambda m: quantize_model(m, per_channel=True),
}

print(f"\n{'config':>17} | {'memory':>8} | {'perplexity':>10} | {'tok/s':>6}")
for name, quantize in configs.items():
    model = quantize(GPT2.from_pretrained("gpt2")).to(device)
    mb = model_bytes(model) / 1e6
    ppl = perplexity(model, eval_ids)
    tps = tok_per_sec(model)
    print(f"{name:>17} | {mb:6.0f}MB | {ppl:10.2f} | {tps:6.1f}")
