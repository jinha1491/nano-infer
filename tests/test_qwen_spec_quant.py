import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
if not torch.cuda.is_available():
    print("SKIP: needs a GPU")
    sys.exit(0)

from transformers import AutoTokenizer
from engine.qwen import Qwen
from engine.kv import generate_kv
from engine.batch import generate_static
from engine.speculative import generate_speculative
from engine.quant import quantize_model, model_bytes

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
ids = tok("The best poker players know that", return_tensors="pt").input_ids.cuda()

target = Qwen.from_pretrained("Qwen/Qwen3-1.7B").cuda()
draft = Qwen.from_pretrained("Qwen/Qwen3-0.6B").cuda()
N = 40
ref = generate_kv(target, ids, N)
for mode in ["tuple", "dynamic", "static", "graph"]:
    for n_draft in [1, 3, 5]:
        stats = {}
        out = generate_speculative(target, draft, ids, N, n_draft=n_draft,
                                   stats=stats, draft_mode=mode, max_len=128)
        assert torch.equal(ref, out), f"speculative {mode}, n_draft={n_draft}: mismatch!"
    rate = stats["accepted"] / stats["proposed"]
    print(f"PASS: speculative draft_mode={mode}, n_draft 1/3/5 "
          f"(acceptance at n_draft=5: {rate:.0%})")
del target

texts = ["The best poker players know that", "Hi",
         "When you are holding pocket aces before the flop, you should usually"]
prompts = [tok(t, return_tensors="pt").input_ids.cuda() for t in texts]

for dtype in [torch.float32, torch.float16]:
    full = Qwen.from_pretrained("Qwen/Qwen3-0.6B", dtype=dtype).cuda()
    q = quantize_model(Qwen.from_pretrained("Qwen/Qwen3-0.6B", dtype=dtype)).cuda()
    label = str(dtype).split(".")[-1]
    with torch.no_grad():
        lf, lq = full(prompts[2])[0].float(), q(prompts[2])[0].float()
    top1 = (lf.argmax(-1) == lq.argmax(-1)).float().mean().item()
    print(f"int8 vs {label}: max logit diff {(lf - lq).abs().max().item():.3f}, "
          f"top-1 agreement {top1:.0%}, weights {model_bytes(full) / 2**20:.0f} MB "
          f"-> {model_bytes(q) / 2**20:.0f} MB")
    assert top1 >= 0.8, "int8 model disagrees too often"

    graph = generate_static(q, prompts, 30, max_len=128, mode="graph")
    comp = generate_static(q, prompts, 30, max_len=128, mode="compiled")
    same = all(torch.equal(a, b) for a, b in zip(graph, comp))
    print(f"PASS: int8 {label} runs in graph and compiled modes; outputs identical: {same}")
    print("  sample:", repr(tok.decode(graph[0][prompts[0].size(1):])))
    del full, q
