import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
if not torch.cuda.is_available():
    print("SKIP: CUDA graphs need an NVIDIA GPU")
    sys.exit(0)

from transformers import AutoTokenizer
from engine.model import GPT2
from engine.kv import generate_kv
from engine.batch import generate_static
from engine.continuous import Request, generate_continuous

tok = AutoTokenizer.from_pretrained("gpt2")
model = GPT2.from_pretrained("gpt2").cuda()

texts = [
    "The best poker players know that",
    "In a tournament, the most important skill is",
    "Hi",
    "When you are holding pocket aces before the flop, you should usually",
]
prompts = [tok(t, return_tensors="pt").input_ids.cuda() for t in texts]
lengths = [20, 35, 5, 30]

static_out = generate_static(model, prompts, 20, max_len=128, mode="static")
graph_out = generate_static(model, prompts, 20, max_len=128, mode="graph")
for i in range(len(prompts)):
    assert torch.equal(static_out[i], graph_out[i]), f"static batch {i}: mismatch!"
print("PASS: graph matches static, static batching")

def run(mode):
    reqs = [Request(i, p, n) for i, (p, n) in enumerate(zip(prompts, lengths))]
    return generate_continuous(model, reqs, n_slots=2, max_len=128, mode=mode)

static_res, graph_res = run("static"), run("graph")
for i in range(len(prompts)):
    assert torch.equal(static_res[i], graph_res[i]), f"continuous {i}: mismatch!"
print("PASS: graph matches static, continuous batching")

kv_match = all(torch.equal(generate_kv(model, p, n)[0].cpu(), graph_res[i])
               for i, (p, n) in enumerate(zip(prompts, lengths)))
print("graph also matches single-sequence KV cache:", kv_match)
