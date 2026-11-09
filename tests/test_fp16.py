import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
if not torch.cuda.is_available():
    print("SKIP: fp16 test needs a GPU")
    sys.exit(0)

from transformers import AutoModelForCausalLM, AutoTokenizer
from engine.qwen import Qwen
from engine.kv import generate_kv
from engine.continuous import Request, generate_continuous

name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).cuda().eval()
ours = Qwen.from_pretrained(name, dtype=torch.float16).cuda()
ours32 = Qwen.from_pretrained(name).cuda()

def agree(a, b, start):
    n = min(a.size(-1), b.size(-1))
    same = (a[..., start:n] == b[..., start:n]).flatten()
    first = int(same.int().cumprod(0).sum())
    return same.float().mean().item(), first

text = "When you are holding pocket aces before the flop, you should usually"
ids = tok(text, return_tensors="pt").input_ids.cuda()
T = ids.size(1)

with torch.no_grad():
    l16 = ours(ids)[0].float()
    lhf = hf(ids).logits.float()
    l32 = ours32(ids)[0]
print(f"max logit diff, ours fp16 vs HF fp16: {(l16 - lhf).abs().max().item():.3f}")
print(f"max logit diff, ours fp16 vs ours fp32: {(l16 - l32).abs().max().item():.3f}")
top1 = (l16.argmax(-1) == lhf.argmax(-1)).float().mean().item()
print(f"top-1 agreement with HF fp16 across prompt positions: {top1:.0%}")
assert top1 >= 0.9, "fp16 model disagrees with HF too often"

ref_hf = hf.generate(ids, max_new_tokens=50, do_sample=False, pad_token_id=tok.eos_token_id)
out = generate_kv(ours, ids, 50)
frac, first = agree(out, ref_hf, T)
print(f"greedy vs HF fp16: {frac:.0%} of tokens match, identical for the first {first}")

texts = ["The best poker players know that", "In a tournament, the most important skill is",
         "Hi", text, "Bluffing works best when", "The river card changed everything because"]
lengths = [20, 40, 5, 33, 8, 17]
prompts = [tok(t, return_tensors="pt").input_ids.cuda() for t in texts]
refs = [generate_kv(ours, p, n)[0].cpu() for p, n in zip(prompts, lengths)]

for mode in ["dynamic", "graph"]:
    reqs = [Request(i, p, n) for i, (p, n) in enumerate(zip(prompts, lengths))]
    res = generate_continuous(ours, reqs, 3, max_len=128, mode=mode,
                              n_blocks=64, block_size=16, prefix_cache=True)
    fracs = [agree(res[i], refs[i], prompts[i].size(1))[0] for i in range(len(prompts))]
    print(f"paged {mode} fp16 vs single-sequence fp16: "
          f"{sum(fracs) / len(fracs):.0%} of tokens match on average")

a = ours.layers[0].self_attn
per_tok = lambda m: 2 * len(m.layers) * a.n_kv_head * a.d_head * m.embed_tokens.weight.element_size()
print(f"KV cache per token: fp32 {per_tok(ours32) / 1024:.0f} KB, fp16 {per_tok(ours) / 1024:.0f} KB")
