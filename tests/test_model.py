import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine.model import GPT2

tok = AutoTokenizer.from_pretrained("gpt2")
hf = AutoModelForCausalLM.from_pretrained("gpt2").eval()
ours = GPT2.from_pretrained("gpt2")

ids = tok("The best poker players know that", return_tensors="pt").input_ids

with torch.no_grad():
    ref_logits = hf(ids).logits
    our_logits = ours(ids)
print("logits max diff:", (our_logits - ref_logits).abs().max().item())
assert torch.allclose(our_logits, ref_logits, atol=1e-3), "Logits mismatch!"
print("PASS: logits")

with torch.no_grad():
    ref = hf.generate(ids, max_new_tokens=50, do_sample=False,
                      pad_token_id=tok.eos_token_id)
    out = ids
    for _ in range(50):
        next_id = ours(out)[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_id], dim=1)
assert torch.equal(ref, out), "Generation mismatch!"
print("PASS: generation")
print(tok.decode(out[0]))
