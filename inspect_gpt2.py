from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

print("=== CONFIG ===")
c = model.config
print("layers:", c.n_layer, "| heads:", c.n_head, "| hidden size:", c.n_embd)
print("vocab size:", c.vocab_size, "| max positions:", c.n_positions)

print("\n=== ALL WEIGHTS (block 0 only, others repeat) ===")
for name, p in model.state_dict().items():
    if ".h." in name and not name.startswith("transformer.h.0."):
        continue
    print(f"{name:45s} {tuple(p.shape)}")

total = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total:,}")
