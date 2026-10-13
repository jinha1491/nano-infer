import torch
import torch.nn as nn
import torch.nn.functional as F


class Int8Linear(nn.Module):
    def __init__(self, linear, per_channel=True):
        super().__init__()
        w = linear.weight.data
        if per_channel:
            scale = w.abs().amax(dim=1, keepdim=True) / 127
        else:
            scale = w.abs().max().reshape(1, 1) / 127
        scale = scale.clamp(min=1e-8)
        w_q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
        self.register_buffer("w_q", w_q)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", linear.bias.data.clone())

    def forward(self, x):
        w = self.w_q.to(x.dtype) * self.scale
        return F.linear(x, w, self.bias)


def quantize_model(model, per_channel=True):
    for block in model.h:
        for parent, name in [(block.attn, "c_attn"), (block.attn, "c_proj"),
                             (block.mlp, "c_fc"), (block.mlp, "c_proj")]:
            setattr(parent, name, Int8Linear(getattr(parent, name), per_channel))
    return model


def model_bytes(model):
    tensors = list(model.parameters()) + list(model.buffers())
    return sum(t.numel() * t.element_size() for t in tensors)
