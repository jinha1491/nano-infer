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
        self.register_buffer("scale", scale.view(-1))
        bias = linear.bias.data.clone() if linear.bias is not None else None
        self.register_buffer("bias", bias)

    def forward(self, x):
        scale = self.scale.to(x.dtype)
        if x.dtype == torch.float32:
            y = F.linear(x, self.w_q.to(x.dtype)) * scale
        else:
            y = F.linear(x, self.w_q.to(x.dtype) * scale[:, None])
        return y + self.bias if self.bias is not None else y


def quantize_model(model, per_channel=True):
    blocks = model.layers if hasattr(model, "layers") else model.h
    for block in blocks:
        for parent in list(block.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, name, Int8Linear(child, per_channel))
    return model


def model_bytes(model):
    tensors = list(model.parameters()) + list(model.buffers())
    return sum(t.numel() * t.element_size() for t in tensors)
