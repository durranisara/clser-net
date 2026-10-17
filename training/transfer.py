import torch
import torch.nn as nn

BLOCK_ORDER = ["feature", "clate", "epra", "classifier"]  # low -> high
L = len(BLOCK_ORDER)

def block_of(param_name: str) -> str:
    if param_name.startswith("clate"):   return "clate"
    if param_name.startswith("epra"):    return "epra"
    if param_name.startswith("classifier"): return "classifier"
    return "feature"   # non-parametric in paper; included for completeness

def discriminative_lr(eta_base, gamma, block):
    l = BLOCK_ORDER.index(block) + 1          # 1-based
    return eta_base * (gamma ** (L - l))

def set_trainable(model, trainable_blocks):
    for n, p in model.named_parameters():
        p.requires_grad = block_of(n) in trainable_blocks

def build_optimizer(model, eta_base, gamma, trainable_blocks):
    groups = {b: [] for b in BLOCK_ORDER}
    for n, p in model.named_parameters():
        if p.requires_grad:
            groups[block_of(n)].append(p)
    params = []
    for b in BLOCK_ORDER:
        if groups[b]:
            params.append({"params": groups[b],
                           "lr": discriminative_lr(eta_base, gamma, b)})
    return torch.optim.Adam(params, weight_decay=1e-5)
