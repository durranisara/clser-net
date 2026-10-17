import torch
import torch.nn as nn
from .clate import CLATE
from .epra import EPRA

class CLSERNet(nn.Module):
    def __init__(self, d_f=103, d_h=256, num_emotions=4,
                 dropout=0.3, temperature=0.1):
        super().__init__()
        self.clate = CLATE(d_f, d_h)
        d = 2 * d_h
        self.epra = EPRA(d, num_emotions, temperature)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d, num_emotions)

    def forward(self, F):
        H = self.clate(F)                 # (B, T, d)
        z, a = self.epra(H)               # (B, d), (B, T, C)
        logits = self.classifier(self.dropout(z))
        return logits, z, a, self.epra.prototypes
