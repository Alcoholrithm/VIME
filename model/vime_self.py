import torch
import torch.nn as nn

class VIMESelfSupervised(nn.Module):
    def __init__(self, encoder_dim):
        super().__init__()
        self.h = nn.Linear(encoder_dim, encoder_dim, bias=True)
        self.mask_output = nn.Linear(encoder_dim, encoder_dim, bias=True)
        self.feature_output = nn.Linear(encoder_dim, encoder_dim, bias=True)

    def forward(self, x):
        h = torch.relu(self.h(x))
        mask = torch.sigmoid(self.mask_output(h))
        feature = torch.sigmoid(self.feature_output(h))
        return mask, feature
