import torch 
import torch.nn as nn 

class RMSNorm(nn.Module):
    def __init__(self, eps, size):
        super().__init__()
        self.eps = eps 
        self.size = size 
        self.weight = nn.Parameter(torch.ones(size)) # Adds to list of parameters for parameters() iterator.

    def forward(self, x):
        rms = torch.sqrt(torch.sum(torch.pow(x, 2)) / self.size + self.eps)
        rms_norm = (x / rms) * self.weight
        return rms_norm
