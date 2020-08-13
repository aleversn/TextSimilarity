import torch
import torch.nn as nn

class Ensemble(nn.Module):

    def __init__(self):
        super(Ensemble, self).__init__()

        self.w1 = torch.Tensor([0.5]).cuda()1
        self.b1 = torch.Tensor([0]).cuda()
        self.w2 = torch.Tensor([0.5]).cuda()
        self.b2 = torch.Tensor([0]).cuda()
    
    def forward(self, X1, X2):
        out = (self.w1 * X1 + self.b1 + self.w2 * X2 + self.b2) / 2

        return out