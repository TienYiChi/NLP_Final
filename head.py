import torch
import torch.nn as nn

class My_linear(nn.Module):
    def __init__(self, hidden_dim=768):
        super(My_linear, self).__init__()
        self.hidden_dim=hidden_dim
        self.linear_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.actv = nn.Tanh()
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        self.linear_2 = nn.Linear(self.hidden_dim, 1)



    def forward(self,x): # x of shape (batch_size, hidden_dim)
        l1 = self.linear_1(x) #(B,1)
        a1 = self.actv(l1)
        b1 = self.bn(a1)
        l2 = self.linear_2(b1)
        # x = x.squeeze(dim=1) # (B,1) -> (B,)
        # x = torch.sigmoid(x)
        return l2 # (B,)
