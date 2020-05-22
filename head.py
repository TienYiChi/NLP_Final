import torch
import torch.nn as nn

class My_linear(nn.Module):
    def __init__(self, hidden_dim=768):
        super(My_linear, self).__init__()
        self.hidden_dim=hidden_dim
        self.cls_linear = nn.Linear(self.hidden_dim, 1)


    def forward(self,x): # x of shape (batch_size, hidden_dim)
        x = self.cls_linear(x) #(B,1)
        x = x.squeeze(dim=1) # (B,1) -> (B,)
        x = torch.sigmoid(x)
        return x # (B,)
