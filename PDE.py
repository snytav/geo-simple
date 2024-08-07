import torch
import torch.nn as nn

class PDEnet(nn.Module):
    def __init__(self,N):
        super(PDEnet,self).__init__(N)
        self.N = N
        fc1 = nn.Linear(2,self.N)
        fc2 = nn.Linear(self.N, 1)

    def forward(self,x):
        x = x.reshape(1, 2)
        y = self.fc1(x)
        y = torch.sigmoid(y)
        y = self.fc2(y.reshape(1, self.N))
        return y

