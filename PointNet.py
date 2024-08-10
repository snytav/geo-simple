import numpy as np
import torch
import torch.nn as nn


class PointNet(nn.Module):
    def __init__(self,fi,al,rhs,N):
         super(PointNet, self).__init__()
         self.fi  = fi
         self.al  = al
         self.rhs = rhs
         fc1 = nn.Linear(2,N)
         self.fc1 = fc1
         self.fc2 = nn.Linear(N,1)

    def forward(self):
        x = torch.tensor([self.fi,self.al])
        x = x.float()
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x

    def loss(self):
        y = self.forward()
        t = torch.abs(y-self.rhs)
        return t

    def train(self):
        optim = torch.optim.Adam(self.parameters(),lr=0.01)
        n = 0
        lf = torch.ones(1)*1.0e3
        while lf.item() > 1e-2:
            optim.zero_grad()
            lf = self.loss()
            lf.backward()
            optim.step()
            #print(n,lf.item())
            n =  n+1

        qq = 0
        return lf.item()


def create_point_net_array(fi2D,al2D,rhs2D,N):
    Ni,Nk = fi2D.shape
    module_list = []
    for i in range(Ni):
        list_along_k = []
        for k in range(Nk):
            fi = fi2D[i][k]
            al = al2D[i][k]
            rhs = rhs2D[i][k]
            pn = PointNet(fi, al,rhs, N)
            list_along_k.append(pn)
        ml = nn.ModuleList(list_along_k)
        module_list.append(ml)

    ml2d = nn.ModuleList(module_list)

    return ml2d

if __name__ == '__main__':
    # pn = PointNet(0,1,10.0,10)
    # ppn = nn.ModuleList([PointNet(0,i,10.0,10) for i in range(10)])
    # qq = 0
    ml2 = create_point_net_array(np.random.randn(2,2),
                                 np.random.randn(2,2),
                                 np.random.randn(2,2),10)
    qq = 0
