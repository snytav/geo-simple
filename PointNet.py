import numpy as np
import torch
import torch.nn as nn


class PointNet(nn.Module):
    def __init__(self,fi,al,rhs,N):
         super(PointNet, self).__init__()
         self.fi  = fi
         self.al  = al
         self.rhs = rhs
         fc1 = nn.Linear(1,N)
         self.fc1 = fc1
         self.fc2 = nn.Linear(N,1)

    def forward(self):
        x = torch.tensor([self.fi,self.al])
        x = self.fc1(x)
        x = torch.nn.Sigmoid(x)
        x = self.fc2(x)
        return x

    def loss(self):
        y = self.forward()
        t = torch.abs(y-self.rhs)
        return t

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

    ml2d = nn.ModuleList(ml)

    return ml2d

if __name__ == '__main__':
    # pn = PointNet(0,1,10.0,10)
    # ppn = nn.ModuleList([PointNet(0,i,10.0,10) for i in range(10)])
    # qq = 0
    ml2 = create_point_net_array(np.random.randn(2,2),
                                 np.random.randn(2,2),
                                 np.random.randn(2,2),10)
    qq = 0
