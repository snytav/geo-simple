import torch
import torch.nn as nn
import numpy as np
from laplace2D import laplace2D
from PointNet import PointNet,create_point_net_array

class PDEnet(nn.Module):

    def __init__(self,N,fi,al,v,draft):
        super(PDEnet,self).__init__()
        self.fi = fi
        self.al = al
        self.v  = v
        self.N  = N
        self.draft = draft
        N_fi = np.unique(self.fi).shape[0]
        N_al = np.unique(self.al).shape[0]
        self.v2D = self.v.reshape(N_al,N_fi).numpy()
        self.rhs = laplace2D(self.v2D)
        self.fi2D = self.fi.reshape(N_al, N_fi)
        self.al2D = self.al.reshape(N_al, N_fi)
        fc1 = nn.Linear(2,self.N)
        fc2 = nn.Linear(self.N, 1)
        self.pn2D = create_point_net_array(self.fi2D,self.al2D,self.rhs,10)



    def get_v(self,fi,lb):
        for f,l,v in zip(self.fi,self.al,self.v):
            if np.abs(f-fi) < 1e-6 and np.abs(l-lb) < 1e-6:
                return v

    def get_ik(self, fi, lb):
        for (i, k), f in np.ndenumerate(self.fi2D):
            if np.abs(self.fi2D[i][k] - fi) < 1e-6 and np.abs(self.al2D[i][k] - lb) < 1e-6:
                i0, k0 = i, k
        return i0,k0



    def laplace1D_numerical(self,p_fi, p_al):
        i, k = self.get_ik(p_fi, p_al)

        d2x = self.v2D[i-1][k] -2.0*self.v2D[i][k] + self.v2D[i+1][k]
        d2y = self.v2D[i][k-1] -2.0*self.v2D[i][k] + self.v2D[i][k+1]
        return [d2x,d2y]

    def laplace_pointwise(self):
        rhsX_p = np.zeros_like(self.rhsX)
        rhsY_p = np.zeros_like(self.rhsY)

        for (i,k),p in np.ndenumerate(self.fi2D):
            a = self.al2D[i][k]
            p2x,p2y = self.laplace1D_numerical(p,a)
            rhsX_p[i-1][k-1] = p2x
            rhsY_p[i-1][k-1] = p2y

    def forward(self,x):
        if self.draft:
            fi = x[0]
            al = x[1]
            i,k = self.get_ik(fi,al)
            return self.v2D[i][k]

        x = x.reshape(1, 2)
        y = self.fc1(x)
        y = torch.sigmoid(y)
        y = self.fc2(y.reshape(1, self.N))
        return y

    def train_points(self):
        Ni,Nk = self.fi2D.shape
        learn_map = np.zeros((Ni,Nk))
        for i in range(Ni):
            for k in range(Nk):
                pn = self.pn2D[i][k]
                t = pn.train()
                learn_map[i][k] = t

        qq = 0
