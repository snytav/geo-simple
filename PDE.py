import torch
import torch.nn as nn
import numpy as np

class PDEnet(nn.Module):
    def __init__(self,N,fi,al,v,draft,rhs):
        super(PDEnet,self).__init__()
        self.fi = fi
        self.al = al
        self.v  = v
        self.N  = N
        self.draft = draft
        self.rhs   = rhs
        N_fi = np.unique(self.fi).shape[0]
        N_al = np.unique(self.al).shape[0]
        self.v2D = self.v.reshape(N_al,N_fi).numpy()
        self.fi2D = self.fi.reshape(N_al, N_fi)
        self.al2D = self.al.reshape(N_al, N_fi)
        fc1 = nn.Linear(2,self.N)
        fc2 = nn.Linear(self.N, 1)



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

