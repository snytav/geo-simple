import torch
import torch.nn as nn
import numpy as np

class PDEnet(nn.Module):
    def __init__(self,N,fi,al,v,draft):
        super(PDEnet,self).__init__()
        self.fi = fi
        self.al = al
        self.v  = v
        self.N  = N
        self.draft = draft
        fc1 = nn.Linear(2,self.N)
        fc2 = nn.Linear(self.N, 1)

    def get_v(self,fi,lb):
        for f,l,v in zip(self.fi,self.al,self.v):
            if np.abs(f-fi) < 1e-6 and np.abs(l-lb) < 1e-6:
                return v

    def get_previous(self,fi,lb):
        fi_uniq = np.unique(self.fi)
        i = np.where(fi_uniq == fi)
        i = i[0]
        fi_prev = fi_uniq[i-1]
        v_prev_fi = self.get_v(fi_prev,lb)

        lb_uniq = np.unique(self.al)
        k = np.where(lb_uniq == lb)
        k = k[0]
        lb_prev = lb_uniq[k-1]
        v_prev_lb = self.get_v(fi,lb_prev)
        return v_prev_fi,v_prev_lb
    def laplace1D_numerical(self,fi,lb):
        return


    def forward(self,x):
        if self.draft:
            fi = x[0]
            al = x[1]

        x = x.reshape(1, 2)
        y = self.fc1(x)
        y = torch.sigmoid(y)
        y = self.fc2(y.reshape(1, self.N))
        return y

