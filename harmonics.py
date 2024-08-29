import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from laplace2D import laplace2D

def make_small_debug_file(fi2D,al2D,v2D,rhs2D):
    N = 80
    # return fi2D, al2D, v2D, rhs2D, True
    fi2D_10  = fi2D[:N,:N]
    al2D_10  = al2D[:N, :N]
    v2D_10   = v2D[:N, :N]
    rhs2D_10 = rhs2D[:N, :N]
    val = np.zeros((N*N,3))
    val[:,0] = fi2D_10.reshape(N*N)
    val[:,1] = al2D_10.reshape(N*N)
    val[:,2] = v2D_10.reshape(N*N)
    return fi2D_10, al2D_10, v2D_10,rhs2D_10, True

class HarmNet(nn.Module):
    def __init__(self,N,fi,al,v,koef):
        super(HarmNet,self).__init__()
        self.fi = fi
        self.al = al
        self.v = v


        N_fi = np.unique(self.fi).shape[0]
        N_al = np.unique(self.al).shape[0]
        self.v2D = self.v.reshape(N_al, N_fi).numpy()
        self.rhs = laplace2D(self.v2D)
        self.fi2D = self.fi.reshape(N_al, N_fi)
        self.al2D = self.al.reshape(N_al, N_fi)

        # temporarily reduce size for debug purpose
        self.fi2D, self.al2D, self.v2D,self.rhs, self.debug_mode = make_small_debug_file(self.fi2D, self.al2D, self.v2D,self.rhs)
        self.koef = koef.reshape(1,koef.shape[0]*koef.shape[1])
        self.N = N
        self.koef = torch.from_numpy(self.koef).float()
        fc1 = nn.Linear(self.koef.shape[1],N)
        fc2 = nn.Linear(N, self.fi2D.shape[0]*self.fi2D.shape[1])
        self.fc1 = fc1
        self.fc2 = fc2
    def forward(self,x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x

    def train(self):

        optim = torch.optim.Adam(self.parameters(),lr=0.01)

        lf = torch.sum(torch.abs(torch.from_numpy(self.rhs)-self.forward(self.koef).reshape(self.fi2D.shape[0],self.fi2D.shape[1])))

        n = 0
        while lf.item() > 1.5:
            optim.zero_grad()
            lf = torch.sum(torch.abs(torch.from_numpy(self.rhs)-self.forward(self.koef).reshape(self.fi2D.shape[0],self.fi2D.shape[1])))
            lf.backward()
            optim.step()
            print(n,lf.item())
            n = n+1

        mape_torch = torch.mean(torch.divide(torch.abs(torch.from_numpy(self.rhs) - self.forward(self.koef).reshape(self.fi2D.shape[0], self.fi2D.shape[1])),torch.abs(torch.from_numpy(self.rhs))))
        a = self.rhs
        a_tilde = self.forward(self.koef).reshape(self.fi2D.shape[0], self.fi2D.shape[1]).detach().numpy()

        from sklearn.metrics import mean_absolute_percentage_error
        mape = mean_absolute_percentage_error(a,a_tilde)
        qq = 0

if __name__ == '__main__':
    crd = np.loadtxt('testgrid.grid')
    N_fi = np.unique(crd[:, 0]).shape[0]
    N_al = np.unique(crd[:,1]).shape[0]
    df = df_user_key_word_org = pd.read_csv("new",
                                   sep="\s+|;|:",
                                   engine="python")
    # net = GeoNet(10)
    v = df['P_mod'].values
    fi = df['fi'].values
    lb = df['lb'].values
    # t = np.concatenate((fi.reshape(fi.shape[0], 1), lb.reshape(fi.shape[0], 1)), axis=1)
    # t = torch.from_numpy(t)
    # vt = net.forward(t.float())
    # optimizer = torch.optim.Adam(net.parameters(),lr = 0.01)
    # lf = torch.ones(1)*1e9
    v_torch = torch.from_numpy(v)
    # v2D = v_torch.reshape(N_al,N_fi)

    from read_harmonics import read_gfc
    koef = read_gfc()

    #from PDE import PDEnet
    pde  = HarmNet(10,fi,lb,v_torch,koef)
    y = pde.forward(pde.koef)
    pde.train()



    qq = 0
