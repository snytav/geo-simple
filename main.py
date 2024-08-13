import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from laplace2D import laplace2D


def poisson2D(pd):

    d2x = pd[2:, 1:-1] + pd[:-2, 1:-1] - 2.0 * pd[1:-1, 1:-1]
    d2y = pd[1:-1,2:]  + pd[1:-1,:-2]  - 2.0 * pd[1:-1, 1:-1]

    return d2x,d2y


class GeoNet(nn.Module):

    def __init__(self,N):
        super(GeoNet,self).__init__()
        fc1 = nn.Linear(2,N)
        self.fc1 = fc1
        self.act1 = torch.sigmoid
        self.fc2 = nn.Linear(N,1)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

def loss_function(net,df):

    loss = 0.0
    t = torch.zeros(2)

    fi = df['fi'].values
    lb = df['lb'].values
    v =  df['P_mod'].values

    fi = torch.from_numpy(fi)
    lb = torch.from_numpy(lb)
    v = torch.from_numpy(v)
    fi.requires_grad = True
    lb.requires_grad = True
    v.requires_grad  = True

    for x,y,v in zip(fi,lb,v):
        t = torch.tensor([x,y]).float()
        vt = net.forward(t)
        loss += torch.pow(vt - v,2.0)
    return (loss/df.shape[0])


def MAPE_loss(vt,v_torch):
    s = 0.0
    for v1,v2 in zip(vt,v_torch):
        s += torch.abs(v1-v2)/torch.abs(v2)

    s /= vt.shape[0]
    return s


def read_train_set(fname):
    data = np.loadtxt(fname)
    col_names = 'm,n,ca,r0,sa,r1,K_vbv[n;m],P1'
    col_names = col_names.split(',')
    df = pd.DataFrame(data,columns=col_names)

    return df






def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
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



    from PDE import PDEnet
    pde  = PDEnet(10,fi,lb,v_torch,True)
    lf = pde.loss()
    pde.train()
    pde.train_points()


    v = pde.get_v(86.25,173.25)
    i,k = pde.get_ik(86.25,173.25)

    x = torch.tensor([fi[1],lb[1]])
    y = pde.forward(x)
    lp = pde.laplace1D_numerical(86.25,173.25)
    pde.laplace_pointwise()




    n = 0
    while lf.item() > 1.0001:
        optimizer.zero_grad()
        vt = net.forward(t.float())


        lf = MAPE_loss(vt,v_torch)
        #torch.mean(torch.divide(torch.abs(v_torch-vt),torch.abs(v_torch))))
        #lf = loss_function(net,df)
        lf.backward(retain_graph=True)
        optimizer.step()


        from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error
        #v_torch = torch.from_numpy(v)
        mape = mean_absolute_percentage_error(v,vt.detach().numpy())
        mae = mean_absolute_error(v, vt.detach().numpy())




        print(n,'{:e}'.format(lf.item()),mape,mae)
        n = n+1





    qq = 0

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
