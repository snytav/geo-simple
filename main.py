import numpy as np
import torch
import torch.nn as nn
import pandas as pd


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
    return loss


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
    net = GeoNet(10)
    optimizer = torch.optim.Adam(net.parameters(),lr = 0.1)
    lf = torch.ones(1)*1e4
    n = 0
    while lf.item() > 1.0:
        optimizer.zero_grad()
        lf = loss_function(net,df)
        lf.backward()
        optimizer.step()
        print(n,lf.item())
        n = n+1





    qq = 0

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
