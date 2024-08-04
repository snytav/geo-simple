import numpy as np
import torch
import torch.nn as nn
import pandas as pd


class GeoNet(nn.Module):

    def __init__(self,N):
        super.__init__(self)
        fc1 = nn.Linear(2,N)
        self.fc1 = fc1
        self.act1 = torch.sigmoid
        self.fc2 = nn.Linear(N,1)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)

def loss_function(net,df):

    loss = 0.0
    t = torch.zeros(2)

    N_fi = np.unique(df['fi'].values).shape[0]
    N_al = np.unique(df['al'].values).shape[0]
    vm = df['P_mod'].values
    vm = vm.reshape(N_fi,N_al)



    for i,x in enumerate(np.unique(df.values[:,0])):
        for j,y in enumerate(np.unique(df.values[:,1])):
            t[0] = x
            t[1] = y
            vt = net.forward(t)
            loss += torch.pow(vt - vm[i][j],2.0)


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
    N_fi = np.unique(df.values[:, 0]).shape[0]
    N_al = np.unique(df.values[:, 1]).shape[0]


    qq = 0

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
