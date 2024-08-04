import numpy as np
import torch
import pandas as pd

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
    V  = np.loadtxt('new')


    qq = 0

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
