import numpy as np
import torch
from sklearn.metrics import mean_absolute_percentage_error


def get_v_tilde(self):
    Ni, Nk = self.v2D.shape
    v_tilde = np.zeros((Ni, Nk))
    # self.pointwise_MAPE = np.zeros((Ni,Nk))
    for i in range(Ni):
        for k in range(Nk):
            fi = self.fi2D[i][k]
            al = self.al2D[i][k]
            t = torch.Tensor([fi, al]).cuda()
            y = self.forward(t)
            v_tilde[i][k] = y.cpu().detach().numpy()

    return v_tilde

def print_result(self,fname):
    Ni, Nk = self.v2D.shape
    v_tilde = np.zeros((Ni, Nk))
    # self.pointwise_MAPE = np.zeros((Ni,Nk))

    v_tilde = self.get_v_tilde()
    file2 = open(fname, "w+")
    for i in range(Ni):
        for k in range(Nk):
            fi = self.fi2D[i][k]
            al = self.al2D[i][k]
            mape = np.abs(v_tilde[i][k]-self.v2D[i][k])/np.abs(self.v2D[i][k])
            file2.write('fi '+'{:10.5f}'.format(fi)+' al '+'{:10.5f}'.format(al) +
                        ' v_tilde '+'{:15.5e}'.format(v_tilde[i][k])+
                        ' v ' + '{:15.5e}'.format(self.v2D[i][k]) +
                        ' mape ' +'{:10.3e}'.format(mape)+
                        '\n')

    file2.close()
    qq = 0


def MAPE(self):

    v_tilde = self.get_v_tilde()


    self.pointwise_MAPE = np.divide(np.abs(self.v2D-v_tilde),np.abs(self.v2D))
    np.savetxt('mape.txt', self.pointwise_MAPE, fmt='%15.5e')

    mape = mean_absolute_percentage_error(self.v2D,v_tilde)
    return mape
