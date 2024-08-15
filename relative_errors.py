import numpy as np
import torch
from sklearn.metrics import mean_absolute_percentage_error

def MAPE(self):

    Ni,Nk = self.v2D.shape
    v_tilde = np.zeros((Ni,Nk))
    # self.pointwise_MAPE = np.zeros((Ni,Nk))
    for i in range(Ni):
        for k in range(Nk):
            fi = self.fi2D[i][k]
            al = self.al2D[i][k]
            t = torch.Tensor([fi,al])
            y = self.forward(t)
            v_tilde[i][k] = y.detach().numpy()


    self.pointwise_MAPE = np.divide(np.abs(self.v2D-v_tilde),np.abs(self.v2D))
    np.savetxt('mape.txt', self.pointwise_MAPE, fmt='%15.5e')

    mape = mean_absolute_percentage_error(self.v2D,v_tilde)
    return mape
