import torch
import torch.nn as nn
import numpy as np
from laplace2D import laplace2D
import numpy as np

from PointNet import PointNet,create_point_net_array
import pandas as pd
from torch.autograd.functional import jacobian,hessian



# def make_small_debug_file(fi2D,al2D,v2D,rhs2D):
#     fi2D_10  = fi2D[:10,:10]
#     al2D_10  = al2D[:10, :10]
#     v2D_10   = v2D[:10, :10]
#     rhs2D_10 = rhs2D[:10, :10]
#     val = np.zeros((100,3))
#     val[:,0] = fi2D_10.reshape(100)
#     val[:,1] = al2D_10.reshape(100)
#     val[:,2] = v2D_10.reshape(100)
#     return fi2D_10, al2D_10, v2D_10,rhs2D_10, True

    # df = pd.DataFrame(val, columns=['fi', 'lb', 'P_mod'])
    # df.to_csv('10.csv',sep=' ')
#    return fi2D_10,al2D_10,v2D_10,True



class PDEnet(nn.Module):
    from diff import Ax,Ay,A,psy_trial,psy_trial1,psy_trial2,loss_pointwise
    from diff import loss,f_Lx,f_Ly,f_x_0,f_y_0,set_boundary_values
    from relative_errors import MAPE,get_v_tilde,print_result
    def make_small_debug_version(fi2D, al2D, v2D):
        fi2D_10 = fi2D[:10, :10]
        al2D_10 = al2D[:10, :10]
        v2D_10 = v2D[:10, :10]
        return fi2D_10,al2D_10,v2D_10,True
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

        self.Lx = np.max(self.fi2D)
        self.Ly = np.max(self.al2D)
        from surface import plot_density_surface
        fic = sorted(np.unique(self.fi))
        alc = sorted(np.unique(self.al))
        df = fic[1] - fic[0]
        af = alc[1] - alc[0]
        #plot_density_surface(self.v2D,self.v2D.shape,[df,af],'phi exact')
        # https://stackoverflow.com/questions/33259896/python-interpolation-2d-array-for-huge-arrays
        from interpolate import interpolate
        # (fi,al,v,N)
        u1 = interpolate(self.fi2D,self.al2D,self.v2D, 10)
        plot_density_surface(u1, u1.shape, [36.0, 18.0], 'phi interpolated')

        #temporarily reduce size for debug purpose
        from harmonics import make_small_debug_file
        self.fi2D, self.al2D, self.v2D, self.rhs, self.debug_mode = make_small_debug_file(self.fi2D, self.al2D,
                                                                                          self.v2D, self.rhs)

        from harmonics import HarmNet
        from read_harmonics import read_gfc
        koef = read_gfc()
        hnn = HarmNet(10, fi, al, v, koef)
        self.hnn = hnn

        #self.set_boundary_values(

        qq = 0





        fc1 = nn.Linear(2,self.N)
        fc2 = nn.Linear(self.N, 1)
        self.fc1 = fc1
        self.fc2 = fc2
        #self.pn2D = create_point_net_array(self.fi2D,self.al2D,self.rhs,10)



    def get_v(self,fi,lb):
        for f,l,v in zip(self.fi,self.al,self.v):
            if np.abs(f-fi) < 1e-6 and np.abs(l-lb) < 1e-6:
                return v

    def get_ik(self, fi, lb):
        i0,k0 = 0,0
        for (i, k), f in np.ndenumerate(self.fi2D.cpu().detach().numpy()):
            if np.abs(self.fi2D.cpu().detach().numpy()[i][k] - fi) < 1e-6 and np.abs(self.al2D.cpu().detach().numpy()[i][k] - lb) < 1e-6:
                i0, k0 = i, k
        return i0,k0

    def train(self):

        params = list(self.parameters()) + list(self.hnn.parameters())


        optim = torch.optim.Adam(params, lr=0.01)
        n = 0
        lf0 = 1e13
        hist = []
        lf = torch.ones(1) * lf0
        while lf.item()/lf0 > 1e-2:
              optim.zero_grad()
              lf = self.loss(n,lf)
              lf.backward()
              optim.step()
              hist.append(lf.item()/lf0)
              print(n,'{:15.5e}'.format(lf.item()/lf0))
              n = n + 1

        mape = self.MAPE()
        self.print_result('phi.txt')
        hist = np.array(hist)
        # import matplotlib.pyplot as plt
        # plt.plot(hist, marker='o', linestyle='solid')
        # plt.title('Neural network training')
        # plt.ylabel('loss function')
        # plt.xlabel('number of epoch')
        # plt.show(block=True)
        qq = 0

        

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

    def f(self, i, k):
        return self.v2D[i][k]

    def forward(self,x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x

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
                print(i,k)

        np.savetxt('learn_map.txt',learn_map.reshape(Ni*Nk),fmt='%15.5e')
        qq = 0
