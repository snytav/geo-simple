import torch
from torch.autograd.functional import jacobian
from torch.autograd.functional import hessian

Lx = 90

Ly = 190.5

def f_Lx(self,x):
    return torch.ones(1,requires_grad=True)
def f_x_0(self,x):
    return -torch.ones(1,requires_grad=True)
def f_Ly(self,x):
    return 2.0*torch.ones(1,requires_grad=True)
def f_y_0(self,x):
    return -2.0*torch.ones(1,requires_grad=True)

def A(self,x,net_out):
    if x[0] == Lx:
       return self.f_Lx(x[1])
    else:
       if x[0] == 0.0:
          return self.f_x_0(x[1])
    if x[1] == Ly:
       return self.f_Ly(x[0])
    else:
       if x[1] == 0.0:
          return self.f_y_0(x[0])
    return net_out


def psy_trial(self,x, net_out):
    return self.A(x,net_out) # + x[0] * (Lx - x[0]) * x[1] * (Ly - x[1]) * net_out

def psy_trial1(self,x, net_out):
    return x[0] * (Lx - x[0]) * x[1] * (Ly - x[1]) * net_out

def psy_trial2(self,x, net_out):
    return x[0] * (Lx - x[0]) * x[1] * (Ly - x[1]) #* net_out




def loss_pointwise(self,xi,yi,i,k):
    input_point = torch.tensor([xi, yi])
    input_point = input_point.float()
    input_point.requires_grad = True
    net_out = self.forward(input_point)[0]

    net_out_jacobian = jacobian(self.forward, input_point, create_graph=True)
    net_out_hessian = hessian(self.forward, input_point, create_graph=True)

    psy_t = self.psy_trial(input_point, self.forward(input_point))
    psy_t_jacobian = jacobian(self.psy_trial, inputs=(input_point, self.forward(input_point)), create_graph=True)
    # psy_t_jacobian1 = jacobian(self.psy_trial1, inputs=(input_point, self.forward(input_point)), create_graph=True)
    # psy_t_jacobian2 = jacobian(self.psy_trial2, inputs=(input_point, self.forward(input_point)), create_graph=True)
    psy_t_hessian = hessian(self.psy_trial, inputs=(input_point, self.forward(input_point)), create_graph=True)
    #psy_t_hessian = hessian(psy_trial1, inputs=(input_point, pde.forward(input_point)), create_graph=True)

    gradient_of_trial_dx = psy_t_jacobian[0][0][0]

    #if np.abs(dpsy_dx[n] - gradient_of_trial_dx.item()) > 1e-3:
       # qq = 0
    gradient_of_trial_dy = psy_t_jacobian[0][0][1]
    # if np.abs(dpsy_dy[n] - gradient_of_trial_dy.item()) > 1e-3:
    #     qq = 0

    gradient_of_trial_d2x = psy_t_hessian[0][0]
    gradient_of_trial_d2y = psy_t_hessian[1][1]

    err_sqr = ((gradient_of_trial_dx + gradient_of_trial_dy) - self.v2D[i][k]) ** 2
    #loss_sum += err_sqr

    return err_sqr

def loss(self):

    Ni,Nk = self.fi2D.shape
    lf = 0.0
    for i in range(Ni):
        for k in range(Nk):
            from diff import loss_pointwise
            fi = self.fi2D[i][k]
            al = self.al2D[i][k]
            t_pois = self.loss_pointwise(fi,al,i,k)
            lf += t_pois
    return lf