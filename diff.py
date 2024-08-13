import torch
from torch.autograd.functional import jacobian
from torch.autograd.functional import hessian

Lx = 90

Ly = 190.5

def f_Lx(x):
    return 1.0
def f_x_0(x):
    return -1.0
def f_Ly(x):
    return 2.0
def f_y_0(x):
    return -2.0

def A(x,net_out):
    if x[0] == Lx:
       return f_Lx(x[1])
    else:
       if x[0] == 0.0:
          return f_x_0(x[1])
    if x[1] == Ly:
       return f_Ly(x[0])
    else:
       if x[1] == 0.0:
          return f_y_0(x[0])
    return net_out


def psy_trial(x, net_out):
    return A(x) + x[0] * (Lx - x[0]) * x[1] * (Lt - x[1]) * net_out

def psy_trial1(x, net_out):
    return x[0] * (Lx - x[0]) * x[1] * (Lt - x[1]) * net_out

def psy_trial2(x, net_out):
    return x[0] * (Lx - x[0]) * x[1] * (Lt - x[1]) #* net_out

def f(pde,i,k):
    return pde.v2D[i][k]


def loss_pointwise(pde,xi,yi,i,k):
    input_point = torch.tensor([xi, yi])
    input_point = input_point.float()
    net_out = pde.forward(input_point)[0]

    net_out_jacobian = jacobian(pde.forward, input_point, create_graph=True)
    net_out_hessian = hessian(pde.forward, input_point, create_graph=True)

    psy_t = psy_trial(input_point, pde.forward(input_point))
    psy_t_jacobian = jacobian(psy_trial, inputs=(input_point, pde.forward(input_point)), create_graph=True)
    psy_t_jacobian1 = jacobian(psy_trial1, inputs=(input_point, pde.forward(input_point)), create_graph=True)
    psy_t_jacobian2 = jacobian(psy_trial2, inputs=(input_point, pde.forward(input_point)), create_graph=True)
    psy_t_hessian = hessian(psy_trial, inputs=(input_point, pde.forward(input_point)), create_graph=True)
    # psy_t_hessian = hessian(psy_trial1, inputs=(input_point, pde.forward(input_point)), create_graph=True)

    gradient_of_trial_dx = psy_t_jacobian[0][0][0][0]

    #if np.abs(dpsy_dx[n] - gradient_of_trial_dx.item()) > 1e-3:
       # qq = 0
    gradient_of_trial_dy = psy_t_jacobian[0][0][0][1]
    # if np.abs(dpsy_dy[n] - gradient_of_trial_dy.item()) > 1e-3:
    #     qq = 0

    gradient_of_trial_d2x = psy_t_hessian[0][0]
    gradient_of_trial_d2y = psy_t_hessian[1][1]

    err_sqr = ((gradient_of_trial_dx + gradient_of_trial_dy) - func(pde,)) ** 2
    #loss_sum += err_sqr

    return err_sqr