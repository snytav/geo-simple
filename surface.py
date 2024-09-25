import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot, cm

def plot_density_surface(f,Nx,dx,title):
    x = np.linspace(0,dx[0]*Nx[0],f.shape[0])
    y = np.linspace(0, dx[1] * Nx[1], f.shape[1])
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    f = ax.plot_surface(X, Y, f.T, rstride=1, cstride=1, cmap=cm.viridis,linewidth=0, antialiased=False)
    plt.colorbar(f)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show(block=True)

    qq = 0