import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator

def laplace2D(pd):
    d2x = pd[2:, 1:-1] + pd[:-2, 1:-1] - 2.0 * pd[1:-1, 1:-1]
    d2y = pd[1:-1, 2:] + pd[1:-1, :-2] - 2.0 * pd[1:-1, 1:-1]

    f = -(d2x+d2y)
    return f


if __name__ == '__main__':
    N = 10
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)

    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    X,Y = np.meshgrid(x,y)

    k1 = 1.0
    k2 = 1.0
    f = k1*X+k2*Y
    f = np.sin(2*np.pi*X)
    # surf = ax.plot_surface(X, Y, f, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    # plt.show(block=True)

    r = laplace2D(f)
    plt.figure()
    plt.plot(x, f[5, :], color='blue')
    plt.plot(x[1:-1], r[5, :], color='red')
    plt.show(block=True)

    #    plt.show(block=True)
    #plt.figure()
    # surf = ax.plot_surface(X[1:-1,1:-1], Y[1:-1,1:-1], r, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    #plt.show(block=True)
    qq = 0