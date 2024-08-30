def plot_result(self,mape_file):

    import matplotlib.pyplot as plt
    import numpy as np

    from matplotlib import cm
    from matplotlib.ticker import LinearLocator

    mape = np.loadtxt(mape_file)
    mape = mape.reshape(self.fi2D.shape[0],self.fi2D.shape[1])

    Ni, Nk = self.v2D.shape
    v_tilde = np.zeros((Ni, Nk))
    # self.pointwise_MAPE = np.zeros((Ni,Nk))
    for i in range(Ni):
        for k in range(Nk):
            if mape[i][k] < 0.9:
                mape[i][k] = 0.9
            if mape[i][k] > 1.1:
                mape[i][k] = 1.1

    from laplace2D import smooth2D
    mape = smooth2D(mape)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    X = self.fi2D.copy()
    Y = self.al2D.copy()
    Z = mape

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-10.0, 10.0)

    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel('широта')
    plt.ylabel('долгота')
    plt.title('Абсолютная ошибка в процентах')

    plt.show(block=True)
    qq = 0