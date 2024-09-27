import numpy as np
import matplotlib.pyplot as plt
import os
def interpolate(u,N,fi2D,al2D):
    u1 = np.zeros((u.shape[0],N))
    if not os.path.exists('interp_images'):
       os.mkdir('interp_images')

    os.chdir('interp_images')
    for i,y in enumerate(u):
        x = fi2D[:,0]
        x = np.unique(x)
        y = u[i,:]

        x1 = np.linspace(np.min(x), np.max(x), N)
        y1 = np.interp(x1, x, y)

        # plt.figure()
        # plt.plot(x,y,color='blue')
        # plt.plot(x1,y1,'o',color='red')
        # plt.xlabel('$\phi$')
        # plt.title('u[i,:], i = '+str(i))
        # plt.show()
        # plt.savefig('interpX_ui_'+str(i)+'.png')
        u1[i,:] = y1
        qq = 0

    u2 = np.zeros((N,N))
    for i,z in enumerate(u1.T):
        x = al2D[:,0]
        x = np.unique(x)
        y = u1[:,i]

        x2 = np.linspace(np.min(x), np.max(x), N)
        y2 = np.interp(x1, x, y)
        u2[i,:] = y2

        plt.figure()
        plt.plot(x,y,color='green')
        plt.plot(x2,y2,'o',color='red')
        plt.xlabel('$\lambda$')
        plt.title('u[i,:], i = '+str(i))
        plt.show()
        plt.savefig('interpY_ui_'+str(i)+'.png')

        qq = 0


    return u2

# https://numpy.org/devdocs/reference/generated/numpy.interp.html

    # from scipy import interpolate
    # f = interpolate.interp2d(fi2D, al2D, u, kind='cubic')
    # use linspace so your new range also goes from 0 to 3, with 8 intervals
    xmin = np.min(fi2D)
    xmax = np.max(fi2D)
    ymin = np.min(al2D)
    ymax = np.max(al2D)
    Xnew = np.linspace(xmin, xmax, N)
    Ynew = np.linspace(ymin, ymax, N)

    u1 = f(Xnew, Ynew)
    return u1
    # https://www.askpython.com/python-modules/numpy/bilinear-interpolation-python

    import numpy as np
    from scipy.interpolate import griddata
    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1], [2, 2], [1, 2], [0, 2]])
    x = fi2D[:, 0]
    x = np.unique(x)
    y = al2D[:, 0]
    y = np.unique(y)
    X,Y = np.meshgrid(x,y)

    values = np.array([1, 4, 7, 9, 3, 8, 2, 7])
    x = 0.5
    y = 0.5
    interpolated_value = griddata(points, values, (x, y), method='nearest')
    print("Interpolated value at ({x}, {y}): {interpolated_value}")
