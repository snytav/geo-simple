import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, RectBivariateSpline

x = np.arange(-5.01, 5.01, 0.25)
y = np.arange(-5.01, 7.51, 0.25)
xx, yy = np.meshgrid(x, y)
z = np.sin(xx**2 + 2.*yy**2)
f = interp2d(x, y, z, kind='cubic')

def plot(f, xnew, ynew):
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
   znew = f(xnew, ynew)
   ax1.plot(x, z[0, :], 'ro-', xnew, znew[0, :], 'b-')
   im = ax2.imshow(znew)
   plt.colorbar(im, ax=ax2)
   plt.show(block=True)
   return znew

xnew = np.arange(-5.01, 5.01, 1e-2)
ynew = np.arange(-5.01, 7.51, 1e-2)
znew_i = plot(f, xnew, ynew)
