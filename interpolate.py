import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, RectBivariateSpline


def plot(f, xnew, ynew):
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
   znew = f(xnew, ynew)
   ax1.plot(x, z[0, :], 'ro-', xnew, znew[0, :], 'b-')
   im = ax2.imshow(znew)
   plt.colorbar(im, ax=ax2)
   plt.show(block=True)
   return znew

def interpolate(fi,al,v,N):
    x = fi
    y = al
    xx, yy = np.meshgrid(x, y)
    z = v
    f = interp2d(x, y, z, kind='cubic')

    xnew = np.linspace(np.min(x), np.max(x), N)
    ynew = np.linspace(np.min(y), np.max(y), N)
    znew_i = plot(f, xnew, ynew)
