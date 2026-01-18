
from fealpy.backend import bm
from lafemeit.data import *

from matplotlib import pyplot as plt


num = 2
clim = (-0.9, 0.9, -0.9, 0.9)
XLIM = (-1, 1)
YLIM = (-1, 1)

# model = random_gaussian2d_model(num, clim, (2, 6), (0.5, 1))
# model = random_unioned_circles_model(num)
model = random_unioned_triangles_model(num, clim, rlim=(0.2, 0.5), kind="equ")
print(model)

X = bm.linspace(XLIM[0], XLIM[1], 100)
Y = bm.linspace(YLIM[0], YLIM[1], 100)
X, Y = bm.meshgrid(X, Y)
points = bm.stack((X, Y), axis=-1)
phi = model.coef(points)
ai = plt.pcolormesh(X, Y, phi, cmap="jet")
plt.colorbar(ai)

plt.show()
