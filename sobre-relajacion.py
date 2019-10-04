"""
Resuelve la ec.:
nabla^2 phi = q

q = 2 ( 2  - x^2 - y^2)

Usando el metodo de sobre-relajacion
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


w = 1.8
Nx = Ny = 35
Lx = Ly = 2
h = Lx / (Nx - 1)
max_iter = 1000

def q(x, y):
    output = 2 * (2 - x**2 - y**2)
    return output

def una_iteracion(phi, h=h, q=q):
    """
    Implementa una iteracion de sobre-relajacion
    """
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            x_i = i * h - Lx/2
            y_j = j * h - Ly /2
            phi[i, j] = (1-w) * phi[i, j] + w/4 * (phi[i+1, j] + phi[i-1, j] +
                                                   phi[i, j+1] + phi[i, j-1] +
                                                   h**2 * q(x_i, y_j))


phi = np.zeros((Ny, Nx))
phi_prev = phi.copy()

una_iteracion(phi)

def no_ha_convergido(phi, phi_prev, tol=1e-3):
    not_zero = phi != 0
    diff_relat = (phi_prev[not_zero] - phi[not_zero]) / phi[not_zero]
    max_diff = np.fabs(diff_relat).max()
    if max_diff > tol:
        convergio = False
    else:
        convergio = True
    return not convergio

counter = 1
while no_ha_convergido(phi, phi_prev, tol=1e-9) and counter<max_iter:
    phi_prev = phi.copy()
    una_iteracion(phi)
    counter += 1

print("Counter = {}".format(counter))

print(phi)

x = np.linspace(-1, 1, Nx)
y = np.linspace(-1, 1, Ny)
X, Y = np.meshgrid(x, y)

fig = plt.figure(1)
fig.clf()

ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, phi, rstride=3, cstride=3)

fig.show()

fig2 = plt.figure(2)
fig2.clf()

ax2 = fig2.add_subplot(111)
ax2.imshow(phi, origin='bottom')
ax2.contour(phi, origin='lower', colors='k')

fig2.show()