# 1 dimention of Heat equation,
# solve with a sparse matrix
# 2019, University of Bordeaux
## by Élodie MILLAN

import numpy as np
import scipy.sparse as sparse  # Algèbre linéaire creuse
import matplotlib.pyplot as plt  # Pour les graphiques
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D
import sys
import matplotlib.colors as colors
from scipy import meshgrid, linspace, ones

from fonction_matrice_dirichlet_1D import matrix_dirichlet_1D
from fonction_source import f


# paramétres de l'expérience

a = 0.0
b = 20.0  # box 20 time highter than nutrient
L = b - a

tini = 0.0
tfin = 1.0
T = tfin - tini  # times

e = 1.0  # nutrient diameter

# initialisations
Omega = np.array([a, b, tini, tfin])  # space and time

nx = 500.0
nt = 1400.0

Nx = int(nx)
Nt = int(nt)

dx = L / nx
dt = T / nt

# stability of schema
if dt > (dx * dx) * (2.0):
    print("Warning, instable schema : dt should be less than a : ", (dx * dx) * (2.0))
    sys.exit(1)

# - initiales condition and bondaries conditions
Ubords = 0.0  # bondaries
Uini = 1.0  # initiales
NA = int(Nx / 2)  # domaine of initiales condition
NB = NA + int((e * Nx) / L)

##################################
##### initialisation of x, y #####

x = np.arange(a, b + dx, dx)

t = np.arange(tini, tfin + dt, dt)

list_U = np.zeros([Nx + 1, Nt + 1])

# Construction of heaviside initiale condition between NA and NB
for i in range(NA, NB + 1):
    list_U[i, 0] = 1


fig1 = plt.figure()
plt.plot(x, list_U[:, 0])
plt.title("Conditions initiales (t = 0)")
plt.xlabel("Position")
plt.ylabel("Concentration addimentionné")
# plt.savefig("../Images/Initial_conditions_1D.png")


A = matrix_dirichlet_1D(Omega, Nx, Nt)


for j in range(0, Nt):
    list_U[:, j + 1] = spsolve(A, list_U[:, j])

A_inverse = np.linalg.inv(A.toarray())
print("sont elles identiques : {} ".format(np.mean(A_inverse - A)))

################################# PLOT ########################

X, T = np.meshgrid(x, t)

fig3 = plt.figure(figsize=(14, 8))
ax = fig3.add_subplot(111, projection="3d")
surf = ax.plot_surface(
    X, T, np.transpose(list_U), antialiased=False, cmap="twilight_shifted"
)
ax.set_xlabel("X/e", fontsize=16)
ax.set_ylabel("temps (s)", fontsize=16)
ax.set_zlabel("Concentration adimentionnée", fontsize=16)
ax.view_init(elev=15, azim=120)

norm = colors.Normalize(Ubords, Uini)
p = ax.plot_surface(X, T, np.transpose(list_U), cstride=1, linewidth=0, cmap="jet")
cb = fig3.colorbar(p, ax=ax)
# plt.show()

plt.savefig("../Images/1D_Dirichlet_explicit.png")
