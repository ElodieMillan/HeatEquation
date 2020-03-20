# Matrix sparse function 1D
# 2019, University of Bordeaux
## by Élodie MILLAN

import numpy as np
import scipy.sparse as sparse  # Algèbre linéaire creuse
import matplotlib.pyplot as plt  # Pour les graphiques
import sys as sys


def matrix_dirichlet_1D(Omega, Nx, Nt):
    """Returns the matrix that discretizes the mapping u --> -Laplacian(u)
on the domain Omega = [xmin,xmax,ymin,ymax] split into Nx parts in x
and Ny parts in y. The final matrix is a scipy.sparse CSR matrix. It
is of size (Nx+1)*(Ny+1)."""

    # Parametres
    dx = (Omega[1] - Omega[0]) / Nx
    dx2 = dx * dx
    dt = (Omega[3] - Omega[2]) / Nt
    N = Nx + 1

    # Coef dans la matrice
    alpha = 1 + ((2 * dt) / dx2)  # diagonale principale
    print("La mémoire pour un nombre est de : {} bytes".format(sys.getsizeof(alpha)))
    beta = -(dt / dx2)  # diagonale -1 t +1
    print("La mémoire pour un nombre est de : {} bytes".format(sys.getsizeof(beta)))
    # initialisation matrice trigonal par des 0 de taille  3 * Nx+1
    diags = np.zeros((3, N))

    # Diagonale 0
    diags[1, :] = alpha

    # Diagonale -1
    diags[0, :] = beta  # en général
    # diags[0, 0] = 0.0  # bord gauche (premier terme nul)

    # Diagonale +1
    diags[2, :] = beta  # en général
    # diags[2, Nx] = 0.0  # bord droit (dernier terme nul)

    # Construction de la matrice creuse de u --> -Laplacien(u)
    A = sparse.spdiags(diags, [-1, 0, 1], (Nx + 1), (Nx + 1), format="csr")

    return A


### Plot of matrix to verify

Omega = np.array([0, 20, 0, 1.0])
Nx = 500
Nt = 1400


A = matrix_dirichlet_1D(Omega, Nx, Nt)
# print("La taille de cette matrice est de : {} bytes".format(sys.getsizeof(A)))
plt.spy(A)
plt.show()
# plt.savefig("../Images/Matrice_sparse_1D.png")
