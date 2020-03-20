# Matrix sparse function 2D
# 2019, University of Bordeaux
## by Élodie MILLAN

import numpy as np
import scipy.sparse as sparse  # Algèbre linéaire creuse
import matplotlib.pyplot as plt  # Pour les graphiques


def matrix_H(dx, dt, N):
    """Matrix intern for the main diagonal.
        In : - dx discretization of space.
             - dt discretization of time
             - N size of matrix
        Out : return the matrix of the main diagonal of the 2D matrix."""

    # Parameters
    dx2 = dx * dx
    Nx = N - 1  # last element of the N = Nx + 1 list

    # Matrix coeficients
    lamda = dt / dx2

    # initialisation of 3 diagonals by 0
    diags = np.zeros((3, N))

    # Diagonal 0 ----> Nx + 1 terms
    diags[1, :] = 1 - 4 * lamda
    diags[1, 0] = 0  # fist term zero
    diags[1, Nx] = 0  # last term zero

    # Diagonal -1 ----> Nx terms
    diags[0, :] = lamda
    diags[0, Nx - 1] = 0  # last term zero

    # Diagonal +1 ----> Nx terms
    diags[2, :] = lamda
    diags[2, 1] = 0.0  # first term zero

    # Construction of sparse matrix 3 diagonals
    A = sparse.spdiags(diags, [-1, 0, 1], N, N, format="csr")

    return A


def matrix_I(dx, dt, N):
    """Matrix intern for main matrix (diagonal - 1) and (diagonal + 1).
        In : - dx discretization of space.
             - dt discretization of time
             - N size of matrix
        Out : return the matrix of the diagonal-1 and diagonal+1
              of the 2D matrix."""

    # Parameters
    dx2 = dx * dx
    Nx = N - 1  # last element of the N = Nx + 1 list

    # Coef dans la matrice
    lamda = dt / dx2

    # initialisation matrix 3 diagonals by 0 (size 3 * Nx+1)
    diags = np.zeros((3, N))

    # Diagonale 0
    diags[1, :] = lamda
    diags[1, 0] = 0  # first term zero
    diags[1, Nx] = 0  # last term zero

    # Construction matrix with 1 diagonal
    A = sparse.spdiags(diags, [-1, 0, 1], N, N, format="csr")

    return A


def matrix_dirichlet_2D(Omega, Nx, Nt):
    """Returns the matrix that discretizes the mapping u --> -Laplacian(u)
on the domain Omega = [SpaceMin, SpaceMax, tmin, tmax] split into Nx parts
in x and Ny parts in y (Nx = Ny). The final matrix is a scipy.sparse CSR matrix.
It is of size (Nx+1)*(Nx+1)."""

    dx = (Omega[1] - Omega[0]) / Nx
    dx2 = dx * dx

    dt = (Omega[3] - Omega[2]) / Nt

    # Matrix of size N
    N = (1 + Nx) * (1 + Nx)  # we choose Nx = Ny

    A = sparse.csr_matrix((0, 0))

    for i in range(Nx + 1):

        col = sparse.csr_matrix((0, 0))

        for j in range(Nx + 1):
            if j == 0 or j == Nx:
                # Put a zero matrix
                col = sparse.hstack([col, sparse.csr_matrix((Nx + 1, Nx + 1))])
                continue

            if j == i:
                # put the matrix_H
                col = sparse.hstack([col, matrix_H(dx, dt, Nx + 1)])
                continue

            if j == i - 1 or j == i + 1:
                # put the matrix_I
                col = sparse.hstack([col, matrix_I(dx, dt, Nx + 1)])
                continue

            col = sparse.hstack([col, sparse.csr_matrix((Nx + 1, Nx + 1))])

        A = sparse.hstack([A, col.transpose()], format="csr")

    return A


###### Plot of the matrix to verify its construction ######

Omega = np.array([0, 20, 0, 0.5])
Nx = 500
Nt = 1400

dx = (Omega[1] - Omega[0]) / Nx
dt = (Omega[3] - Omega[2]) / Nt


A = matrix_dirichlet_2D(Omega, Nx, Nt)
plt.spy(A)

# plt.savefig("../Images/Matrice_dirichlet_2D.png")
plt.show()
