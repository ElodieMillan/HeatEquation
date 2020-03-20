# 2 dimentions of Heat equation,
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
import matplotlib.animation as anim
from scipy import meshgrid, linspace, ones

from fonction_matrice_dirichlet_2D import matrix_dirichlet_2D
from tqdm import tqdm


class AnimatedGif:
    def __init__(self, size=(1000, 1000)):
        self.fig = plt.figure()
        self.size = size
        self.fig.set_size_inches(size[0] / 100 * 1.5, size[1] / 100 * 1.5)
        ax = self.fig.add_axes([0.16, 0.05, 0.85, 0.85], frameon=True, aspect="auto")
        ax.set_xlabel("Position x/e")
        ax.set_ylabel("Position y/e")

        self.images = []

    def add(self, image, vmax=1, label=""):
        plt_im = plt.imshow(
            image, cmap="rainbow", origin="lower", vmin=0, vmax=vmax, animated=True
        )
        plt_txt = plt.text(self.size[0] // 2.3, self.size[1] + 10, label, fontsize=14)
        self.images.append([plt_im, plt_txt])

    def save(self, filename):
        plt.colorbar()
        animation = anim.ArtistAnimation(self.fig, self.images)
        animation.save(filename, writer="imagemagick", fps=10)


# paramétres de l'expérience

a = 0.0
b = 20.0  # petri disch 20 fois plus grand que e (mettre 5 ou 10 pour les test c'est suffisant)
L = b - a

tini = 0.0
tfin = 1  # Pour les tests mettre 0.2 s
T = tfin - tini

e = 1.0  # diametre de nutriment (on se met en umité de cette distance car pb memoire)

# initialisations
Omega = np.array([a, b, tini, tfin])  # espace de 0 à 1 et temps de 0 à 1

nx = 300.0
nt = 1000.0

Nx = int(nx)
Nt = int(nt)

dx = L / nx
dt = T / nt

# critère de stabilité
if dt > (dx * dx) * (2.0):
    print("Attention schema instable : dt doit etre inferier a : ", (dx * dx) * (2.0))
    sys.exit(1)

# - Initial conditions and bondaries conditions
Ubords = 0.0
Uini = 1
NA = int(Nx / 2)
NB = NA + int((e * Nx) / L)

##################################
##### initialisation of x, y #####

x = np.arange(a, b + dx, dx)
y = np.arange(a, b + dx, dx)

t = np.arange(tini, tfin + dt, dt)

# Finale liste in fonction of space (Nx+1)(Nx+1) and time Nt+1
flatten_U = np.zeros([(Nx + 1) * (Nx + 1), Nt + 1])

# just for space initialisation with square because it's more easy
U = np.zeros((Nx + 1, Nx + 1))

# intitial conditions
U[NA : NB + 1, NA : NB + 1] = Uini

# put the initiales conditions as list in List
flatten_U[:, 0] = U.flatten()

# fig1 = plt.figure()
# plt.plot(flatten_U[:, 0])

# X, Y = np.meshgrid(x, y)
# plt.pcolormesh(X, Y, U[:, :], cmap=plt.get_cmap("rainbow"))
# plt.colorbar()
# plt.title("Concentration initiales (t = 0)")
# plt.xlabel("Position x/e")
# plt.ylabel("Position y/e")
# plt.show()
# plt.savefig("../Images/Conditions_initiales_2D_square.png")


###########################################
####### PDE resolution  #######

A = matrix_dirichlet_2D(Omega, Nx, Nt)

# As gif
animated_gif = AnimatedGif(size=(Nx + 1, Nx + 1))


for j in tqdm(range(0, Nt)):
    flatten_U[:, j + 1] = A * flatten_U[:, j]
    if j % 10 == 0:
        animated_gif.add(
            np.reshape(flatten_U[:, j + 1], (Nx + 1, Nx + 1)),
            vmax=Uini,
            label="t={0:.2f}".format(j * dt),
        )

animated_gif.save(
    "../Images/2D_diffusion_dirichlet_explicite_Uini1_tfin_0p5s_Nt_1000_nx_300.gif"
)
#########


# np.save("result_2D_diffusion_dirichlet_Uini_Uini1_tfin_0p5s_Nt_1000_nx_300, flatten_U)

# plt.figure()
# plt.pcolormesh(
#    X, Y, np.reshape(flatten_U[:, -1], (Nx + 1, Nx + 1)), cmap=plt.get_cmap("Purples"),
# )
# plt.colorbar()
# plt.title("Concentration initiales (t = 0)")
# plt.xlabel("Position x")
# plt.ylabel("Position y")
# plt.show()

#    plt.savefig("../Images/Initial_conditions_2D{}.png".format(j))
