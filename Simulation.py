"""
Comment Gérer Optimalement, et de Manière Durable, une Forêt
"""

# Packages
import numpy as np
import math as m
import seaborn as sns
from matplotlib import pyplot as plt

# from aenum import Enum
from dataclasses import dataclass
import TimeExecution.Timer as Timer



__author__ = ["BARTKOWIAK Valentin", "ELANKUMARAN Vishwa", "PUEL Marius"]
__copyright__ = "Copyright 2020, Projet d'Expertise en Statistique et Probabilités"
__version__ = "1.0"
__maintainer__ = ["BARTKOWIAK Valentin", "ELANKUMARAN Vishwa", "PUEL Marius"]
__email__ = ["valentin.bartkowiak@etu.u-bordeaux.fr",
             "vishwa.elankumaran@etu.u-bordeaux.fr",
             "marius.puel@etu.u-bordeaux.fr"]
__status__ = "in development"


@dataclass
class Process:
    """Process"""

    theta: float    # Convergence speed
    l: float        # Asymptotic limit
    h: float        # Time Step
    y0: float       # Starting point
    sig: float

    def b(self, x: float) -> float:
        return -self.theta * (x - self.l)

    def sigma(self, x: float) -> float:
        return self.sig

    def calcProcess(self, x: float, eps: float) -> float:
        return x + self.b(x) * self.h + self.sigma(x) * np.sqrt(self.h) * eps

    
    def calcProcessWithIteration(self, n: int) -> np.ndarray:
        process = np.zeros(n)
        eps = np.random.normal(0, 1, n)
        process[0] = self.y0

        return np.array([
            self.calcProcess(process[i - 1], eps[i]) for i in range(1, n)])


def b(x, theta, l):
    return(-theta * (x - l))


def sigma(x):
    return 0.05

 # On se fixe une stratégie dans laquelle on coupe la forêt une fois le seuil y*= 0.8*l dépassé. On coupe jusqu'à y0


def reward(x):
    return np.sqrt(x)



# def simu():
#     # Simple simulation de la foret
#     n = 1000000
#     eps = np.random.normal(0, 1, n)

#     X = np.zeros(n)
#     X[0] = 0
#     h = 1 / n
#     for i in range(1, len(X)):
#         X[i] = X[i - 1] + b(X[i - 1], theta=5, l=0.1) * h + \
#             m.sqrt(h) * sigma(X[i - 1]) * eps[i]

#     plt.plot(np.linspace(0, n-1, n), X)
#     plt.show()
# simu()
# Simu d'une foret avec une limite
@Timer.timeit
def t():
    l = 0.1
    y0 = 0
    y_star = 0.8 * l
    theta = 5

    T = 1000000
    eps = np.random.normal(0, 1, T)


    Y = [[y0]]
    # X_tho = np.zeros(T)
    h = 1 / T

    for i in range(1, T):
        if Y[-1][-1] >= y_star:
            Y.append([y0])

            #X_tho[i] = Y[i - 1]
        else:
            Y[-1].append(Y[-1][-1] + b(Y[-1][-1], theta, l) *
                         h + m.sqrt(h) * sigma(Y[-1][-1]) * eps[i])
    return
    c = [len(i) for i in Y]
    c.insert(0, 0)

    c = [sum(c[:i]) for i in range(1, len(c))]
    c.append(T)
    print(c)

    for i in range(1, len(c)):
        plt.plot((range(c[i - 1], c[i])), Y[i - 1], color='b')
    plt.plot(np.linspace(0, T - 1, T), np.ones(T) * y_star, color='r')
    # plt.show()

# # ########
# # np.seterr(divide='ignore', invalid='ignore')
# # nb_coupe = np.count_nonzero(X_tho)
# # M_empirique = np.cumsum(reward(X_tho)) / nb_coupe
# # G = np.divide(M_empirique, np.linspace(0, T - 1, T))
# # plt.plot(np.linspace(0, T - 1, T), G)
# # plt.show()

# # ######
# # l = 0.1
# # y0 = 0.1 * l
# # N = 1000
# # y_star = np.linspace(0.1 * l, 2 * l, N)
# # theta = 40

# # T = 2000
# # eps = np.random.normal(0, 1, T)

# # G = []
# # Y = np.zeros(T)

# # Y[0] = y0
# # h = 1 / T
# # for y_s in y_star:
# #     X_tho = np.zeros(T)
# #     for i in range(1, len(Y)):
# #         if Y[i - 1] >= y_s:
# #             Y[i] = y0
# #             X_tho[i] = Y[i - 1]
# #         else:
# #             Y[i] = Y[i - 1] + b(Y[i - 1], theta, l) * h + \
# #                 m.sqrt(h) * sigma(Y[i - 1]) * eps[i]

# #     nb_coupe = np.count_nonzero(X_tho)
# #     M_empirique = np.cumsum(reward(X_tho)) / nb_coupe
# #     G.append(M_empirique[-1] / T)

# # plt.plot(y_star, G)
# # plt.xlim(np.min(y_star), np.max(y_star))
# # plt.show()

# # # ≤

# # # Optimisation dans le cas déterministe
# # l = 0.1
# # y0 = 0.1 * l
# # N = 1000
# # y_star = np.linspace(0.1 * l, l, N)
# # theta = 40

# # T = 2000
# # eps = np.random.normal(0, 10, T)

# # G = []
# # Y = np.zeros(T)

# # Y[0] = y0
# # h = 1 / T
# # for y_s in y_star:
# #     X_tho = np.zeros(T)
# #     for i in range(1, len(Y)):
# #         if Y[i - 1] >= y_s:
# #             Y[i] = y0
# #             X_tho[i] = Y[i - 1]
# #         else:
# #             # + m.sqrt(h)*sigma(Y[i-1])*eps[i]
# #             Y[i] = Y[i - 1] + b(Y[i - 1], theta, l) * h
# #     nb_coupe = np.count_nonzero(X_tho)
# #     M_empirique = np.cumsum(reward(X_tho)) / nb_coupe
# #     G.append(M_empirique[-1] / T)

# # plt.plot(y_star, G)
# # plt.show()
