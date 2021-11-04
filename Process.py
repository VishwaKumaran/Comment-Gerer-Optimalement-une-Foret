"""
Le processus Xk
"""

# Packages
import numpy as np

# from aenum import Enum
from dataclasses import dataclass
from pympler import asizeof

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
    __slots__ = ["theta", "l", "h", "y0", "sig"]
    theta: float    # Convergence speed
    l: float        # Asymptotic limit
    h: float        # Time Step
    y0: float       # Starting point
    sig: float

    def b(self, x: float) -> float:
        """
            Dynamique de la foret

            Parameter
            ---------
                x   -> Temps/pas de temps

            Return
            ------
                Dynamique
        """
        return -self.theta * (x - self.l)

    def sigma(self, x: float) -> float:
        return self.sig

    def calcProcess(self, x: float, eps: float) -> float:
        """
            Calcul du processus au temps x

            Parameter
            ---------
                x   -> Temps/pas de temps
                eps -> Epsilon : bruit

            Return
            ------
                Calcul du processus au temps x
        """
        return x + self.b(x) * self.h + self.sigma(x) * np.sqrt(self.h) * eps

    @Timer.timeit
    def calcProcessWithIteration(self, n: int) -> np.ndarray:
        """
            Calcul du processus pendant n iteration.

            Parameter
            ---------
                n   -> iteration

            Return
            ------
                Le processus pour n iteration
        """
        process: np.ndarray = np.zeros(n)
        eps: np.ndarray = np.random.normal(0, 1, n)
        process[0]: float = self.y0

        for i in range(1, n):
            process[i]: float = self.calcProcess(process[i - 1], eps[i])

        return process
