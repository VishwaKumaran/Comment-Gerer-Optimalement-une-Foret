"""
Comment Gérer Optimalement, et de Manière Durable, une Forêt
"""

# Packages
import math
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
# from aenum import Enum
# from dataclasses import dataclass
# from pympler import asizeof
# asizeof.asizeof(foret)

# import TimeExecution.Timer as Timer


__author__ = ["BARTKOWIAK Valentin", "ELANKUMARAN Vishwa", "PUEL Marius"]
__copyright__ = "Copyright 2020, Projet d'Expertise en Statistique et Probabilités"
__version__ = "1.0"
__maintainer__ = ["BARTKOWIAK Valentin", "ELANKUMARAN Vishwa", "PUEL Marius"]
__email__ = ["valentin.bartkowiak@etu.u-bordeaux.fr",
             "vishwa.elankumaran@etu.u-bordeaux.fr",
             "marius.puel@etu.u-bordeaux.fr"]
__status__ = "in development"


class Plot:
    """
        Plot anything

        Supported plot :
            - 2D Graph
            - Histogram
    """
    sns.set()

    def getRandomColor(self, numberGraph: int):
        return np.array(
            [np.random.rand(3,) for element in range(numberGraph)]
        )

    def graph2D(self, xContainer: np.ndarray, yContainer: np.ndarray,
                color: np.ndarray = None, title: str = "") -> plt.show:
        """
            Plot a 2D graph

            Parameter
            ---------
                xContainer  -> Abscissa axis container
                yContainer  -> Ordinate axis container
                color       -> Color Container
                title       -> Title of the graph

            Return
            ------
                Display a 2D graph
        """
        if not hasattr(xContainer[0], "__len__"):
            # If the color is not specified
            if color is None:
                # Default color
                color: np.ndarray = self.getRandomColor(numberGraph=1)

            # Add a graph
            plt.plot(xContainer, yContainer, color=color)

        else:
            # If the color is not specified
            if color is None:
                # Default color
                color: np.ndarray = self.getRandomColor(
                    numberGraph=len(xContainer))

            # Add as many graph as possible depending on xContainer
            list(
                map(
                    lambda x, y, col: plt.plot(x, y, color=col),
                    xContainer, yContainer, color
                )
            )

        # Add a title to the graph
        plt.title(title)

        # Show the graph
        return plt.show()

    def histogram(self, container: np.ndarray, density: bool,
                  histRange: np.ndarray = None, title="",
                  color: np.ndarray = None,
                  callFunction=None) -> plt.show:
        """
            Plot a histogram

            Parameter
            ---------
                container       -> Sample container
                histRange       -> The lower and upper range of the bins
                density         -> If True, draw and
                                   return a probability density
                color           -> Color Container
                title           -> Title of the graph
                callFunction    -> Call another function to draw something

            Return
            ------
                Display a 2D graph
        """
        # If the color is not specified
        if color is None:
            # Default color
            color: np.ndarray = self.getRandomColor(
                numberGraph=len(container))

        # Plot a histogram
        if not hasattr(container[0], "__len__"):
            # Plot a histogram
            plt.hist(
                x=container,
                density=density,
                bins=int(1 + math.log2(len(container))),
                range=histRange,
                color=color
            )

        else:
            # Add as many hist as possible depending on container
            list(
                map(
                    lambda x, histR, col: plt.hist(
                        x=x,
                        density=density,
                        bins=int(1 + math.log2(len(x))),
                        range=histR,
                        color=col
                    ),
                    container, histRange, color
                )
            )

        # Add a title to the graph
        plt.title(title)

        # If there is no callFunction
        if callFunction is None:
            return plt.show()

        # Call another function to draw something else
        else:
            return (
                callFunction['whichFunction'](
                    callFunction["xContainer"],
                    callFunction['yContainer'],
                    color=callFunction["color"],
                    title=title
                ),
            )
