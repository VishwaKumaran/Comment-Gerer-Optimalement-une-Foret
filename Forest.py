"""
Comment Gérer Optimalement, et de Manière Durable, une Forêt
"""

# Packages
import math
import numpy as np
import scipy.stats
import random
#from sklearn.neighbors import KernelDensity

import TimeExecution.Timer as Timer
from Process import Process
from Display.Plot import Plot
from Display.Manim import Manim
from Verhulst import Verhulst


__author__ = ["BARTKOWIAK Valentin", "ELANKUMARAN Vishwa", "PUEL Marius"]
__copyright__ = "Copyright 2020, Projet d'Expertise en Statistique et Probabilités"
__version__ = "1.0"
__maintainer__ = ["BARTKOWIAK Valentin", "ELANKUMARAN Vishwa", "PUEL Marius"]
__email__ = ["valentin.bartkowiak@etu.u-bordeaux.fr",
             "vishwa.elankumaran@etu.u-bordeaux.fr",
             "marius.puel@etu.u-bordeaux.fr"]
__status__ = "in development"

# model = "Verhulst"
model = "Process"

class Forest((Process if model == "Process" else Verhulst)):
    """ Needed to be complete """

    def __init__(
        self, plotMethod: str, h: float,
        y0: float, sig: float,
        a: float = None, K: float = None,
        theta: float = None,
        l: float = None
    ):

        if model == "Verhulst":
            super().__init__(
                theta=a,
                l=K,
                h=h,
                y0=y0,
                sig=sig
            )

        else:
            # Call Process class
            super().__init__(
                theta=theta,
                l=l,
                h=h,
                y0=y0,
                sig=sig
            )

        # If the user want to plot using Matplotlib
        if plotMethod == "Plot":
            self.display = Plot()

        # If the user want to animate his output
        else:
            self.display = Manim()

    def trajectory(
        self,
        n: int,
        color: str = "steelblue",
        wantToPlot: bool = True
    ):
        """
            The forest pathway

            Parameter
            ---------
                n           -> The growth of the forest until it reaches n
                color       -> The color of the pathway
                wantToPlot  -> Display the result throw a graph
        """
        if wantToPlot:
            return self.display.graph2D(
                xContainer=np.linspace(0, n, n),
                yContainer=self.calcProcessWithIteration(n),
                color=color,
                title="Trajectoire de la forêt $\\tilde{X}_{kh}$"
            )
        return {
            # xContainer
            "xContainer": np.linspace(0, n, n),

            # yContainer
            "yContainer": self.calcProcessWithIteration(n=n)
        }

    @Timer.timeit
    def cut(self, yStar: float = None, nCut: int = None,
            nTime: int = None, color: (list or np.ndarray) = None,
            wantToPlot: bool = True):
        """
            Cut the forest when it reaches the threshold n cuts
            or during n time

            Parameter
            ---------
                yStar       -> The threshold level at which we cut the forest
                nCut        -> How many times you want to cut the forest
                nTime       -> Cut the forest during the given time
                color       -> The color of the pathway
                wantToPlot  -> Display the result throw a graph
        """
        @Timer.timeit
        def cutForest(threshold: float, **time) -> np.ndarray:
            """
                Cut the forest when it reaches the threshold

                Parameter
                ---------
                    threshold   -> Threshold at which we cut the forest

                Return
                ------
                list
                    The growth of the forest until it reaches the threshold
            """
            # Epsilon
            eps: np.ndarray = np.random.normal(
                0, 1, int(2 * (1 / self.h))
            )

            # Forest container starting at y0
            forestPath: np.ndarray = np.zeros(
                int(2 * (1 / self.h))
            )
            forestPath[0] = self.y0

            # Get the given arguments and attribute them a value
            for key in ('kTime', 'nTime'):
                time[key] = time.get(key, None)

            if time['kTime'] is None:
                time['kTime'] = 0
                time['nTime'] = forestPath.size

            # As long as we don't cut the forest
            for i in range(1, forestPath.size):

                # If it reaches the threshold level -> We cut
                # Or he reaches the limit duration -> Stop
                if forestPath[i - 1] >= threshold or time[
                        'kTime'] + i >= time['nTime']:
                    return forestPath[:i]

                # We let the forest grow
                else:
                    forestPath[i] = (
                        self.calcProcess(
                            x=forestPath[i - 1], eps=eps[i]
                        )
                    )

        # If the user don't specify any color or incomplete color
        if not color or len(color) == 1:
            color: list = ['steelblue', 'red']

        # If the user don't specify the threshold
        if not yStar:
            yStar: float = self.l * 0.8

        # If the user don't specify the number of cuts and time
        if not nCut and not nTime:
            nTime: int = 1 / self.h

        # If the user enter the number of cuts
        if not nTime and nCut > 0:
            cutContainer: list = [
                cutForest(threshold=yStar) for i in range(nCut)
            ]

        # If the user enter time
        elif not nCut and nTime > 0:
            cutContainer: list = []
            time: int = 0

            while time < nTime:
                cutContainer.append(
                    cutForest(threshold=yStar, kTime=time,
                              nTime=nTime)
                )

                time += len(cutContainer[-1])

        # Anything else will be an error of arguments
        else:
            return

        # Get absicissa value
        abscissaLength: list = [0] + [len(cut) for cut in cutContainer]
        abscissaLength: list = [
            sum(abscissaLength[:i]) for i in range(1, len(abscissaLength) + 1)
        ]

        if wantToPlot:
            # Threshold line
            cutContainer.append(
                np.array([yStar] * (int(abscissaLength[-1] * 1.02)))
            )

            return self.display.graph2D(
                xContainer=np.array(
                    list(
                        range(
                            abscissaLength[
                                numberCut - 1], abscissaLength[numberCut]
                        ) for numberCut in range(1, len(abscissaLength))
                    ) + list(
                        [range(0, int(abscissaLength[-1] * 1.02))]
                    ),
                    dtype=object
                ),
                yContainer=np.asarray(cutContainer, dtype=object),
                color=np.array(
                    [color[0]] * (len(cutContainer) - 1) + [color[1]]),
                title="Evolution d'une forêt avec seuil de coupe"
            )
        return (
            # xContainer
            np.array(
                list(
                    range(
                        abscissaLength[
                            numberCut - 1], abscissaLength[numberCut]
                    ) for numberCut in range(1, len(abscissaLength))
                ),
                dtype=object
            ),

            # yContainer
            np.asarray(cutContainer, dtype=object)
        )

    @Timer.timeit
    def rhoLimitLaw(self, iteration: int = None,
                    color: str = "steelblue"):
        """
            Limit law of $\\rho_x$ of our woods.
            In which $\\rho_x$ is refered to Borodin and Salminen
            invariant density.

            $\\rho(x) = \\rho_b(x) := \\frac{1}{C_{b,\\sigma^2} \\sigma^2(x)}
            \\exp(\\int_0^x \\frac{Zb(y)}{\\sigma^2(y)}dy)$

            Parameter
            ---------
                iteration   -> How much forest you want
                               The more it has the more precise will be
                               the result
        """

        # The number of forest we simulate
        if iteration is None:
            # Default value
            iteration: int = 5000

        # How long we let a single forest to grow
        # Careful, it will stabilize around the l value
        step: int = int(1 / self.h)

        # Build Limit Law

        # Container that contain each last value of a forest
        rhoContainer: np.ndarray = np.array(
            [self.calcProcessWithIteration(n=step)[-1]
             for i in range(iteration)]
        )

        # Variance
        tau2: float = (self.sig ** 2) / (2 * self.theta)

        # Absciassa container
        xAbs: np.ndarray = np.linspace(
            self.l - 3 * math.sqrt(tau2),
            self.l + 3 * math.sqrt(tau2),
            100
        )

        # Plot limit law
        self.display.histogram(
            # Histogram parameter
            container=rhoContainer,
            histRange=(self.l - 3 * math.sqrt(tau2),
                       self.l + 3 * math.sqrt(tau2)),
            density=True,
            color=color,
            title="Loi limite de $\\rho$",

            # Draw a graph inside the hist graph
            callFunction={
                "whichFunction": self.display.graph2D,
                "color": "red",
                "xContainer": xAbs,

                # Density of the limit law
                "yContainer": scipy.stats.norm.pdf(
                    x=xAbs,
                    loc=self.l,
                    scale=math.sqrt(tau2)
                )
            }
        )

    def reward(self, x: float, fixedCost: float = 0.5) -> float:
        """
            Reward function

            Parameter
            ---------
                x           -> The moment when you cut the forest
                fixedCost   -> Fixed Costs like machines...

            Return
            ------
                The reward at a given moment
        """
        return np.sqrt(x - fixedCost)

    def rewardConvergence(self,
                          sample: int = 25,
                          yStar: float = None,
                          nCut: int = 100):
        """
            The ps convergence of the reward function

            Parameter
            ---------
                sample  -> The number of forest
                yStar   -> The threshold level at which we cut the forest
                nCut    -> How many times you want to cut the forest


        """
        # If the user don't specify the threshold
        if not yStar:
            yStar: float = self.l * 0.8

        # Container that contains all n - 1 cutting times $\\tau_y^n$
        tauYStar: np.ndarray = np.array([
            [
                # Time before we cut the forest
                (len(forestCutSubSet) - 1) * self.h

                # Subset of forest
                for forestCutSubSet in self.cut(
                    nCut=nCut, yStar=yStar, wantToPlot=False)[1]

                # Samples (set of forest)
            ] for forest in range(sample)
        ])

        # Woodcutter's prize money (gain function)
        yStarReward: float = self.reward(x=yStar)

        averageGainG: np.ndarray = np.array([
            [
                # The average gain function per unit time $G(y)$
                yStarReward / np.mean(subSetForest[:index + 1])

                for index, x in enumerate(subSetForest)
            ] for subSetForest in tauYStar
        ])

        return self.display.graph2D(
            xContainer=np.array([
                np.arange(0, nCut) for forest in range(sample)
            ]),
            yContainer=averageGainG,
            title=(
                "Convergence presque sûre du gain G aléatoire vers le gain G déterministe")
        )

    def deterministicAverageGain(self, yStar: float = None) -> float:
        """
            The average gain function per unit time

            Parameter
            ---------
                yStar   -> The threshold level at which we cut the forest

            Return
            ------
                Average gain function per unit time
        """
        # If the user don't specify the threshold
        if yStar is None:
            yStar: float = self.l * 0.8

        # Average cutting time of your forest
        averageCuttingTime: float = (-1 / self.theta) * \
            np.log(1 - np.divide(yStar, self.l))

        # Return average gain per unit of time
        return np.divide(self.reward(x=yStar), averageCuttingTime)

    def gainEvolution(self, perOfLim: float = 1, nCut: int = 50, sample: int = 100):
        """
            Evolution of the gain according to several cutting thresholds

            Parameter
            ---------
                sample      -> The number of forest
                perOfLim    -> The percentage of the threshold level
                               at which we cut the forest
                nCut        -> How many times you want to cut the forest

        """
        # Several thresholds level
        yStarContainer: np.ndarray = np.linspace(
            0, self.l * perOfLim, sample)

        # Container that contains all n - 1 cutting times $\\tau_y^n$
        tauYStar: np.ndarray = np.array([
            [
                # Time before we cut the forest
                (len(forestCutSubSet) - 1) * self.h

                # Subset of forest
                for forestCutSubSet in self.cut(
                    nCut=nCut,
                    yStar=yStarContainer[forestNumber],
                    wantToPlot=False
                )[1]

                # Samples (set of forest)
            ] for forestNumber in range(sample)
        ])

        # The average gain function per unit time $G(y)$
        averageGainG: np.ndarray = np.array([
            # The average gain function
            # reward at different threshold level / Expected Value(TauYStar)
            self.reward(x=yStarContainer[forestNumber]) / np.mean(forest)

            for forestNumber, forest in enumerate(tauYStar)
        ])

        return self.display.graph2D(
            xContainer=yStarContainer,
            yContainer=averageGainG,
            color="steelblue",
            title="Evolution du gain aléatoire selon plusieurs seuils de coupe"
        )

    def averageGainMultiThreshold(self,
                                  case: str,
                                  perOfLim: float = 1,
                                  whichKernel="triangular",
                                  estimatedDensity=None,
                                  estimatedCDF=None
                                  ):
        """
            Evolution of the average gain function per unit time
            for different cutting thresholds in the deterministic case

            Parameter
            ---------
                perOfLim    -> The percentage of the threshold level
                               at which we cut the forest
                case        -> If deterministic (no randomness)
                               or has randomness
                whichKernel -> Choose a kernel :
                               uniform, gaussian, triangular,
                               epanechnikov, quadratic

        """
        # xContainer
        xContainer: np.ndarray = np.arange(0, self.l * perOfLim, self.h)

        # Deterministic case
        if case == 'deterministic':
            return self.display.graph2D(
                xContainer=xContainer,
                yContainer=self.deterministicAverageGain(yStar=xContainer),
                color='steelblue'
                #title="Evolution de la fonction $G$ pour différents seuils de coupe dans le cas déterministe"
            )

        # Random case
        elif case == 'random':
            return self.display.graph2D(
                xContainer=xContainer,

                # Average gain per unit of time
                yContainer=np.array(
                    list(
                        map(
                            lambda yStar: self.averageGainOpti(
                                yStar=yStar, case='random'),
                            xContainer
                        )
                    )
                ),
                color="steelblue"
                #title="Evolution de la fonction $G$ pour différents seuils de coupe dans le cas aléatoire"
            )

        # For a unknown dynamic
        elif case == "unknown":
            # Estimated density from a forest growth
            # estimatedDensity: np.ndarray = (
            #     self.kernelDensityEsimationForDynamic(
            #         n=10**6,
            #         wantToPlot=False,
            #         whichKernel=whichKernel
            #     )["yContainer"]
            # )

            # # We don't need the 0 values
            # estimatedDensity: np.ndarray = estimatedDensity[
            #     estimatedDensity > 0
            # ]

            # # Build the cumulative distribution from the density
            # estimatedCDF: np.ndarray = np.cumsum(estimatedDensity) * self.h

            return self.display.graph2D(
                xContainer=xContainer,

                # Average gain per unit of time
                yContainer=np.array(
                    list(
                        map(
                            lambda yStar: self.averageGainOpti(
                                yStar=yStar,
                                case="unknown",
                                estimatedDensity=estimatedDensity,
                                estimatedCDF=estimatedCDF
                            ),
                            xContainer
                        )
                    )
                ),
                color="steelblue"
                #title="Evolution de la fonction $G$ pour différents seuils pour une dynamique estimée"
            )

    def deterministicAverageGainMultiThresholdOpti(self, perOfLim: float = 1):
        """
            Evolution of the average gain function per unit time
            for different cutting thresholds when the
            process dynamics are known

            Parameter
            ---------
                perOfLim    -> The percentage of the threshold level
                               at which we cut the forest
        """
        yStarContainer: np.ndarray = np.linspace(0, self.l * perOfLim, 200)

        return self.display.graph2D(
            xContainer=yStarContainer,
            yContainer=np.array([
                self.averageGainOpti(yStar=yStar) for yStar in yStarContainer
            ]),
            color='steelblue',
            title="La fonction $G$ pour différents seuils de coupe lorsque la dynamique est connue"
        )

    def dynamicSolution(self, yStar: float, case: str,
                        estimatedDensity: np.ndarray = None,
                        estimatedCDF: np.ndarray = None) -> float:
        """
            The solution of a dynamic process

            Parameter
            ---------
                yStar               -> Threshold level
                case                -> If deterministic (no randomness)
                                       or has randomness or unknown dynamic
                estimatedDensity    -> The kernel estimated density (limit law)
                estimatedCDF        -> The cumulative distribution of the
                                       limit law

            Return
            ------
                The solution of the dynamic process
        """
        def unknownCase(density: np.ndarray,
                        cdf: np.ndarray) -> np.ndarray:
            """
                Evaluate the solution of the dynamic process

                Parameter
                ---------
                    density         -> The kernel density (limit law)
                    cdf             -> The cumulative ditribution
                                       of the density

                Return
                ------
                    The y value of the integrate
            """
            return 2 * (
                (1 / self.sig**2) * np.divide(1, density) * cdf
            )

        def randomCase(yStarContainer: np.ndarray) -> np.ndarray:
            """
                The optimal thresold in the deterministic case

                Parameter
                ---------
                    yStarContainer  -> Differents threshold level

                Return
                ------
                    The y value of each cutoff
            """
            # ksi formula in the deterministic case
            return 2 * (
                (1 / self.sig**2) * (1 / scipy.stats.norm.pdf(
                    yStarContainer,
                    loc=self.l,
                    scale=np.sqrt(self.sig**2 / (2 * self.theta))
                )) * scipy.stats.norm.cdf(
                    yStarContainer,
                    loc=self.l,
                    scale=np.sqrt(self.sig**2 / (2 * self.theta))
                )
            )

        def deterministicCase(yStarContainer: np.ndarray) -> np.ndarray:
            return (-1 / self.theta) * \
                np.log(1 - np.divide(yStarContainer, self.l))

        # Case when the dynamics of the process is estimated (unknown)
        if case == "unknown":
            estimatedDensity: np.ndarray = estimatedDensity[
                np.where(estimatedDensity <= yStar)
            ]

            estimatedCDF: np.ndarray = estimatedCDF[
                np.where(estimatedCDF <= yStar)
            ]

            sameLength: int = min(len(estimatedDensity), len(estimatedCDF))

            y: np.ndarray = np.linspace(0, yStar, sameLength)

            return np.trapz(
                unknownCase(
                    density=estimatedDensity[:sameLength],
                    cdf=estimatedCDF[:sameLength]
                ),
                x=y
            )
        else:
            y: np.ndarray = np.arange(0, yStar, self.h)

        # Numerical approximation of the integral using trapezoidal method
        return np.trapz(
            randomCase(yStarContainer=y),
            x=y
        ) if case == "random" else deterministicCase(yStarContainer=yStar)

    def averageGainOpti(self, yStar: float, case: str,
                        estimatedDensity: np.ndarray = None,
                        estimatedCDF: np.ndarray = None) -> float:
        """
            The average gain function per unit time $G(y)$ when
            the process dynamics are known

            Parameter
            ---------
                yStar   -> The threshold level
                case    -> Either deterministic case or random case

            Return
            ------
                The average gain depending on the given threshold
        """
        return np.divide(
            self.reward(x=yStar),
            self.dynamicSolution(
                yStar=yStar,
                case=case,
                estimatedDensity=estimatedDensity,
                estimatedCDF=estimatedCDF
            )
        )

    @Timer.timeit
    def bestThreshold(self, threshold: float, case: str, alpha: float = 0.01,
                      eps: float = 10**(-9), dy: float = 10**(-2),
                      maxIter: int = 100000, n: int = 10**3) -> float:
        """
            Determines the optimal cutting threshold
            by using a gradient ascent

            Parameter
            ---------
                threshold   -> Initial value, random threshold
                case        -> If it's deterministic or has randomness
                               or unknown dynamic
                alpha       -> The learning rate which determines
                               the speed of the ascent
                eps         -> Is the desired precision
                dy          -> Step of the derivate approximation
                maxIter     -> The maximum number of iterations
                n           -> Forest growth

            Return
            ------
                The best threshold
        """

        # Initialize variables
        # Initial gradient
        grad: float = 1.0

        # Number of iteration
        iteration: int = 0

        if case == "unknown":
            # Optimal value for the convergence of the result
            dy: float = 10 ** (-1)
            alpha: float = 0.01

            # Get the dynamics of the process using a trajectory
            estimatedDensity: np.ndarray = (
                self.kernelDensityEsimationForDynamic(
                    n=n,
                    wantToPlot=False)['yContainer']
            )

            # The 0 values aren't necessary
            estimatedDensity: np.ndarray = estimatedDensity[
                np.where(estimatedDensity > 0)
            ]

            # Use this density to build his cumulative distribution
            estimatedCDF: np.ndarray = np.cumsum(estimatedDensity) * self.h

        else:
            estimatedDensity = None
            estimatedCDF = None

        # While we don't find the best threshold
        while abs(grad) >= eps:
            # Numerical approximation of the derivative
            grad: float = (
                (
                    self.averageGainOpti(
                        yStar=threshold + dy,
                        case=case,
                        estimatedDensity=estimatedDensity,
                        estimatedCDF=estimatedCDF) - self.averageGainOpti(
                        yStar=threshold - dy,
                        case=case,
                        estimatedDensity=estimatedDensity,
                        estimatedCDF=estimatedCDF
                    )
                ) / ((2 * 10**-2))
            )

            # We increase or descrease the threshold
            threshold += alpha * grad

            # We have done so far an iteration
            iteration += 1

            # print(iteration)
            # print("threshold=", threshold, " et gradient=", grad)

            # If we reach maxIter
            if iteration >= maxIter:
                print("maxIter atteint pour a=", threshold)
                return "Le programme a tourné trop longtemps, veuillez revoir les paramètres"

        # Best threshold
        print(
            threshold,
            self.averageGainOpti(
                yStar=threshold, case=case,
                estimatedDensity=estimatedDensity,
                estimatedCDF=estimatedCDF)
        )

        return self.averageGainMultiThreshold(case=case,estimatedCDF=estimatedCDF,estimatedDensity=estimatedDensity)

    def optimalThresholdForUnknownDynamic(self,
                                          thresholdStart: float,
                                          sample: int = 50
                                          ):
        return np.mean(
            np.array([
                self.bestThreshold(threshold=thresholdStart, case="unknown")
                for i in range(sample)
            ])
        )

    def kernelDensityEsimationForKnownDynamic(
            self, n: int = 1000000, color: str = "steelblue"):
        """
            Determine the dynamics of the process
            by assuming that it is Gaussian

            Parameter
            ---------
                n       -> Forest growth
                color   -> Color of the histogram
        """
        kde = KernelDensity(kernel="tophat", bandwidth=0.025)
        kde.fit(
            np.array(
                list(
                    self.trajectory(n=n, wantToPlot=False).values()
                )
            )
        )
        density = kde.sample(1)[0]

        tau2 = self.sig**2 / (2 * self.theta)

        xAbs: np.ndarray = np.linspace(
            self.l - 3 * np.sqrt(tau2),
            self.l + 3 * np.sqrt(tau2),
            100
        )

        self.display.histogram(
            container=density,
            density=True,
            # histRange=(self.l - 3 * math.sqrt(tau2),
            #            self.l + 3 * math.sqrt(tau2)),
            title="Estimation par noyau 'Tophat' de $\\rho_b$",
            color=color,
            callFunction={
                "whichFunction": self.display.graph2D,
                "color": "red",
                "xContainer": xAbs,
                "yContainer": scipy.stats.norm.pdf(
                    x=xAbs,
                    loc=self.l,
                    scale=np.sqrt(tau2)
                )
            }
        )

    @Timer.timeit
    def kernelDensityEsimationForDynamic(self,
                                         perOfLim: float = 1, n: int = 500000,
                                         whichKernel: str = "marius",
                                         wantToPlot: bool = True):
        """
            Determine the dynamics of the process for any trajectory
            and therefore his law using a kernel density estimation

            Parameter
            ---------
                n               -> Growth of the forest
                whichKernel     -> Choose a kernel :
                                   uniform, gaussian, triangular,
                                   epanechnikov, quadratic
                perOfLim        -> The percentage of the threshold level
                                   at which we cut the forest
                wantToPlot      -> Display the result throw a graph
        """
        def dynamicEstimation(x: float, y: np.ndarray,
                              whichKernel: str
                              ) -> float:
            """
                Evaluate the kernel density estimation
                depending on the kernel we choose

                Parameter
                ---------
                    x               -> Abcissa cordinate
                    y               -> Forest growth
                    whichKernel     -> Choose a kernel :
                                       uniform, gaussian, triangular,
                                       epanechnikov, quadratic

                Return
                ------
                    The dynamic process at a given abscissa
            """
            # Uniform or rectangular kernel
            if whichKernel == "uniform":
                def kernel(X: float):
                    return (1 / 2) * (np.abs(X) <= 1)

            # Gaussian kernel
            elif whichKernel == "gaussian":
                def kernel(X: float):
                    return (1 / np.sqrt(2 * np.pi)) * \
                        np.exp((-1 / 2) * (X**2))

            # Triangular kernel
            elif whichKernel == "triangular":
                def kernel(X: float):
                    return (1 - np.abs(X)) * (np.abs(X) <= 1)

            # Epanechnikov or parabolic kernel
            elif whichKernel == "epanechnikov":
                def kernel(X: float):
                    return (3 / 4) * (1 - X**2) * (np.abs(X) <= 1)

            # Quadratic kernel
            elif whichKernel == "quadratic":
                def kernel(X: float):
                    return (15 / 16) * ((1 - X**2)**2) * (np.abs(X) <= 1)

            elif whichKernel == "marius":
                def kernel(X: float):
                    return (2 - 4 * np.abs(X)) * (np.abs(X) <= 0.5)

            # Kernel Density estimation
            return (
                (1 / (len(y) * self.h)) * np.sum(
                    kernel(
                        np.divide((x - y), self.h)
                    )
                )
            )

        # Abscissa Container
        xAbs: np.ndarray = np.arange(0, perOfLim * self.l, self.h)

        xAbsTh: np.ndarray = np.linspace(0, perOfLim * self.l, int(n / 3))

        # Trajectory of a forest
        forest: np.ndarray = self.trajectory(
            n=n, wantToPlot=False)['yContainer']

        if wantToPlot:
            return self.display.graph2D(
                # xContainer for the evaluate density and the known density
                xContainer=np.array([xAbs, xAbsTh]),

                # yContainer
                # The estimated density
                yContainer=np.array([
                    list(
                        map(
                            lambda X: dynamicEstimation(
                                x=X, y=forest, whichKernel=whichKernel
                            ),
                            xAbs
                        )
                    ),

                    # The known density
                    scipy.stats.norm.pdf(
                        x=xAbsTh,
                        loc=self.l,
                        scale=np.sqrt((self.sig**2) / (2 * self.theta))
                    )
                ], dtype=object),
                color=np.array(["steelblue", "red"]),
                title="Estimation de $\\rho$ avec un noyau " + whichKernel
            )
        return {
            "xContainer": xAbs,
            "yContainer": np.array(
                list(
                    map(
                        lambda X: dynamicEstimation(
                            x=X, y=forest, whichKernel=whichKernel
                        ),
                        xAbs
                    )
                )
            )
        }

    def estimateCDF(self):
        density: list = self.kernelDensityEsimationForDynamic(
            wantToPlot=False, whichKernel="gaussian")
        cdf = np.cumsum(density["yContainer"]) * self.h

        self.display.graph2D(
            xContainer=density["xContainer"],
            yContainer=cdf
        )

a = Forest('Plot', 0.05, 0, 0.05, None, None, 5, 0.1)
a.trajectory(10000)