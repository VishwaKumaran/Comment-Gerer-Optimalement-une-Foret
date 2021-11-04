from manim import *

# import math
import numpy as np


# forest = Forest(theta=5, l=0.1, h=0.0005, y0=0, sig=0.05)
# n = 100
# xAbs, yOrd = forest.trajectory(n)

class Manim:
    """
        Animate anything

        Supported animation :
            - Nothing for now
    """
    class graph2D(GraphScene):
        def __init__(self, **kwargs):
            self.parameter: dict = kwargs
            print(self.parameter)

            if not hasattr(self.parameter["xContainer"][0], "__len__"):
                # Get the max of the abscissa
                maxAbs: float or int = max(self.parameter["xContainer"])

                # Get the min of the ordinate
                minOrd: float or int = min(self.parameter["yContainer"])

                # Get the max of the ordinate
                maxOrd: float or int = max(self.parameter["yContainer"])

                # Get the max of the ordinate
                n: int = len(self.parameter["yContainer"])

            else:
                # Get the max of the abscissa
                maxAbs: float or int = max(
                    [max(abscissa)
                     for abscissa in self.parameter['xContainer']]
                )

                # Get the min of the ordinate
                minOrd: float or int = min(
                    [min(ordinate)
                     for ordinate in self.parameter['yContainer']]
                )

                # Get the max of the ordinate
                maxOrd: float or int = max(
                    [max(ordinate)
                     for ordinate in self.parameter['yContainer']]
                )

                # Get the max of the ordinate
                n: int = max(
                    [len(ordinate)
                     for ordinate in self.parameter['yContainer']]
                )

            GraphScene.__init__(
                self,
                x_min=0,
                x_max=maxAbs - 1,
                y_min=minOrd,
                y_max=maxOrd,
                x_labeled_nums=np.arange(0, int(maxAbs), int(n / 5)),
                y_labeled_nums=np.linspace(minOrd, maxOrd, 5),
                x_axis_label="Time",
                x_axis_config={
                    "tick_frequency": int(n / 5)
                },
                y_axis_config={
                    "decimal_number_config": {
                        "num_decimal_places": 3
                    }
                },
                axes_color=BLUE
            )

            self.render(preview=True)

        def construct(self):
            # Create Graph
            self.setup_axes(animate=True)
            # self.axes.center()
            # graph: np.ndarray = np.array(
            #     [
            #         self.get_graph(
            #             lambda x: self.parameter['yContainer'][i][int(
            #                 x)], GREEN
            #         ) for i in range(len(self.parameter['yContainer']))
            #     ]
            # )

            graph = self.get_graph(
                lambda x: self.parameter['yContainer'][int(x)],
                GREEN
            )

            print(graph)

            graph_label = self.get_graph_label(
                graph,
                label="\\tilde{X}_{kh}"
            )

            graph_title = Tex("Trajectoire de la forêt $\\tilde{X}_{kh}$")
            graph_title.next_to(graph, UP)

            self.add(graph_title)
            VGroup(*self.mobjects).center()

            # Display graph
            # self.play(Write(graph_title))
            self.play(ShowCreation(graph), Write(graph_label))
            self.wait(1)
