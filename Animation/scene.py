from manim import *
import numpy as np
from Forest import Forest

class CreateGraph(GraphScene):
    def __init__(self, **kwargs):
        foret = Forest(theta = 5, l = 0.1, h= 1/100000, y0 = 0, sig = 0.05)
        print(foret)
        GraphScene.__init__(
            self,
            x_min=-3,
            x_max=3,
            y_min=-5,
            y_max=5,
            graph_origin=ORIGIN,
            axes_color=BLUE
        )

    def construct(self):
        # Create Graph
        self.setup_axes(animate=True)
        graph = self.get_graph(lambda x: x**2, WHITE)
        graph_label = self.get_graph_label(graph, label='x^{2}')

        graph2 = self.get_graph(lambda x: x**3, WHITE)
        graph_label2 = self.get_graph_label(graph2, label='x^{3}')

        # Display graph
        self.play(ShowCreation(graph), Write(graph_label))
        self.wait(1)
        self.play(Transform(graph, graph2), Transform(graph_label, graph_label2))
        self.wait(1)