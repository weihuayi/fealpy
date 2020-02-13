import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import HalfEdgeDomain


class HalfEdgeDomainTest:

    def __init__(self):
        pass

    def square_domain_test(self, plot=True):

        node = np.array([
            (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float)
        halfedge = np.array([
            (1, 0, 1, 3, 4, 1, 1),
            (2, 0, 2, 0, 5, 1, 1),
            (3, 0, 3, 1, 6, 1, 1),
            (0, 0, 0, 2, 7, 1, 1),
            (0, 1, 7, 5, 0, 0, 1),
            (1, 1, 4, 6, 1, 0, 1),
            (2, 1, 5, 7, 2, 0, 1),
            (3, 1, 6, 4, 3, 0, 1)], dtype=np.int)

        domain = HalfEdgeDomain(node, halfedge, 1)

        if plot:
            domain.app_plot(plt)
            plt.show()



test = HalfEdgeDomainTest()
