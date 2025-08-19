from typing import Any, Optional

from ..backend import bm
from ..typing import TensorLike
from ..decorator import variantmethod



class DLDMicrofluidicChipModeler:
    """
    A class for modeling microfluidic chips using deep learning.
    """
    def __init__(self, options):
        self.options = options


    @variantmethod('circle')
    def build(self, gmsh=None):
        """
        Build the microfluidic chip model with circle pillar.

        Parameters:
            gmsh: Optional; GMSH instance for mesh generation.
        """

        self.circles_ = bm.array([
            [0.0, 0.0, 1.0]
            ], dtype=bm.float64)  # Example circle at origin with radius 1.0
        self.boundary_ = bm.array([
            [-5.0, -5.0],
            [5.0, -5.0],
            [5.0, 5.0],
            [-5.0, 5.0]
        ], dtype=bm.float64)

        return self.circles_, self.boundary_

    def add_plot(self, ax= None, **kwargs):
        """
        Plot the DLD geometry (boundary polygon and pillar circles) using Matplotlib.

        Parameters:
            ax (matplotlib.axes.Axes, optional):
                An existing axes object to plot on. If None, a new figure and
                axes are created.
            
            **kwargs:
                Additional keyword arguments passed to the circle patch
                (e.g., ``facecolor``, ``edgecolor``, ``alpha``).

        Returns:
            matplotlib.axes.Axes:
                The axes containing the plotted geometry.

        Notes:
            - Requires that :meth:`build` has already been called so that
              :attr:`circles_` and :attr:`boundary_` are populated.
            - The boundary is drawn as a closed polygon with a blue edge,
              and circles are drawn with white fill and black edge by default.
        """
        if self.circles_ is None or self.boundary_ is None:
            raise RuntimeError("Geometry not built yet. Call `build()` first.")

        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # plot boundary
        bnd = bm.concat((self.boundary_, self.boundary_[[0], :]), axis=0)
        ax.plot(
            bnd[:, 0],
            bnd[:, 1],
            color="blue",
            linewidth=1.5,
        )

        # plot circles
        from matplotlib.patches import Circle
        for (x, y, r) in self.circles_:
            circ = Circle(
                (x, y),
                r,
                facecolor=kwargs.get("facecolor", "white"),
                edgecolor=kwargs.get("edgecolor", "black"),
                linewidth=0.5,
                alpha=kwargs.get("alpha", 1.0),
            )
            ax.add_patch(circ)

        ax.set_aspect("equal", "box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("DLD Geometry")
        return ax
