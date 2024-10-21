from typing import Tuple
# from .mesh_base import HomoMesh

import jax.numpy as jnp


class UniformMesh2d():

    def __init__(self):
        """
        @brief: A class for representing a two-dimensional structured mesh with uniform discretization \
                in both x and y directions.
        """

        def __init__(self, extent: Tuple[int, int, int, int], \
                           h: Tuple[float, float] = (1.0, 1.0), \
                           origin: Tuple[float, float] = (0.0, 0.0), \
                           itype: type = jnp.int32,
                           ftype: type = jnp.float64):
            """
            @brief: Initialize the 2D uniform mesh.

            @param[in] extent: A tuple representing the range of the mesh in the x and y directions.
            @param[in] h: A tuple representing the mesh step sizes in the x and y directions, \
                          default: (1.0, 1.0).
            @param[in] origin: A tuple representing the coordinates of the starting point, \
                               default: (0.0, 0.0).
            @param[in] itype: Integer type to be used, default: jnp.int32.
            @param[in] ftype: Floating point type to be used, default: jnp.float64.
            """

            super().__init__()
            # Mesh properties
            self.extent = extent
            self.h = h
            self.origin = origin

            # Mesh dimensions
            self.nx = self.extent[1] - self.extent[0]
            self.ny = self.extent[3] - self.extent[2]
            self.NN = (self.nx + 1) * (self.ny + 1)
            self.NC = self.nx * self.ny

            self.itype = itype
            self.ftype = ftype

            self.meshtype = 'UniformMesh2d'

        def cell_area(self):
            """
            @brief: Calculate and return the area of a single cell in the mesh

            @return: The area of a single cell (all cells have the same area)
            """
            return self.h[0] * self.h[1]

        def gradient(self, f, order=1):
            """
            @brief 求网格函数 f 的梯度
            """
            hx = self.h[0]
            hy = self.h[1]
            fx, fy = jnp.gradient(f, hx, hy, edge_order=order)

            return fx, fy
