import numpy as np
from typing import Callable, Dict, Literal, Sequence
from fealpy.backend import TensorLike 
from fealpy.backend import backend_manager as bm
from fealpy.model.boundary_condition import BoundaryCondition, bc_mask, bc_value
from fealpy.model import PDEDataManager
pde = PDEDataManager('elliptic').get_example('sinsin')


# Test script
if __name__ == "__main__":
    # Create an instance of SinSinData2D
    sin_sin_data = pde

    # Define a set of test points
    test_points = bm.array([[0.5, 0.5], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])

    # Compute and print the exact solution at test points
    print("Exact solution at test points:")
    print(sin_sin_data.solution(test_points))

    # Compute and print the gradient at test points
    print("\nGradient at test points:")
    print(sin_sin_data.gradient(test_points))

    # Compute and print the flux at test points
    print("\nFlux at test points:")
    print(sin_sin_data.flux(test_points))

    # Compute and print the source term at test points
    print("\nSource term at test points:")
    print(sin_sin_data.source(test_points))

    # Compute and print the Dirichlet boundary condition at test points
    print("\nDirichlet boundary condition at test points:")
    print(sin_sin_data.dirichlet(test_points))

    # Check and print if test points are on the Dirichlet boundary
    print("\nAre test points on the Dirichlet boundary?")
    print(sin_sin_data.is_dirichlet_boundary(test_points))
