
from typing import TypeVar, Generic

from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..opt import Objective
from .mesh_base import Mesh


class MeshCellQuality():
    mesh: Mesh
    def __call__(self, x: TensorLike) -> TensorLike:
        return self.fun(x)

    def fun(self, x: TensorLike) -> TensorLike:
        """Mesh cell quality function.

        Parameters:
            x (TensorLike): Cartesian coordinates of the mesh nodes, shaped (NN, GD).

        Returns:
            TensorLike: Cell quality in the shape of (NC, ).
        """
        raise NotImplementedError

    def jac(self, x: TensorLike) -> TensorLike:
        """Gradient of the mesh cell quality function.

        Parameters:
            x (TensorLike): Cartesian coordinates of the mesh nodes, shaped (NN, GD).

        Returns:
            TensorLike: Gradient of cell quality in the shape of (NC, NVC, GD).
        """
        raise NotImplementedError

    def hess(self, x: TensorLike) -> TensorLike:
        """Hessian of the mesh cell quality function.

        Parameters:
            x (TensorLike): Cartesian coordinates of the mesh nodes, shaped (NN, GD).

        Returns:
            TensorLike: Hessian of cell quality.
        """
        raise NotImplementedError

_QT = TypeVar('_QT', bound=MeshCellQuality)


class SumObjective(Objective, Generic[_QT]):
    def __init__(self, mesh_quality: _QT, /):
        self.mesh_quality = mesh_quality

    def fun(self, x: TensorLike):
        """Objective function.

        Parameters:
            x (TensorLike): Cartesian coordinates of the mesh nodes, shaped (NN, GD).

        Returns:
            TensorLike: Global cell quality in the shape of (1, ).
        """
        return bm.sum(self.mesh_quality.fun(x), axis=0)

    def jac(self, x: TensorLike):
        """Gradient of the objective function.

        Parameters:
            x (TensorLike): Cartesian coordinates of the mesh nodes, shaped (NN, GD).

        Returns:
            TensorLike: Gradient of cell quality in the shape of (NN, GD).
        """
        cell = self.mesh_quality.mesh.entity('cell')
        NN = self.mesh_quality.mesh.number_of_nodes()
        TD = self.mesh_quality.mesh.TD
        grad = self.mesh_quality.jac(x)
        jacobi = bm.zeros((NN,TD))
        for i in range(TD):
            bm.index_add(jacobi[:,i],cell.flatten(),grad[:,:,i].flatten())
        return jacobi 
