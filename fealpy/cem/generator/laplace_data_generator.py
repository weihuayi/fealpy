
from typing import Callable, Tuple, Iterable, Generator, Any, Dict, Optional, Union
import numpy as np
from numpy import float32
from numpy.typing import NDArray
from scipy.sparse.linalg import splu
from scipy.sparse import spdiags, hstack, vstack, csr_matrix

from fealpy.functionspace import LagrangeFESpace
from fealpy.mesh import TriangleMesh, UniformMesh2d
from fealpy.fem import BilinearForm
from fealpy.fem import ScalarDiffusionIntegrator
from fealpy.fem import ScalarNeumannSourceIntegrator


ArrayFunction = Callable[..., NDArray]
ArrayOrFunc = Union[NDArray, ArrayFunction]


class LaplaceFEMSolver():
    """
    Finite element method-based solver for the Laplace equation.

    This solver is designed to efficiently solve the Laplace equation under various boundary conditions,
    utilizing LU decomposition on the same matrix for faster consecutive solutions.

    Methods:
    1. solve_from_gd(self, gd):
        Solve the Laplace equation with Dirichlet boundary conditions.

        Parameters:
        - gd: numpy.ndarray. Dirichlet boundary conditions.

        Returns:
        numpy.ndarray: The solution to the Laplace equation.

    2. solve_from_gn(self, gn):
        Solve the Laplace equation with Neumann boundary conditions.

        Parameters:
        - gn: numpy.ndarray. Neumann boundary conditions.

        Returns:
        numpy.ndarray: The solution to the Laplace equation.
    """
    def __init__(self, space, sigma: Optional[NDArray]=None) -> None:
        """
        @brief Build a laplace equation solver based on FEM.

        @param space: a finite element space object in FEALPy.
        @param sigma: array of sigma values in each cell, with shape (NC, ).
        """
        self.space = space
        self.ndof = space.number_of_global_dofs()

        bform = BilinearForm(space)
        bform.add_domain_integrator(ScalarDiffusionIntegrator(c=sigma, q=3))
        self.A_ = bform.assembly()

    def _init_gd(self):
        space = self.space
        isDDof = space.is_boundary_dof()
        A_ = self.A_

        bdIdx = np.zeros(A_.shape[0], dtype=np.int_)
        bdIdx[isDDof.reshape(-1)] = 1
        D0 = spdiags(1-bdIdx, 0, A_.shape[0], A_.shape[0])
        D1 = spdiags(bdIdx, 0, A_.shape[0], A_.shape[0])
        A_ = D0@A_@D0 + D1

        self.AD_lu = splu(A_.tocsc())

    def solve_from_gd(self, gd: ArrayOrFunc) -> NDArray:
        """
        @brief Solve the laplace equation from one dirichlet boundary condition.
        """
        space = self.space

        if not hasattr(self, "AD_lu"):
            self._init_gd()

        uh = np.zeros((self.ndof, ), dtype=space.ftype)
        isDDof = space.boundary_interpolate(gd, uh)
        b_ = np.zeros((self.ndof, ), dtype=space.ftype)
        b_[:] = b_ - self.A_@uh.reshape(-1)
        b_[isDDof] = uh[isDDof]

        uh[:] = self.AD_lu.solve(b_)
        return uh

    def solve_from_gds(self, gd_iterable: Iterable[ArrayOrFunc]) -> Generator[NDArray, Any, None]:
        """
        @brief The generator version of `solve_from_gd`.
        """
        for gD_ in gd_iterable:
            yield self.solve_from_gd(gD_)

    def _init_gn(self):
        space = self.space
        A_ = self.A_
        C_ = ScalarNeumannSourceIntegrator(1., q=3).assembly_face_vector(space)

        A_C = hstack([A_, C_.reshape(-1, 1)])
        A_C = vstack([A_C, hstack([C_.reshape(1, -1), csr_matrix((1, 1), dtype=space.ftype)])])

        self.AC_lu = splu(A_C.tocsc())

    def solve_from_gn(self, gn: ArrayOrFunc) -> NDArray:
        """
        @brief Solve the laplace equation from one neumann boundary condition.
        """
        space = self.space

        if not hasattr(self, "AC_lu"):
            self._init_gn()

        intergrator = ScalarNeumannSourceIntegrator(gn, q=3)
        F_ = np.zeros((self.ndof+1, ), dtype=space.ftype)
        intergrator.assembly_face_vector(space, out=F_[:-1])

        uh = np.zeros((self.ndof, ), dtype=space.ftype)
        uh[:] = self.AC_lu.solve(F_)[:-1]
        return uh

    def solve_from_gns(self, gn_iterable: Iterable[ArrayOrFunc]) -> Generator[NDArray, Any, None]:
        """
        @brief The generator version of `solve_from_gn`.
        """
        for gN_ in gn_iterable:
            yield self.solve_from_gn(gN_)


def get_gN_func(freq: int, phrase: float=0.):
    """
    @brief Define a 2D gN function.
    """
    def gN(p, *args):
        theta = np.arctan2(p[..., 1], p[..., 0])
        return np.cos(freq * theta + phrase)
    return gN


class LaplaceDataGenerator2d():
    """
    A 2-d GD & GN generator based on FEM.
    """
    def __init__(self, box: Tuple[float, float, float, float],
                 nx: int, ny: int,
                 gn_funcs: Iterable[ArrayFunction],
                 sigma_vals: Tuple[float, float]=(1., 1.),
                 levelset: Optional[ArrayFunction]=None):
        """
        @brief Generate gD and gN by FEM.

        @param box: solving area.
        @param nx: int.
        @param ny: int.
        @param gn_funcs: An iterable of gN functions.
        @param sigma_vals: tuple of two floats representing sigma inside and\
               outside the level set.
        @param levelset: the levelset function of the interface of sigma, optional.\
               The outside sigma value will be taken if no levelset provided.
        """
        # construct FE space
        if levelset is None:
            self.mesh = TriangleMesh.from_box(box, nx, ny)
        else:
            self.mesh = TriangleMesh.interfacemesh_generator(box, nx, ny, levelset)
        self.space = LagrangeFESpace(self.mesh, p=1)
        self.bd_node_flag = self.mesh.ds.boundary_node_flag()
        self.box = box
        self.shape = (nx, ny)

        # set the value of sigma
        def _coef_func(p: NDArray):
            inclusion = levelset(p) < 0.
            sigma = np.empty(p.shape[:2], dtype=p.dtype) # (Q, C)
            sigma[inclusion] = sigma_vals[0]
            sigma[~inclusion] = sigma_vals[1]
            return sigma

        if levelset is None:
            sigma = None
        else:
            sigma = _coef_func

        self.gn_funcs = gn_funcs
        self.levelset = levelset

        # prepare for solving
        self.solver = LaplaceFEMSolver(self.space, sigma)
        self.uh_generator = self.solver.solve_from_gns(self.gn_funcs)

    def __iter__(self):
        yield from zip(self.gd(), self.gn())

    @classmethod
    def from_cos(cls, box, nx: int, ny: int, levelset: ArrayFunction,
                 sigma_vals: Tuple[float, float],
                 freq: Iterable[int], phrase: Iterable[float]):

        gn_list = [get_gN_func(f, p) for f in freq for p in phrase]

        return cls(box=box, nx=nx, ny=ny, levelset=levelset, sigma_vals=sigma_vals,
                   gn_funcs=gn_list)

    def _build_uniform_mesh(self):
        box = self.box
        shape = self.shape
        h = [0., 0.]
        for d in range(2):
            h[d] = (box[2*d+1] - box[2*d]) / shape[d]

        return UniformMesh2d([0, shape[0], 0, shape[1]], h, origin=[box[0], box[2]])

    def is_available(self):
        """
        @brief Determines whether a suitable interface mesh has been generated
        for the level set.

        Use another level-function if this method returns `False`.
        """
        bd_dof_flag = self.bd_node_flag
        NN_bd: int = np.sum(bd_dof_flag)
        return sum(self.shape) * 2 == NN_bd

    def label(self):
        if not hasattr(self, 'label_mesh'):
            self.label_mesh = self._build_uniform_mesh()

        val = self.levelset(self.label_mesh.node)
        return val < 0.

    def gd(self) -> Generator[NDArray[np.floating], Any, None]:
        """
        @brief Get gn data in boundary nodes, in shape (NN_bd, ).
        """
        bd_dof_flag = self.bd_node_flag
        for uh in self.uh_generator:
            yield uh[bd_dof_flag]

    def gn(self) -> Generator[NDArray[np.floating], Any, None]:
        """
        @brief Get gn data of each boundary nodes, in shape (NN_bd, ).

        This is done by interpolating the given gN functions into to the
        boundary nodes.
        """
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        qf = mesh.integrator(q=3, etype='face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        bd_node_flag = self.bd_node_flag
        bd_face_flag = mesh.ds.boundary_face_flag()
        face2node = mesh.entity('face', index=bd_face_flag)
        pts = mesh.bc_to_point(bcs, index=bd_face_flag)
        face_measure = mesh.entity_measure('face', index=bd_face_flag)
        face_measure_prop = face_measure / np.sum(face_measure)
        n = mesh.face_unit_normal(index=bd_face_flag)

        # NOTE: How many faces are there around a boundary node.
        # The `face2node` has already been sliced by boundary flag,
        # so we are sure that only boundary faces are counted here.
        count = np.zeros((NN, ), dtype=np.int_)
        np.add.at(count, face2node, 1)

        for gn in self.gn_funcs:
            val = gn(pts, n) # (QN, NFbd)
            fval = ws @ val # fval = np.einsum("q, qf -> f", ws, val)
            fval -= np.sum(fval * face_measure_prop)
            nval = np.zeros((NN, ), dtype=np.float64)
            np.add.at(nval, face2node, fval[:, None])
            nval[bd_node_flag] /= count[bd_node_flag]

            yield nval[bd_node_flag]

    def save(self, sigma_idx: int, path: str, gn_names: Iterable[str], *,
             dtype=float32, **kwargs: NDArray):
        """
        @brief Generate GD & GN data and save as `.npz` file.

        @param sigma_idx: int. The index of the $\sigma$ parameter to present in
        the file name.
        @param path: str. Starts and ends with '/'.
        @param gn_names: An iterable containing names for each gs&gn data in a
        `.npz` file.
        """
        data_dict: Dict[str, NDArray] = {}
        data_dict["label"] = self.label()
        for name, gd, gn in zip(gn_names, self.gd(), self.gn()):
            data = np.stack([gd, gn], axis=0, dtype=dtype)
            data_dict[name] = data
        for k, v in kwargs.items():
            data_dict[k] = v
        np.savez(path + f"{sigma_idx}.npz", **data_dict)
