from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from typing import Dict
import pyvista

import jax #打印
#jax.debug.print("result:{}", bm.sum(a))
import numpy as np #画图

# Types
Box = TensorLike
f32 = bm.float32
EPS = bm.finfo(float).eps

class Space:
    def raw_transform(self, box:Box, R:TensorLike):
        if box.ndim == 0 or box.size == 1:
            
            return R * box
        elif box.ndim == 1:
            indices = self._get_free_indices(R.ndim - 1) + "i"
            
            return bm.einsum(f"i,{indices}->{indices}", box, R)
        elif box.ndim == 2:
            free_indices = self._get_free_indices(R.ndim - 1)
            left_indices = free_indices + "j"
            right_indices = free_indices + "i"
            
            return bm.einsum(f"ij,{left_indices}->{right_indices}", box, R)
        raise ValueError(
            ("Box must be either: a scalar, a vector, or a matrix. " f"Found {box}.")
        )

    def _get_free_indices(self, n: int):
        
        return "".join([chr(ord("a") + i) for i in range(n)])

    def pairwise_displacement(self, Ra: TensorLike, Rb: TensorLike):
        if len(Ra.shape) != 1:
            msg = (
				"Can only compute displacements between vectors. To compute "
				"displacements between sets of vectors use vmap or TODO."
				)
            raise ValueError(msg)

        if Ra.shape != Rb.shape:
            msg = "Can only compute displacement between vectors of equal dimension."
            raise ValueError(msg)

        return Ra - Rb

    def periodic_displacement(self, side: Box, dR: TensorLike):
        _dR = ((dR + side * 0.5) % side) - 0.5 * side
        return _dR

    def periodic_shift(self, side: Box, R: TensorLike, dR: TensorLike):

        return (R + dR) % side

    def periodic(self, side: Box, wrapped: bool = True):
        def displacement_fn( Ra: TensorLike, Rb: TensorLike, perturbation = None, **unused_kwargs):
            if "box" in unused_kwargs:
                raise UnexpectedBoxException(
                    (
                        "`space.periodic` does not accept a box "
                        "argument. Perhaps you meant to use "
                        "`space.periodic_general`?"
                    )
                )
            dR = self.periodic_displacement(side, self.pairwise_displacement(Ra, Rb))
            if perturbation is not None:
                dR = self.raw_transform(perturbation, dR)
            
            return dR
        if wrapped:
            def shift_fn(R: TensorLike, dR: TensorLike, **unused_kwargs):
                if "box" in unused_kwargs:
                    raise UnexpectedBoxException(
                        (
                            "`space.periodic` does not accept a box "
                            "argument. Perhaps you meant to use "
                            "`space.periodic_general`?"
                        )
                    )

                return self.periodic_shift(side, R, dR)
        else:
                def shift_fn(R: TensorLike, dR: TensorLike, **unused_kwargs):
                    if "box" in unused_kwargs:
                        raise UnexpectedBoxException(
                            (
                                "`space.periodic` does not accept a box "
                                "argument. Perhaps you meant to use "
                                "`space.periodic_general`?"
                            )
                        )
                    return R + dR

        return displacement_fn, shift_fn

    def distance(self, dR: TensorLike):
        dr = self.square_distance(dR)
        return self.safe_mask(dr > 0, bm.sqrt, dr)

    def square_distance(self, dR: TensorLike):
        return bm.sum(dR**2, axis=-1)

    def safe_mask(self, mask, fn, operand, placeholder=0):
        masked = bm.where(mask, operand, 0)
        return bm.where(mask, fn(masked), placeholder)

class SPHSolver:
    def __init__(self, mesh):
        self.mesh = mesh 
    
    @staticmethod
    #@bm.compile
    def compute_rho(mass, i_node, w_ij):
        """Density summation"""
        a = bm.zeros_like(mass)
        return mass * bm.index_add(a, i_node, w_ij, axis=0, alpha=1)

    @staticmethod
    #@bm.compile
    def tait_eos(rho, c0, rho0, gamma=1.0, X=0.0):
        """Equation of state update pressure"""
        return gamma * c0**2 * ((rho/rho0)**gamma - 1) / rho0 + X

    @staticmethod
    #@bm.compile
    def compute_mv_acceleration(state, i_node, j_node, r_ij, dij, grad, p):
        """Momentum velocity variation"""
        def compute_A(rho, mv, tv):
            a = rho[:, bm.newaxis] * mv
            dv = tv - mv
            result = bm.einsum('ki,kj->kij', a, dv).reshape(a.shape[0], 2, 2)
            return result

        eta_i = state["eta"][i_node]
        eta_j = state["eta"][j_node]
        p_i = p[i_node]
        p_j = p[j_node]
        rho_i = state["rho"][i_node]
        rho_j = state["rho"][j_node]
        m_i = state["mass"][i_node]
        m_j = state["mass"][j_node]
        mv_i = state["mv"][i_node]
        mv_j = state["mv"][j_node]
        tv_i = state["tv"][i_node]
        tv_j = state["tv"][j_node]   

        volume_square = ((m_i/rho_i)**2 + (m_j/rho_j)**2) / m_i
        eta_ij = 2 * eta_i * eta_j / (eta_i + eta_j + EPS)
        p_ij = (rho_j * p_i + rho_i * p_j) / (rho_i + rho_j)
        c = volume_square * grad / (dij + EPS)
        A = (compute_A(rho_i, mv_i, tv_i) + compute_A(rho_j, mv_j, tv_j))/2
        mv_ij = mv_i -mv_j
        b = bm.sum(A * r_ij[:, bm.newaxis, :], axis=2)
        a = c[:, None] * (-p_ij[:, None] * r_ij + b + (eta_ij[:, None] * mv_ij))
        add_dv = bm.zeros_like(state["mv"])
        return bm.index_add(add_dv, i_node, a, axis=0, alpha=1)
        

    

    def write_vtk(self, data_dict: Dict, path: str):
        """Store a .vtk file for ParaView."""
        data_pv = self.dict2pyvista(data_dict)
        data_pv.save(path)

    def dict2pyvista(self, data_dict):
        # TODO bm
        # PyVista works only with 3D objects, thus we check whether the inputs
        # are 2D and then increase the degrees of freedom of the second dimension.
        # N is the number of points and dim the dimension
        r = np.asarray(data_dict["position"])
        N, dim = r.shape
        # PyVista treats the position information differently than the rest
        if dim == 2:
            r = np.hstack([r, np.zeros((N, 1))])
        data_pv = pyvista.PolyData(r)
        # copy all the other information also to pyvista, using plain numpy arrays
        for k, v in data_dict.items():
            # skip r because we already considered it above
            if k == "r":
                continue
            # working in 3D or scalar features do not require special care
            if dim == 2 and v.ndim == 2:
                v = np.hstack([v, np.zeros((N, 1))])
            data_pv[k] = np.asarray(v)
        return data_pv