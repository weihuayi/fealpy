#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: solevr.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Tue 18 Jun 2024 10:45:15 AM CST
	@bref 
	@ref 
'''  
from jax import ops, vmap
import jax.numpy as jnp
import numpy as np
import h5py
import pyvista
from typing import Dict
import enum

#设置标签
class Tag(enum.IntEnum):
    """Particle types."""
    fill_value = -1 #当粒子数量变化时，用 -1 填充
    fluid = 0
    solid_wall = 1 #固壁墙粒子
    moving_wall = 2 #移动墙粒子
    dirichlet_wall = 3 #温度边界条件的狄利克雷墙壁粒子
wall_tags = jnp.array([tag.value for tag in Tag if "WALL" in tag.name])

EPS = jnp.finfo(float).eps

class SPHSolver:
    def __init__(self, mesh):
        self.mesh = mesh 

    #状态方程更新压力
    def tait_eos(self, rho, c0, rho0, gamma=1.0, X=0.0):
        """

        Parameters:
            

        Notes:
            
        """
        return gamma * c0**2 * ((rho/rho0)**gamma - 1) / rho0 + X
    
    #计算密度
    def compute_rho(self, mass, i_node, w_ij):
        return mass * ops.segment_sum(w_ij, i_node, len(mass))

    #计算运输速度的加速度``
    def compute_tv_acceleration(self, state, i_node, j_node, grad_w_ij):
        mesh = self.mesh
        m_i = state["mass"][i_node]
        m_j = state["mass"][j_node]
        rho_i = state["rho"][i_node]
        rho_j = state["rho"][j_node]
        pb = state["pb"][i_node]

        volume_square = ((m_i/rho_i)**2 + (m_j/rho_j)**2) / m_i
        a = volume_square[:, None] * pb[:, None] * grad_w_ij
        return ops.segment_sum(a, i_node, len(state["mass"]))

    #计算A
    def compute_A(self, state):
        rho = state["rho"]
        mv = state["mv"]
        tv = state["tv"]
        return rho[:, None] * mv * (tv - mv)

    #计算动量速度的加速度
    def compute_mv_acceleration(self, state, i_node, j_node, r_ij, dij, grad, p):
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
        

        volume_square = ((m_i/rho_i)**2 + (m_j/rho_j)**2) / m_i
        eta_ij = 2 * eta_i * eta_j / (eta_i + eta_j + EPS) 
        p_ij = (rho_j * p_i + rho_i * p_j) / (rho_i + rho_j)
        c = volume_square * grad / (dij + EPS)

        #A = (self.compute_A(state)[i_node] + self.compute_A(state)[j_node])/2
        mv_ij = mv_i - mv_j
        #a = volume_square[:, None] * grad_w_ij * (-p_ij[:, None] + A + (eta_ij[:, None] * mv_ij / position_ij[:,None]))
        a = c[:, None] * (-p_ij[:, None] * r_ij + (eta_ij[:, None] * mv_ij))
        return ops.segment_sum(a, i_node, len(state["mass"]))

    def forward(self, state, neighbors):
        position = state['position']
        tag = state['tag']


    def write_h5(self, data_dict: Dict, path: str):
        """Write a dict of numpy or jax arrays to a .h5 file."""
        hf = h5py.File(path, "w")
        for k, v in data_dict.items():
            hf.create_dataset(k, data=jnp.array(v))
        hf.close()
    
    def dict2pyvista(self, data_dict):
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

    def write_vtk(self, data_dict: Dict, path: str):
        """Store a .vtk file for ParaView."""
        data_pv = self.dict2pyvista(data_dict)
        data_pv.save(path)
