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
        """Equation of state update pressure"""
        return gamma * c0**2 * ((rho/rho0)**gamma - 1) / rho0 + X
    
    def tait_eos_p2rho(slef, p, p0, rho0, gamma=1.0, X=0.0):
        """Calculate density by pressure."""
        p_temp = p + p0 - X
        return rho0 * (p_temp / p0) ** (1 / gamma)
    
    #计算密度
    def compute_rho(self, mass, i_node, w_ij):
        """Density summation."""
        return mass * ops.segment_sum(w_ij, i_node, len(mass))

    #计算运输速度的加速度
    def compute_tv_acceleration(self, state, i_node, j_node, r_ij, dij, grad, pb):
        m_i = state["mass"][i_node]
        m_j = state["mass"][j_node]
        rho_i = state["rho"][i_node]
        rho_j = state["rho"][j_node]

        volume_square = ((m_i/rho_i)**2 + (m_j/rho_j)**2) / m_i 
        c = volume_square * grad / (dij + EPS)
        a = c[:, None] * 1.0 * pb[i_node][:, None] * r_ij       
        return ops.segment_sum(a, i_node, len(state["mass"]))

    #计算张量A，用于计算动量速度的加速度
    def compute_A(self, rho, mv, tv):
        a = rho[:, jnp.newaxis] * mv
        dv = tv - mv
        result = jnp.einsum('ki,kj->kij', a, dv).reshape(a.shape[0], 2, 2)
        return result

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
        tv_i = state["tv"][i_node]
        tv_j = state["tv"][j_node]   

        volume_square = ((m_i/rho_i)**2 + (m_j/rho_j)**2) / m_i
        eta_ij = 2 * eta_i * eta_j / (eta_i + eta_j + EPS) 
        p_ij = (rho_j * p_i + rho_i * p_j) / (rho_i + rho_j)
        c = volume_square * grad / (dij + EPS)
        A = (self.compute_A(rho_i, mv_i, tv_i) + self.compute_A(rho_j, mv_j, tv_j))/2
        mv_ij = mv_i -mv_j
        b = jnp.sum(A * r_ij[:, jnp.newaxis, :], axis=2)
        a = c[:, None] * (-p_ij[:, None] * r_ij + b + (eta_ij[:, None] * mv_ij))
        return ops.segment_sum(a, i_node, len(state["mass"]))

    def forward(self, state, neighbors):
        position = state['position']
        tag = state['tag']

    def boundary_conditions(self, state, box_size, n_walls=3, dx=0.02, T0=1.0, hot_T=1.23): 
        fluid = state["tag"] == 0

        #将输入流体温度设置为参考温度
        inflow = fluid * (state["position"][:,0] < n_walls * dx)
        state["T"] = jnp.where(inflow, T0, state["T"])
        state["dTdt"] = jnp.where(inflow, 0.0, state["dTdt"])

        #设置热壁温度
        hot = state["tag"] == 3
        state["T"] = jnp.where(hot, hot_T, state["T"])
        state["dTdt"] = jnp.where(hot, 0.0, state["dTdt"])

        #将墙设置为参考温度
        solid = state["tag"] == 1
        state["T"] = jnp.where(solid, T0, state["T"])
        state["dTdt"] = jnp.where(solid, 0, state["dTdt"])

        #确保静态墙没有速度或加速度
        static = (hot + solid)[:, None]
        state["mv"] = jnp.where(static, 0.0, state["mv"])
        state["tv"] = jnp.where(static, 0.0, state["tv"])
        state["dmvdt"] = jnp.where(static, 0.0, state["dmvdt"])
        state["dtvdt"] = jnp.where(static, 0.0, state["dtvdt"])
        
        #将出口温度梯度设置为零，以避免与流入相互作用
        bound = np.array([np.zeros_like(box_size), box_size]).T.tolist()
        outflow = fluid * (state["position"][:, 0] > bound[0][1] - n_walls * dx)
        state["dTdt"] = jnp.where(outflow, 0.0, state["dTdt"])
        return state

    def external_acceleration(self, position, box_size, n_walls=3, dx=0.02, g_ext=2.3):
        dxn = n_walls * dx
        res = jnp.zeros_like(position)
        force = jnp.ones((len(position)))
        fluid = (position[:, 1] < box_size[1] - dxn) * (position[:, 1] > dxn)
        force = jnp.where(fluid, force, 0)
        res = res.at[:, 0].set(force)
        return res * g_ext

    def enforce_wall_boundary(self, state, p, g_ext, i_s, j_s, w_dist, dr_i_j, c0=10.0, rho0=1.0, X=5.0, p0=100.0, with_temperature=False):
        """Enforce wall boundary conditions by treating boundary particles in a special way."""
        mask_bc = jnp.isin(state["tag"], jnp.array([1, 3]))
        mask_j_s_fluid = jnp.where(state["tag"][j_s] == 0, 1.0, 0.0)
        w_j_s_fluid = w_dist * mask_j_s_fluid
        w_i_sum_wf = ops.segment_sum(w_j_s_fluid, i_s, len(state["position"]))
        
        #no-slip边界条件
        def no_slip_bc(x):
            #对于墙粒子，流体速度求和
            x_wall_unnorm = ops.segment_sum(w_j_s_fluid[:, None] * x[j_s], i_s, len(state["position"]))
            #广义壁边界条件
            x_wall = x_wall_unnorm / (w_i_sum_wf[:, None] + EPS)
            x = jnp.where(mask_bc[:, None], 2 * x - x_wall, x)
            return x
        mv = no_slip_bc(state["mv"])
        tv = no_slip_bc(state["tv"])
        
        #对于墙粒子，流体压力求和
        p_wall_unnorm = ops.segment_sum(w_j_s_fluid * p[j_s], i_s, len(state["position"]))
        
        #外流体加速度项
        rho_wf_sum = (state["rho"][j_s] * w_j_s_fluid)[:, None] * dr_i_j
        rho_wf_sum = ops.segment_sum(rho_wf_sum, i_s, len(state["position"]))
        p_wall_ext = (g_ext * rho_wf_sum).sum(axis=1)
        
        #normalize
        p_wall = (p_wall_unnorm + p_wall_ext) / (w_i_sum_wf + EPS)
        p = jnp.where(mask_bc, p_wall, p)
        
        rho = self.tait_eos_p2rho(p, p0, rho0, X=5.0)
        
        if with_temperature:
            #对于墙粒子，流体温度求和
            t_wall_unnorm = ops.segment_sum(w_j_s_fluid * state["T"][j_s], i_s, len(state["position"]))
            t_wall = t_wall_unnorm / (w_i_sum_wf + EPS)
            """1:SOLID_WALL,2:MOVING_WALL"""
            mask = jnp.isin(state["tag"], jnp.array([1, 2]))
            t_wall = jnp.where(mask, t_wall, state["T"])
            T = t_wall
            return p, rho, mv, tv, T
        else:
            return p, rho, mv, tv

    def temperature_derivative(self, state, kernel, e_s, dr_i_j, dist, i_s, j_s, grad):
        """compute temperature derivative for next step."""
        kernel_grad_vector = grad[:, None] * e_s
        k = (state["kappa"][i_s] * state["kappa"][j_s]) / \
            (state["kappa"][i_s] + state["kappa"][j_s])
        a = jnp.sum(dr_i_j * kernel_grad_vector, axis=1)
        F_ab = a / ((dist * dist) + EPS) 
        b = 4 * state["mass"][j_s] * k * (state["T"][i_s] - state["T"][j_s])
        dTdt = (b * F_ab) / (state["Cp"][i_s] * state["rho"][i_s] * state["rho"][j_s])
        result = ops.segment_sum(dTdt, i_s, len(state["position"]))
        return result

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
