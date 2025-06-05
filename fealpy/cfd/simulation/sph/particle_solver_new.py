from fealpy.backend import backend_manager as bm
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from fealpy.backend import TensorLike
from typing import Dict
import pyvista

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
    def compute_rho(mass, i_node, w_ij, dtype=bm.float64):
        """Density summation"""
        a = bm.zeros_like(mass, dtype=dtype)
        return mass * bm.index_add(a, i_node, w_ij, axis=0, alpha=1)

    @staticmethod
    #@bm.compile
    def tait_eos(rho, c0, rho0, gamma=1.0, X=0.0):
        """Equation of state update pressure"""
        return gamma * c0**2 * ((rho/rho0)**gamma - 1) / rho0 + X

    @staticmethod
    #@bm.compile
    def tait_eos_p2rho(p, p0, rho0, gamma=1.0, X=0.0):
        """Calculate density by pressure"""
        p_temp = p + p0 - X
        return rho0 * (p_temp / p0) ** (1 / gamma)

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
        add_dv = bm.zeros_like(state["mv"], dtype=bm.float64)
        return bm.index_add(add_dv, i_node, a, axis=0, alpha=1)
        
    @staticmethod
    #@bm.compile
    def compute_tv_acceleration(state, i_node, j_node, r_ij, dij, grad, pb):
        """Transport velocity variation"""
        m_i = state["mass"][i_node]
        m_j = state["mass"][j_node]
        rho_i = state["rho"][i_node]
        rho_j = state["rho"][j_node]

        volume_square = ((m_i / rho_i) ** 2 + (m_j / rho_j) ** 2) / m_i
        c = volume_square * grad / (dij + EPS)
        a = c[:, None] * pb[i_node][:, None] * r_ij      
        return bm.index_add(bm.zeros_like(state["tv"], dtype=bm.float64), i_node, a, axis=0, alpha=1) 

    @staticmethod
    #@bm.compile
    def boundary_conditions(state, box_size, n_walls=3, dx=0.02, T0=1.0, hot_T=1.23):
        """Boundary condition settings""" 
        # 将输入流体温度设置为参考温度
        fluid = state["tag"] == 0
        inflow = fluid * (state["position"][:, 0] < n_walls * dx)
        state["T"] = bm.where(inflow, T0, state["T"])
        state["dTdt"] = bm.where(inflow, 0.0, state["dTdt"])

        # 设置热壁温度
        hot = state["tag"] == 3
        state["T"] = bm.where(hot, hot_T, state["T"])
        state["dTdt"] = bm.where(hot, 0.0, state["dTdt"])

        # 将墙设置为参考温度
        solid = state["tag"] == 1
        state["T"] = bm.where(solid, T0, state["T"])
        state["dTdt"] = bm.where(solid, 0, state["dTdt"])

        # 确保静态墙没有速度或加速度
        static = (hot + solid)[:, None]
        state["mv"] = bm.where(static, 0.0, state["mv"])
        state["tv"] = bm.where(static, 0.0, state["tv"])
        state["dmvdt"] = bm.where(static, 0.0, state["dmvdt"])
        state["dtvdt"] = bm.where(static, 0.0, state["dtvdt"])

        # 将出口温度梯度设置为零，以避免与流入相互作用
        bound = bm.concatenate([bm.zeros_like(box_size).reshape(-1,1), box_size.reshape(-1,1)], axis=-1)
        outflow = fluid * (state["position"][:, 0] > bound[0, 1] - n_walls * dx)
        state["dTdt"] = bm.where(outflow, 0.0, state["dTdt"])
        return state

    @staticmethod
    #@bm.compile
    def external_acceleration(position, box_size, dx=0.02):
        """Set external velocity field"""
        dxn = 3 * dx
        res = bm.zeros_like(position)
        force = bm.ones(len(position))
        fluid = (position[:, 1] < box_size[1] - dxn) * (position[:, 1] > dxn)
        force = bm.where(fluid, force, 0)
        res = 2.3 * bm.set_at(res, (slice(None), 0), force)
        return res

    @staticmethod
    #@bm.compile
    def enforce_wall_boundary(state, p, g_ext, i_s, j_s, w_dist, dr_i_j, c0=10.0, rho0=1.0, X=5.0, \
        p0=100.0, with_temperature=False, dtype=bm.float64):
        """Enforce wall boundary conditions by treating boundary particles in a special way"""
        mask_bc = bm.isin(state["tag"], bm.array([1, 3]))
        mask_j_s_fluid = bm.where(state["tag"][j_s] == 0, 1.0, 0.0)
        w_j_s_fluid = w_dist * mask_j_s_fluid
        w_i_sum_wf = bm.index_add(bm.zeros(len(state["position"]), dtype=dtype), i_s, w_j_s_fluid, axis=0, alpha=1)

        # no-slip boundary condition
        def no_slip_bc(x):
            x_wall_unnorm = bm.index_add(bm.zeros_like(state["position"], dtype=dtype), i_s, w_j_s_fluid[:, None] * x[j_s], axis=0, alpha=1)
            x_wall = x_wall_unnorm / (w_i_sum_wf[:, None] + EPS)
            x = bm.where(mask_bc[:, None], 2 * x - x_wall, x)
            return x
        mv = no_slip_bc(state["mv"])
        tv = no_slip_bc(state["tv"])
        
        # Pressure summation
        p_wall_unnorm = bm.index_add(bm.zeros(len(state["position"]), dtype=dtype), i_s, w_j_s_fluid * p[j_s], axis=0, alpha=1)

        # External acceleration term
        rho_wf_sum = (state["rho"][j_s] * w_j_s_fluid)[:, None] * dr_i_j
        rho_wf_sum = bm.index_add(bm.zeros_like(state["position"], dtype=dtype), i_s, rho_wf_sum, axis=0, alpha=1)
        p_wall_ext = (g_ext * rho_wf_sum).sum(axis=1)
        
        # Normalize pressure
        p_wall = (p_wall_unnorm + p_wall_ext) / (w_i_sum_wf + EPS)
        p = bm.where(mask_bc, p_wall, p)
        
        rho = SPHSolver.tait_eos_p2rho(p, p0, rho0, X=5.0)
        
        def compute_temperature():
            t_wall_unnorm = bm.index_add(bm.zeros(len(state["position"]), dtype=dtype), i_s, w_j_s_fluid * state["T"][j_s], axis=0, alpha=1)
            t_wall = t_wall_unnorm / (w_i_sum_wf + EPS)
            mask = bm.isin(state["tag"], bm.array([1, 2]))
            t_wall = bm.where(mask, t_wall, state["T"])
            return t_wall

        T = bm.where(bm.array(with_temperature), compute_temperature(), state["T"])
        return p, rho, mv, tv, T 

    @staticmethod
    #@bm.compile(static_argnames=('kernel',))
    def temperature_derivative(state, kernel, e_s, dr_i_j, dist, i_s, j_s, grad):
        """Compute temperature derivative for next step"""
        kernel_grad_vector = grad[:, None] * e_s
        k = (state["kappa"][i_s] * state["kappa"][j_s]) / (state["kappa"][i_s] + state["kappa"][j_s])
        a = bm.sum(dr_i_j * kernel_grad_vector, axis=1)
        F_ab = a / ((dist * dist) + EPS)
        b = 4 * state["mass"][j_s] * k * (state["T"][i_s] - state["T"][j_s])
        dTdt = (b * F_ab) / (state["Cp"][i_s] * state["rho"][i_s] * state["rho"][j_s])
        result = bm.index_add(bm.zeros_like(state["T"]), i_s, dTdt, axis=0, alpha=1)
        return result

    @staticmethod
    def wall_virtual(position, tag):
        """Neighbor relationship between solid wall particles and virtual particles"""
        vir_r = position[tag == 2]
        wall_r = position[tag == 1]
        tree = cKDTree(wall_r)
        distance, neighbors = tree.query(vir_r, k=1)
        
        fuild_len = len(position[tag == 0]) 
        neighbors = neighbors + fuild_len
        node_self = bm.where(tag == 2)[0]
        return node_self, neighbors

    @staticmethod
    def fuild_p(state, B, rho_0, c_1):
        """Updates the pressure of fluid particles"""
        f_rho = state["rho"][(state["tag"] == 0)|(state["tag"] == 3)]
        fg_p = B * (bm.exp((1 - rho_0 / f_rho) / c_1) - 1)
        fg_indx = bm.where((state["tag"] == 0)|(state["tag"] ==3))[0]
        state["p"] = bm.set_at(state["p"], fg_indx, fg_p)
        return state["p"]

    @staticmethod
    def wall_p(state, w_node, f_neighbors, w):
        """Updates the pressure of wall particles"""
        f_mass = state["mass"][f_neighbors]
        f_rho = state["rho"][f_neighbors]
        f_p = state["p"][f_neighbors] 
        w_unique = bm.unique(w_node)
    
        s0 = (f_mass / f_rho) * f_p * w
        result0 = bm.zeros(len(state["p"]), dtype=bm.float64)
        result0 = bm.index_add(result0, w_node, s0, axis=0, alpha=1)
        s1 = (f_mass / f_rho) * w
        result1 = bm.zeros(len(state["p"]), dtype=bm.float64)
        result1 = bm.index_add(result1, w_node, s1, axis=0, alpha=1)

        result0 = result0[w_unique]
        result1 = result1[w_unique]     
        state["p"] = bm.set_at(state["p"], w_unique, result0 / result1) 
        return state["p"]

    @staticmethod
    def virtual_p(state, v_node, w_neighbors):
        """Updates the pressure of virtual particles"""
        w_p = state["p"][w_neighbors]
        state["p"] = bm.set_at(state["p"], v_node, w_p)
        return state["p"]

    @staticmethod
    def sound(state, B, rho_0, c_1):
        """Update sound speed"""
        fg_tag = bm.where((state["tag"] == 0) | (state["tag"] == 3))[0]
        rho = state["rho"][fg_tag]
        value = bm.sqrt(B * bm.exp((1 - rho_0 / rho) / c_1) * (rho_0 / (c_1 * rho**2)))
        state["sound"] = bm.set_at(state["sound"], fg_tag, value)
        return state["sound"]

    @staticmethod
    def mu_wlf(state, node_self, neighbors, dw, mu_0, tau, n):
        """Calculate particle viscosity to further calculate the change in fluid particle velocity"""
        u_ji = state["u"][neighbors] - state["u"][node_self]
        s = state["mass"][neighbors][:, None, None] * bm.einsum('ij,ik->ijk', u_ji, dw)

        du = bm.zeros((len(state["u"]), 2, 2), dtype=bm.float64)
        du = bm.index_add(du, node_self, s, axis=0, alpha=1) / state["rho"][:, None, None]
        D = (du + bm.einsum('...ij->...ji', du)) / 2 #应变率
        DD = bm.einsum('...ij,...ij->...', D, D) # 计算 D:D (张量对自身的双重内积)
        rate = bm.sqrt(2 * DD) #剪切率
        mu_value = mu_0 / (1 + ((mu_0 * rate) / tau)**(1-n))
        return mu_value

    @staticmethod
    def A_matrix(state, node_self, neighbors, dr, dw):
        mass_j = state["mass"][neighbors]
        rho_j = state["rho"][neighbors]
        value = (mass_j / rho_j)[:, None, None] * bm.einsum('ij, ik->ijk', -dr, dw) #外积
        A_s = bm.zeros((len(state["mu"]), 2, 2), dtype=bm.float64)
        A_s = bm.index_add(A_s, node_self, value, axis=0, alpha=1)
        return A_s

    @staticmethod
    def kernel_grad(r, node_self, neighbors, kernel):
        """Calculate the kernel function value and its gradient"""
        EPS = bm.finfo(float).eps
        r_i_s, r_j_s = r[node_self], r[neighbors]
        dr_i_j = r_i_s - r_j_s
        dist = bm.linalg.norm(dr_i_j, axis=1)
        w_dist = bm.vmap(kernel.value)(dist)

        e_s = dr_i_j / (dist[:, None] + EPS)
        grad_w_dist_norm = bm.vmap(kernel.grad_value)(dist)
        grad_w_dist = grad_w_dist_norm[:, None] * e_s
        return w_dist, grad_w_dist, dr_i_j, dist

    @staticmethod
    def fuild_fwvg(state, node_self, neighbors, dr_i_j, dist, w_dist, grad_w_dist):
        """Neighbor relations among fluid particles, solid wall particles, and virtual particles"""
        tag = state["tag"]
        wvg_tag = bm.where((tag == 1) | (tag == 2) | (tag == 3))[0]
        wvg_indx = bm.where(bm.isin(node_self, wvg_tag))[0] 
        wvg_mask = bm.ones(len(node_self), dtype=bm.bool)
        wvg_mask = bm.set_at(wvg_mask, wvg_indx, False)
        f_node = node_self[wvg_mask]
        neighbors = neighbors[wvg_mask]
        dr_i_j = dr_i_j[wvg_mask]
        dist = dist[wvg_mask]
        w_dist = w_dist[wvg_mask]
        grad_w_dist = grad_w_dist[wvg_mask]
        return f_node, neighbors, dr_i_j, dist, w_dist, grad_w_dist

    @staticmethod
    def wall_fg(state, node_self, neighbors, w_dist):
        """Neighbor relationship between fluid particles and solid wall particles"""
        tag = state["tag"]
        fvg_tag = bm.where((tag == 0) | (tag == 2) | (tag == 3))[0]
        fvg_indx = bm.where(bm.isin(node_self, fvg_tag))[0]
        fvg_mask = bm.ones(len(node_self), dtype=bm.bool)
        fvg_mask = bm.set_at(fvg_mask, fvg_indx, False)
        w_node = node_self[fvg_mask]
        fwvg_neighbors = neighbors[fvg_mask]
        w_dist = w_dist[fvg_mask]

        wv_tag = bm.where((tag == 1) | (tag == 2))[0]
        wv_indx = bm.where(bm.isin(fwvg_neighbors, wv_tag))[0]
        wv_mask = bm.ones(len(fwvg_neighbors), dtype=bm.bool)
        wv_mask = bm.set_at(wv_mask, wv_indx, False)
        w_node = w_node[wv_mask]
        fg_neighbors = fwvg_neighbors[wv_mask]
        w_dist = w_dist[wv_mask]
        return w_node, fg_neighbors, w_dist

    @staticmethod
    def vtag_u(state, v_node, w_neighbors, w_node, f_neighbors, w):
        mass_j = state["mass"][f_neighbors]
        rho_j = state["rho"][f_neighbors]
        u_j = state["u"][f_neighbors]
        u_wall = state["u"][w_neighbors]
        
        sum1 = bm.zeros((len(state["tag"]), 2), dtype=bm.float64)
        s1 = (mass_j / rho_j)[:, None] * u_j * w[:, None]
        sum1 = bm.index_add(sum1, w_node, s1, axis=0, alpha=1)
        sum2 = bm.zeros((len(state["tag"]),), dtype=bm.float64)
        s2 = (mass_j / rho_j) * w
        sum2 = bm.index_add(sum2, w_node, s2, axis=0, alpha=1)
        u_ave = sum1 / (sum2[:, None]+EPS) #防止分母为0，加上EPS
        
        virtual_u = 2 * u_wall - u_ave[w_neighbors]
        state["u"] = bm.set_at(state["u"], v_node, virtual_u)
        return state["u"]

    @staticmethod
    def free_surface(state, node_self, neighbors, dr, dist, w, dw, A_s, h):
        """Label the three types of fluid particles: linear motion, interior, and exact free surface"""
        counts = bm.unique(node_self, return_counts=True)[1]
        f_tag = bm.where(state["tag"] == 0)[0]
        
        #找精确自由表面流体粒子的索引
        a = (state["mass"][neighbors] / state["rho"][neighbors]) * w
        C_i = bm.zeros((len(state["tag"]), ), dtype=bm.float64)
        C_i = bm.index_add(C_i, node_self, a, axis=0, alpha=1)
        b = (state["mass"][neighbors] / state["rho"][neighbors])[:, None] * dw
        dC_i = bm.zeros((len(state["tag"]), 2), dtype=bm.float64)
        dC_i = bm.index_add(dC_i, node_self, b, axis=0, alpha=1)
        
        idx_c = bm.where(C_i[state["tag"] == 0] < 0.85)[0] #0.75 TODO
        bool1 = bm.zeros_like(C_i[state["tag"] == 0], dtype=bm.bool)
        bool1 = bm.set_at(bool1, idx_c, True)
        
        A_dC = bm.einsum('ijk,ik->ij', -A_s, dC_i) #A_s TODO
        norm_A_dC = bm.linalg.norm(A_dC, axis=1, keepdims=True)
        normal = A_dC / (norm_A_dC + EPS) #外法向量，防止分母为0，加上EPS
        rotate = bm.array([[0,-1],[1,0]], dtype=bm.float64)
        rotate = bm.broadcast_to(rotate, (len(state["tag"]),2,2)) 
        tau = bm.einsum('ijk,ik->ij', rotate, normal) #与外法向量垂直的单位向量
        node_T = state["position"][state["tag"] == 0] + normal[state["tag"] == 0] * h
        node_T = bm.repeat(node_T, counts, axis=0)
        x_jT = state["position"][neighbors] - node_T
        
        cond1 = dist >= bm.sqrt(bm.array([2], dtype=bm.float64)) * h
        cond2 = bm.linalg.norm(x_jT, axis=1, keepdims=False) < h
        cond3 = dist < bm.sqrt(bm.array([2], dtype=bm.float64)) * h
        cond4 = bm.abs(bm.einsum('ij,ij->i', normal[neighbors], x_jT)) + bm.abs(bm.einsum('ij,ij->i', tau[neighbors], x_jT)) < h

        mask1 = (cond1 & cond2) | (cond3 & cond4)
        mask = bm.zeros((len(state["tag"]),), dtype=bool)
        mask = bm.index_add(mask, node_self, mask1)
        free_mask = ~(mask[f_tag])
        free_mask = bool1 & free_mask
        free = bm.where(free_mask == True)[0]

        wvg_tag = bm.where((state["tag"] == 1) | (state["tag"] == 2) | (state["tag"] == 3))[0]
        wvg_indx = bm.where(bm.isin(node_self, wvg_tag))[0]
        wvg_mask = bm.ones(len(node_self), dtype=bm.bool)
        wvg_mask = bm.set_at(wvg_mask, wvg_indx, False)
        node = node_self[wvg_mask]
        neig = neighbors[wvg_mask]
        free_indices = bm.where(bm.isin(node, free))[0]
        neig_free = bm.unique(neig[free_indices])
        
        #找内部流体粒子的索引
        mask3 = ~bm.isin(bm.where(state["tag"] == 0)[0], neig_free)
        in_f = f_tag[mask3]
        return in_f, neig_free, dC_i, normal

    @staticmethod
    def gate_change(state, dx, domain, H, u_in, rho_0, dt):
        g_tag = state["tag"] == 3
        state["position"] = bm.set_at(state["position"], g_tag, state["position"][g_tag] + state["u"][g_tag] * dt)

        out_of_bounds = bm.where((state["position"][:, 0] > domain[0]) & g_tag)[0]
        state["tag"] = bm.set_at(state["tag"], out_of_bounds, 0)
        
        if out_of_bounds.shape[0] > 0:
            y_new = bm.arange(domain[2] + dx, domain[3], dx, dtype=bm.float64)
            new_r = bm.concatenate((bm.full((len(y_new), 1), domain[0] - 4 * H), y_new.reshape(-1, 1)), axis=1)
            num = len(new_r)
            tag = bm.full((num,), 3, dtype=bm.int64)
            u = bm.tile(u_in, (num, 1))
            rho = bm.full((num,), rho_0, dtype=bm.float64)
            mass_0 = rho_0 * dx * (domain[3]-domain[2]) / num
            mass = bm.full((num,), mass_0, dtype=bm.float64)
            state["position"] =  bm.concatenate((state["position"], new_r), axis=0)
            state["tag"] = bm.concatenate((state["tag"], tag), axis=0)
            state["u"] = bm.concatenate((state["u"], u), axis=0)
            state["dudt"] = bm.concatenate((state["dudt"], bm.zeros((num, 2), dtype=bm.float64)), axis=0)
            state["rho"] = bm.concatenate((state["rho"], rho), axis=0)
            state["drhodt"] = bm.concatenate((state["drhodt"], bm.zeros((num,), dtype=bm.float64)), axis=0)
            state["p"] = bm.concatenate((state["p"], bm.zeros((num,), dtype=bm.float64)), axis=0)
            state["sound"] = bm.concatenate((state["sound"], bm.zeros((num,), dtype=bm.float64)), axis=0)
            state["mass"] = bm.concatenate((state["mass"], mass), axis=0)
            state["mu"] = bm.concatenate((state["mu"], bm.zeros((num,), dtype=bm.float64)), axis=0)
            state["drdt"] = bm.concatenate((state["drdt"], bm.zeros((num, 2), dtype=bm.float64)), axis=0)
        return state

    @staticmethod
    def change_rho(state, node_self, neighbors, dr, dist, dw, A_s):
        """Calculate the density change of fluid particles"""
        mask0 = node_self == neighbors
        node_self = node_self[~mask0]
        neighbors = neighbors[~mask0]
        dr = dr[~mask0]
        dist = dist[~mask0]
        dw = dw[~mask0]

        tag = state["tag"]
        mass_j = state["mass"][neighbors]
        rho_j = state["rho"][neighbors]
        u_ij = state["u"][node_self] - state["u"][neighbors]
        p_ij = state["p"][node_self] - state["p"][neighbors]
        rho = (state["rho"][node_self] + state["rho"][neighbors]) / 2
        c = (state["sound"][node_self] + state["sound"][neighbors]) / 2

        a = (mass_j / rho_j)[:, None] * (u_ij + (p_ij / (rho * c))[:, None] * (dr / (dist+EPS)[:, None]))
        
        A_s_n = A_s[node_self]
        cond_A_s = bm.linalg.cond(A_s_n)
        mask1 = cond_A_s >= 1e15
        if not mask1.all(): 
            corrected_dw = bm.linalg.solve(A_s_n[~mask1], dw[~mask1][..., None]).squeeze(-1) #核梯度修正
            dw = bm.set_at(dw, ~mask1, corrected_dw)

        s = bm.einsum('ij,ij->i', a, dw) 
        result = bm.zeros_like(state["drhodt"], dtype=bm.float64)
        result = bm.index_add(result, node_self, s, axis=0, alpha=1)

        f_tag = bm.where(tag == 0)[0]
        result = result[f_tag]
        rho_i = state["rho"][f_tag]
        state["drhodt"] = bm.set_at(state["drhodt"], f_tag, rho_i * result)
        return state["drhodt"]

    @staticmethod
    def change_u(state, node_self, neighbors, dist, dr, dw, h, eta, A_s):
        """Calculate the velocity change of fluid particles"""
        mask0 = node_self == neighbors
        node_self = node_self[~mask0]
        neighbors = neighbors[~mask0]
        dr = dr[~mask0]
        dist = dist[~mask0]
        dw = dw[~mask0]

        mass_j = state["mass"][neighbors]
        rho_i = state["rho"][node_self]
        rho_j = state["rho"][neighbors]
        c_i = state["sound"][node_self]
        c_j = state["sound"][neighbors]
        p = state["p"][node_self] + state["p"][neighbors]
        u_ij = state["u"][node_self] - state["u"][neighbors]
        mu = state["mu"][node_self] + state["mu"][neighbors]
        
        A_s_n = A_s[node_self]
        cond_A_s = bm.linalg.cond(A_s_n)
        mask1 = cond_A_s >= 1e15
        if not mask1.all(): 
            corrected_dw = bm.linalg.solve(A_s_n[~mask1], dw[~mask1][..., None]).squeeze(-1) #核梯度修正
            dw = bm.set_at(dw, ~mask1, corrected_dw)
        
        a1 = (eta * mu) / (dist+EPS) #防止分母为0，加上EPS
        a2 = ((rho_i + rho_j)/2) * ((c_i + c_j)/2) 
        beta_rs = bm.where(a1 < a2, a1, a2)
        
        b1 = p - beta_rs * bm.sum(dr * u_ij, axis=1) / (dist+EPS) #点积，以及防止分母为0，加上EPS
        b2 = mass_j / (rho_i * rho_j)
        b3 = (b1 * b2)[:, None] * dw
        sum1 = bm.zeros_like(state["u"], dtype=bm.float64)
        sum1 = -bm.index_add(sum1, node_self, b3, axis=0, alpha=1)
        
        c1 = (mass_j * mu) / (rho_i * rho_j)
        c2 = (bm.sum(dr * dw, axis=1) / (dist**2 + (0.01*h)**2))[:, None] * u_ij
        c3 = c1[:, None] * c2
        sum2 = bm.zeros_like(state["u"], dtype=bm.float64)
        sum2 = bm.index_add(sum2, node_self, c3, axis=0, alpha=1)
        
        value = sum1 + sum2
        f_tag = bm.where(state["tag"] == 0)[0]
        state["dudt"] = bm.set_at(state["dudt"], f_tag, value[state["tag"] == 0])
        return state["dudt"]

    @staticmethod
    def shifting_r(state, in_f, free, dC_i, normal, dt, h):
        """Update displacement through displacement technology"""
        drdt = state["drdt"]
        u = state["u"]

        #内部流体粒子运动
        delta_in = (-5 * h * bm.linalg.norm(u[in_f], axis=1))[:, None] * dC_i[in_f] * dt
        drdt = bm.set_at(drdt, in_f, bm.asarray(delta_in, dtype=drdt.dtype))
        
        #精确自由表面流体粒子运动
        I = bm.eye(2, dtype=bm.float64)
        I = bm.broadcast_to(I, (len(free), 2, 2))
        n_outer = bm.einsum('ij,ik->ijk', normal[free], normal[free])  #外积
        p_dC = bm.einsum('ijk,ik->ij', I - n_outer, dC_i[free])
        delta_free = (-5 * h * bm.linalg.norm(u[free], axis=1))[:, None] * p_dC * dt
        drdt = bm.set_at(drdt, free, bm.asarray(delta_free, dtype=drdt.dtype))
        return drdt

    @staticmethod
    def draw(state, i):
        plt.clf
        w_tag = np.array(state["tag"] == 1)
        v_tag = np.array(state["tag"] == 2)
        g_tag = np.array(state["tag"] == 3)
        color = np.where(w_tag, 'red', np.where(v_tag, 'green', np.where(g_tag, 'black', 'blue')))
        c = np.array(state['u'][:,0])
        #c = np.array(state['rho'])
        #c = np.array(state['p'])
        plt.figure(figsize=(20,2))

        plt.scatter(np.array(state['position'][:, 0]), np.array(state['position'][:, 1]), c=c, cmap='jet', s=5)
        plt.colorbar(cmap='jet')
        plt.clim(-7,7)
        
        plt.title(f"Time Step: {i}")
        fname = 'frames/' + 'test_'+ str(i+1).zfill(10) + '.png'
        plt.savefig(fname)

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
