#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: solevr.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Tue 18 Jun 2024 10:45:15 AM CST
	@bref 
	@ref 
'''  
from jax import ops, vmap, jit, lax
import jax.lax as lax 
from functools import partial
import jax.numpy as jnp
import numpy as np
import h5py
import pyvista
from typing import Dict
import enum
from fealpy.jax.sph.jax_md import space
#from jax_md import space
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

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
    @staticmethod
    @jit
    def tait_eos(rho, c0, rho0, gamma=1.0, X=0.0):
        """Equation of state update pressure"""
        return gamma * c0**2 * ((rho/rho0)**gamma - 1) / rho0 + X
    
    @staticmethod
    def tait_eos_p2rho(p, p0, rho0, gamma=1.0, X=0.0):
        """Calculate density by pressure."""
        p_temp = p + p0 - X
        return rho0 * (p_temp / p0) ** (1 / gamma)
    
    #计算密度
    @staticmethod
    @jit
    def compute_rho(mass, i_node, w_ij):
        """Density summation."""
        return mass * ops.segment_sum(w_ij, i_node, len(mass))

    #计算运输速度的加速度
    @staticmethod
    @jit
    def compute_tv_acceleration(state, i_node, j_node, r_ij, dij, grad, pb):
        m_i = state["mass"][i_node]
        m_j = state["mass"][j_node]
        rho_i = state["rho"][i_node]
        rho_j = state["rho"][j_node]

        volume_square = ((m_i / rho_i) ** 2 + (m_j / rho_j) ** 2) / m_i
        c = volume_square * grad / (dij + EPS)
        a = c[:, None] * pb[i_node][:, None] * r_ij       
        return ops.segment_sum(a, i_node, len(state["mass"]))


    #计算动量速度的加速度
    @staticmethod
    @jit 
    def compute_mv_acceleration(state, i_node, j_node, r_ij, dij, grad, p):
        def compute_A(rho, mv, tv):
            a = rho[:, jnp.newaxis] * mv
            dv = tv - mv
            result = jnp.einsum('ki,kj->kij', a, dv).reshape(a.shape[0], 2, 2)
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
        b = jnp.sum(A * r_ij[:, jnp.newaxis, :], axis=2)
        a = c[:, None] * (-p_ij[:, None] * r_ij + b + (eta_ij[:, None] * mv_ij))
        return ops.segment_sum(a, i_node, len(state["mass"]))
    
    @staticmethod
    @jit
    def boundary_conditions(state, box_size, n_walls=3, dx=0.02, T0=1.0, hot_T=1.23): 
        fluid = state["tag"] == 0

        # 将输入流体温度设置为参考温度
        inflow = fluid * (state["position"][:, 0] < n_walls * dx)
        state["T"] = jnp.where(inflow, T0, state["T"])
        state["dTdt"] = jnp.where(inflow, 0.0, state["dTdt"])

        # 设置热壁温度
        hot = state["tag"] == 3
        state["T"] = jnp.where(hot, hot_T, state["T"])
        state["dTdt"] = jnp.where(hot, 0.0, state["dTdt"])

        # 将墙设置为参考温度
        solid = state["tag"] == 1
        state["T"] = jnp.where(solid, T0, state["T"])
        state["dTdt"] = jnp.where(solid, 0, state["dTdt"])

        # 确保静态墙没有速度或加速度
        static = (hot + solid)[:, None]
        state["mv"] = jnp.where(static, 0.0, state["mv"])
        state["tv"] = jnp.where(static, 0.0, state["tv"])
        state["dmvdt"] = jnp.where(static, 0.0, state["dmvdt"])
        state["dtvdt"] = jnp.where(static, 0.0, state["dtvdt"])

        # 将出口温度梯度设置为零，以避免与流入相互作用
        bound = jnp.array([jnp.zeros_like(box_size), box_size]).T
        outflow = fluid * (state["position"][:, 0] > bound[0, 1] - n_walls * dx)
        state["dTdt"] = jnp.where(outflow, 0.0, state["dTdt"])

        return state

    @staticmethod
    def external_acceleration(position, box_size, dx=0.02):
        @jit
        def jit_external_acceleration(position, box_size, n_walls=3, dx=0.02, g_ext=2.3):
            dxn = n_walls * dx
            res = jnp.zeros_like(position)
            force = jnp.ones((len(position)))
            fluid = (position[:, 1] < box_size[1] - dxn) * (position[:, 1] > dxn)
            force = jnp.where(fluid, force, 0)
            res = res.at[:, 0].set(force)
            return res * g_ext
        return jit_external_acceleration(position, box_size, dx=dx)
    
    @staticmethod
    @jit
    def enforce_wall_boundary(state, p, g_ext, i_s, j_s, w_dist, dr_i_j, c0=10.0, rho0=1.0, X=5.0, p0=100.0, with_temperature=False):
        """Enforce wall boundary conditions by treating boundary particles in a special way."""
        mask_bc = jnp.isin(state["tag"], jnp.array([1, 3]))
        mask_j_s_fluid = jnp.where(state["tag"][j_s] == 0, 1.0, 0.0)
        w_j_s_fluid = w_dist * mask_j_s_fluid
        w_i_sum_wf = ops.segment_sum(w_j_s_fluid, i_s, len(state["position"]))
    
        # no-slip boundary condition
        def no_slip_bc(x):
            x_wall_unnorm = ops.segment_sum(w_j_s_fluid[:, None] * x[j_s], i_s, len(state["position"]))
            x_wall = x_wall_unnorm / (w_i_sum_wf[:, None] + EPS)
            x = jnp.where(mask_bc[:, None], 2 * x - x_wall, x)
            return x
        mv = no_slip_bc(state["mv"])
        tv = no_slip_bc(state["tv"])
    
        # Pressure summation
        p_wall_unnorm = ops.segment_sum(w_j_s_fluid * p[j_s], i_s, len(state["position"]))
    
        # External acceleration term
        rho_wf_sum = (state["rho"][j_s] * w_j_s_fluid)[:, None] * dr_i_j
        rho_wf_sum = ops.segment_sum(rho_wf_sum, i_s, len(state["position"]))
        p_wall_ext = (g_ext * rho_wf_sum).sum(axis=1)
    
        # Normalize pressure
        p_wall = (p_wall_unnorm + p_wall_ext) / (w_i_sum_wf + EPS)
        p = jnp.where(mask_bc, p_wall, p)
    
        rho = SPHSolver.tait_eos_p2rho(p, p0, rho0, X=5.0)
    
        def compute_temperature():
            t_wall_unnorm = ops.segment_sum(w_j_s_fluid * state["T"][j_s], i_s, len(state["position"]))
            t_wall = t_wall_unnorm / (w_i_sum_wf + EPS)
            mask = jnp.isin(state["tag"], jnp.array([1, 2]))
            t_wall = jnp.where(mask, t_wall, state["T"])
            return t_wall

        T = lax.cond(with_temperature, compute_temperature, lambda: state["T"])

        return p, rho, mv, tv, T

    @staticmethod
    @partial(jit, static_argnames=('kernel',))
    def temperature_derivative(state, kernel, e_s, dr_i_j, dist, i_s, j_s, grad):
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

    @staticmethod
    @jit
    def wall_extrapolation(state, kernel, i_s, j_s, fw_indices, fw_indices_wall):
        # 只更新流体粒子对固壁粒子的影响
        fw_m = state["mass"][fw_indices]
        fw_rho = state["rho"][fw_indices]
        fw_v = state["v"][fw_indices]
        v_wall = jnp.zeros_like(fw_v)

        sum0 = (fw_m[j_s] / fw_rho[j_s])[:, None] * fw_v[j_s] * kernel[:, None]
        sum1 = (fw_m[j_s] / fw_rho[j_s]) * kernel
        a = ops.segment_sum(sum0, i_s, len(fw_m))
        b = ops.segment_sum(sum1, i_s, len(fw_m))
        v_ave = a / b[:, None]
        result = 2 * v_wall - v_ave

        result = result[fw_indices_wall]
        return result

    def gate_change(self, state, domain, dt, dx, uin):
        H = 1.5 * dx
        dy = dx
        rho0 = 737.54

        #更新位置
        g_idx = jnp.where(state["tag"] == 3)[0]
        r = state["position"][g_idx] + dt * state["v"][g_idx] 
        state["position"] = state["position"].at[state["tag"]==3].set(r)

        #将进入流体区域的门粒子更换标签为0
        g_x = state["position"][:, 0]
        g_i = jnp.where((state["tag"] == 3) & (g_x >= 0))[0] 
        
        if g_i.size > 0:
            state["tag"] = state["tag"].at[g_i].set(0)

            #更新物理量
            y = jnp.arange(domain[2]+dx, domain[3], dx)
            gp = jnp.column_stack((jnp.full_like(y, domain[0]-4*H), y))
            state["position"] = jnp.vstack((gp, state["position"]))
            tag_g = jnp.full((gp.shape[0],), 3,dtype=int)
            state["tag"] = jnp.concatenate((tag_g, state["tag"]))
            v_g = jnp.ones_like(gp) * uin
            state["v"] = jnp.concatenate((v_g, state["v"]))
            state["rho"] = jnp.concatenate((jnp.ones(gp.shape[0])*rho0, state["rho"]))
            state["p"] = jnp.concatenate((jnp.zeros_like(tag_g), state["p"]))
            state["sound"] = jnp.concatenate((jnp.zeros_like(tag_g), state["sound"]))
            mass_g = jnp.ones(gp.shape[0]) * dx * dy * rho0
            state["mass"] = jnp.concatenate((mass_g, state["mass"]))
        
        return state

    def free_surface(self, state, i_s, j_s, kernel, grad_kernel, h):
        #计算流体粒子的浓度
        m_j = state["mass"][j_s]
        rho_j = state["rho"][j_s]
        ci = (m_j/rho_j) * kernel
        grad_ci = (m_j/rho_j)[:, None] * grad_kernel 
        #grad_ci = ops.segment_sum(grad_ci, i_s, len(state["position"][state["tag"] == 0]))
        w_tag = state['tag'] == 1
        d_tag = state['tag'] == 2
        NN_wd = (jnp.where(w_tag==True)[0]).size + (jnp.where(d_tag==True)[0]).size
        
        #找浓度<0.75的流体粒子的索引
        result = ops.segment_sum(ci, i_s, len(state["position"]))
        result = result[state["tag"] == 0]
        #idx = jnp.where(result < 0.75)[0] + NN_wd
        idx = jnp.where(result< 0.75)[0]
        
        #判断区域内是否有其他的粒子
        xji = state["position"][j_s] - state["position"][i_s]
        gx = jnp.einsum('ki,kj->kij', xji, grad_kernel).reshape(xji.shape[0], 2, 2)
        As = (m_j/rho_j).reshape(xji.shape[0],1,1) * gx
        #As = ops.segment_sum(As, i_s, len(state["position"][state["tag"] == 0]))
        normal = jnp.einsum('ijk,ik->ij', As, grad_ci) / \
                 jnp.linalg.norm(jnp.einsum('ijk,ik->ij', As, grad_ci), axis=1, keepdims=True)
        a = jnp.array([-normal[:, 1], normal[:, 0]]).T
        b = jnp.linalg.norm(a, axis=1)
        perpen = a / b[:, jnp.newaxis]
        nodeT = state["position"][i_s] + normal*h
       
        cond1 = jnp.linalg.norm(-xji, axis=1) >= jnp.sqrt(2)*h
        cond2 = jnp.linalg.norm(state["position"][j_s] - nodeT) < h
        cond3 = jnp.linalg.norm(-xji, axis=1) < jnp.sqrt(2)*h
        cond4 = (jnp.abs(jnp.dot(normal, (state["position"][j_s] - nodeT).T)).sum()) + \
                (jnp.abs(jnp.dot(perpen, (state["position"][j_s] - nodeT).T)).sum()) < h
        
        cond = ~(cond1 & cond2) | ~(cond3 & cond4)
        is_free = idx[cond[idx]]
        return is_free

    def change_p(self, state, kernel, i_s, j_s, d_s, B, rho0, c1):
        #更新流体粒子的压力和声速
        p_rho = B * (jnp.exp((jnp.ones_like(state["rho"])-rho0/state["rho"])/c1)-1)
        f_tag = jnp.where(state["tag"] == 0)[0]
        state["p"] = state["p"].at[f_tag].set(p_rho[f_tag])
        sound = jnp.sqrt(B * (rho0/(c1*state["rho"]**2)) * jnp.exp((jnp.ones_like(state["rho"])-rho0/state["rho"])/c1))
        state["sound"] = state["sound"].at[f_tag].set(sound[f_tag])

        #计算固壁粒子的压力
        w_tag = jnp.where(state["tag"] == 1)[0]
        fw_m = state["mass"][(state["tag"] == 0) | (state["tag"] == 1)]
        fw_rho = state["rho"][(state["tag"] == 0) | (state["tag"] == 1)]
        fw_p = state["p"][(state["tag"] == 0) | (state["tag"] == 1)]
        sum0 = (fw_m[j_s]*fw_p[j_s]/fw_rho[j_s]) * kernel
        sum1 = (fw_m[j_s]/fw_rho[j_s]) * kernel
        a = ops.segment_sum(sum0, i_s, len(fw_m))
        b = ops.segment_sum(sum1, i_s, len(fw_m))
        p_w = a / b
        fw_tag = state["tag"][(state["tag"] == 0) | (state["tag"] == 1)]
        idx = jnp.where(fw_tag == 1)[0]
        state["p"] = state["p"].at[w_tag].set(p_w[idx])
    
        #计算虚粒子的压力
        d_tag = jnp.where(state["tag"] == 2)[0]
        state["p"] = state["p"].at[d_tag].set(state["p"][d_s])
        return state

    @staticmethod
    @jit
    def continue_equation(state, i_s, j_s, grad_kernel, f_tag, f_rho):
        v = state["v"]
        x = state["position"]
        rho = state["rho"]
        c = state["sound"]
        p = state["p"]
        m = state["mass"]

        v_ij = v[i_s] - v[j_s]
        x_ij = x[i_s] - x[j_s]
        r_ij = jnp.linalg.norm(x_ij, axis=1)
        rho_c = ((rho[i_s] + rho[j_s]) / 2) * ((c[i_s] + c[j_s]) / 2)
        p_ij = p[i_s] - p[j_s]

        a = jnp.sum(((v_ij + (p_ij / rho_c)[:, None] * (x_ij / r_ij[:, None])) * grad_kernel), axis=1, keepdims=True)
        sum0 = (m[j_s] / rho[j_s])[:, None] * a
        sum1 = ops.segment_sum(sum0, i_s, len(rho))[f_tag]
        result = f_rho[:, None] * sum1
        result = jnp.squeeze(result)
        return result

    @staticmethod
    @jit
    def mu_wlf(state, i_s, j_s, grad_kernel, mu0, tau, n):
        a = state["v"][j_s][:,:,jnp.newaxis] * grad_kernel[:,jnp.newaxis,:]
        a_t = jnp.transpose(a, (0, 2, 1))
        D = (a + a_t) / 2
        gamma = jnp.sqrt(2 * jnp.einsum('ijk,ijk->i', D, D))
        gamma = ops.segment_sum(gamma, i_s, len(state["rho"]))
        mu = mu0 / (1 + (mu0 * gamma / tau)**(1-n))    
        return mu

    @staticmethod
    @jit
    def momentum_equation(state, i_s, j_s, grad_kernel, mu, eta, h, f_tag):
        x_ij = state["position"][i_s] - state["position"][j_s]
        r_ij = jnp.linalg.norm(x_ij, axis=1)
        v_ij = state["v"][i_s] - state["v"][j_s]
        a0 = eta * (mu[i_s] + mu[j_s]) / r_ij
        a1 = ((state["rho"][i_s] + state["rho"][j_s]) / 2) * ((state["sound"][i_s] + state["sound"][j_s]) / 2)
        beta = jnp.minimum(a0, a1)

        b0 = state["mass"][j_s] / (state["rho"][i_s] * state["rho"][j_s])
        b1 = state["p"][i_s] + state["p"][j_s] - beta * (jnp.einsum('ij,ij->i', x_ij, v_ij) / r_ij)
        sum0 = (b0 * b1)[:, None] * grad_kernel
        sum0 = ops.segment_sum(sum0, i_s, len(state["rho"]))

        m_j = state["mass"][j_s]
        c0 = (mu[i_s] + mu[j_s]) / (state["rho"][i_s] * state["rho"][j_s])
        c1 = jnp.einsum('ij,ij->i', x_ij, grad_kernel) / (r_ij**2 + (0.01 * h)**2)
        sum1 = (m_j * c0 * c1)[:, None] * v_ij
        sum1 = ops.segment_sum(sum1, i_s, len(state["rho"]))

        result = (-sum0 + sum1)[f_tag]
        return result

    @staticmethod
    @jit
    def change_position(state, i_s, j_s, kernel, f_tag):
        v_ji = state["v"][j_s] - state["v"][i_s]
        rho_ij = (state["rho"][i_s] + state["rho"][j_s]) / 2
        sum0 = (state["mass"][j_s] * kernel / rho_ij)[:, None] * v_ji
        sum0 = 0.5 * ops.segment_sum(sum0, i_s, len(state["position"]))
        result = (state["v"] + sum0)[f_tag]
        return result

    def create_animation(self, state_history, output_file='animation.gif'):
        fig, ax = plt.subplots(figsize=(20, 5))

        def update(frame):
            ax.clear()
            state = state_history[frame]
            positions = np.array(state["position"])  # 确保数据为 NumPy 数组
            tags = np.array(state["tag"])
            velocities = np.array(state["v"])

            # 计算每个粒子的速度大小
            speed = np.linalg.norm(velocities, axis=1)

            # 画出不同标签的粒子
            fluid_particles = positions[tags == 0]
            wall_particles = positions[tags == 1]
            ghost_particles = positions[tags == 2]
            gate_particles = positions[tags == 3]

            fluid_speeds = speed[tags == 0]
            wall_speeds = speed[tags == 1]
            ghost_speeds = speed[tags == 2]
            gate_speeds = speed[tags == 3]
            
            sc = ax.scatter(fluid_particles[:, 0], fluid_particles[:, 1], c=fluid_speeds, cmap='viridis', s=10, label='Fluid')
            ax.scatter(wall_particles[:, 0], wall_particles[:, 1], c=wall_speeds, cmap='viridis', s=10, label='Wall')
            ax.scatter(ghost_particles[:, 0], ghost_particles[:, 1], c=ghost_speeds, cmap='viridis', s=10, label='Ghost')
            ax.scatter(gate_particles[:, 0], gate_particles[:, 1], c=gate_speeds, cmap='viridis', s=10, label='Gate')

            ax.legend()
            ax.set_xlim(-0.002, 0.05)
            ax.set_ylim(-0.002, 0.006)
            ax.set_title(f"Frame {frame}")

        ani = animation.FuncAnimation(fig, update, frames=len(state_history), repeat=False)
        plt.show()
        #writer = PillowWriter(fps=20)  # 设置每秒帧数
        #ani.save(output_file, writer=writer)

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
    
    def forward_wrapper(self, displacement, kernel):
        def forward(state, neighbors): 
            r = state["position"]
            i_s, j_s = neighbors.idx
            r_i_s, r_j_s = r[i_s], r[j_s]
            dr_i_j = vmap(displacement)(r_i_s, r_j_s)
            dist = space.distance(dr_i_j)
            w_dist = vmap(kernel.value)(dist)  
            e_s = dr_i_j / (dist[:, None] + EPS) 
            grad_w_dist_norm = vmap(kernel.grad_value)(dist)
            grad_w_dist = grad_w_dist_norm[:, None] * e_s
            
            state['rho'] = self.compute_rho(state['mass'], i_s, w_dist)
            p = self.tait_eos(state['rho'],10,1)
            
            state["dmvdt"] = self.compute_mv_acceleration(\
                state, i_s, j_s, dr_i_j, dist, grad_w_dist_norm, p)
            
            return state
        return forward

    def write_vtk(self, data_dict: Dict, path: str):
        """Store a .vtk file for ParaView."""
        data_pv = self.dict2pyvista(data_dict)
        data_pv.save(path)

def TimeLine(model, shift_fn):
    def advance(dt, state, neighbors):
        state["mv"] += 1.0 * dt * state["dmvdt"]
        state["tv"] = state["mv"] 

        # 2. Integrate position with velocity v
        state["position"] = shift_fn(state["position"], 1.0 * dt * state["tv"])

        # 3. Update neighbor list
        neighbors = neighbors.update(state["position"], num_particles=state["position"].shape[0])

        # 4. Compute accelerations
        state = model(state, neighbors)

        # 5. Impose boundary conditions on dummy particles (if applicable)
        #state = bc_fn(state)
        return state, neighbors

    return advance