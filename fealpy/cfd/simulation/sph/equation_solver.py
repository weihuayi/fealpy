from fealpy.backend import backend_manager as bm
from fealpy.cfd.simulation.sph_base import Kernel

class EquationSolver():
    def __init__(self, Vmax=1.0, rho0=1.0, gamma=1.0):
        self.Vmax = Vmax          # 最大速度
        self.rho0 = rho0          # 参考密度
        self.gamma = gamma        
        self.c0 = 10 * Vmax       # 声速

    def simualtion(self):
        pass
    
    def mass_equation_solve(self, improve, state, neighbors, w_ij):
        """质量守恒方程求解"""
        if not improve:
            return self.rho_tradition(state, neighbors, w_ij)
        else:
            # 可扩展改进方法
            raise NotImplementedError("Improved method not implemented yet.")

    def momentum_equation_solve(self, improve, state, neighbors, self_node, dr, dist, grad_w, p):
        """动量守恒方程求解"""
        if not improve:
            return self.u_tradition(state, neighbors, self_node, dr, dist, grad_w, p)
        else:
            # 可扩展改进方法
            raise NotImplementedError("Improved momentum method not implemented.")
        

    def constitutive_equation(self, method, state, **kwargs):
        """选择本构方程"""
        if method == "tait_eos":
            return self.tait_eos(state, **kwargs)
        else:
            # 在此添加其他本构方程
            raise NotImplementedError(f"Constitutive method '{method}' is not implemented.")

    def rho_tradition(self, state, self_node, w_ij, dtype=bm.float64):
        """传统密度更新方法"""
        mass = state["mass"]
        a = bm.zeros_like(mass, dtype=dtype)
        return mass * bm.index_add(a, self_node, w_ij, axis=0, alpha=1)

    def tait_eos(self, state, c0=None, rho0=None, gamma=None, X=0.0):
        """本构方程计算压力"""
        rho = state['rho']
        c0 = self.c0 if c0 is None else c0
        rho0 = self.rho0 if rho0 is None else rho0
        gamma = self.gamma if gamma is None else gamma
        return gamma * c0**2 * ((rho / rho0)**gamma - 1) / rho0 + X

    def u_tradition(self, state, neighbors, self_node, dr, dist, grad_w, p):
        """"传统动量更新方法"""
        EPS = bm.finfo(float).eps
        def compute_A(rho, mv, tv):
            a = rho[:, bm.newaxis] * mv
            dv = tv - mv
            result = bm.einsum('ki,kj->kij', a, dv).reshape(a.shape[0], 2, 2)
            return result

        eta_i = state["eta"][neighbors]
        eta_j = state["eta"][self_node]
        p_i = p[neighbors]
        p_j = p[self_node]
        rho_i = state["rho"][neighbors]
        rho_j = state["rho"][self_node]
        m_i = state["mass"][neighbors]
        m_j = state["mass"][self_node]
        mv_i = state["mv"][neighbors]
        mv_j = state["mv"][self_node]
        tv_i = state["tv"][neighbors]
        tv_j = state["tv"][self_node]   

        volume_square = ((m_i/rho_i)**2 + (m_j/rho_j)**2) / m_i
        eta_ij = 2 * eta_i * eta_j / (eta_i + eta_j + EPS)
        p_ij = (rho_j * p_i + rho_i * p_j) / (rho_i + rho_j)
        c = volume_square * grad_w / (dist + EPS)
        A = (compute_A(rho_i, mv_i, tv_i) + compute_A(rho_j, mv_j, tv_j))/2
        mv_ij = mv_i -mv_j
        b = bm.sum(A * dr[:, bm.newaxis, :], axis=2)
        a = c[:, None] * (-p_ij[:, None] * dr + b + (eta_ij[:, None] * mv_ij))
        add_dv = bm.zeros_like(state["mv"], dtype=bm.float64)
        return bm.index_add(add_dv, neighbors, a, axis=0, alpha=1)


'''
class SPHSimulation(SimulationBase):

class SPHParameters(SimulationParameters):
'''