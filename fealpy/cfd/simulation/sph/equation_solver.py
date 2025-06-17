from fealpy.backend import backend_manager as bm
from fealpy.cfd.simulation.sph.sph_base import Kernel

EPS = bm.finfo(float).eps
class EquationSolver():
    def __init__(self, Vmax=1.0, rho0=1.0, gamma=1.0, X=0.0, B=5.914e7, c1=0.0894):
        self.Vmax = Vmax          # 最大速度
        self.rho0 = rho0          # 参考密度
        self.gamma = gamma        
        self.c0 = 10 * Vmax       # 声速
        self.X = X
        self.B = B
        self.c1 = c1
    
    def mass_equation_solve(self, improve, state, neighbors, w_ij):
        """质量守恒方程求解"""
        if improve == 0:
            return self.rho_tradition(state, neighbors, w_ij)
        else:
            # 可扩展改进方法
            raise NotImplementedError("Improved method not implemented yet.")

    def rho_tradition(self, state, self_node, w_ij, dtype=bm.float64):
        """传统密度更新方法"""
        mass = state["mass"]
        a = bm.zeros_like(mass, dtype=dtype)
        return mass * bm.index_add(a, self_node, w_ij, axis=0, alpha=1)
        
    def momentum_equation_solve(self, improve, state, neighbors, self_node, dr, dist, grad_w, p):
        """动量守恒方程求解"""
        if improve == 0:
            return self.u_tradition(state, neighbors, self_node, dr, dist, grad_w, p)
        elif improve == 1:
            return self.u_improve_1(state, neighbors, self_node, dr, dist, grad_w, p)
        else:
            # 可扩展改进方法
            raise NotImplementedError("Improved momentum method not implemented.")

    def u_improve_1(self, state, neighbors, self_node, dr, dist, grad_w_norm, pb):
        """Transport velocity variation"""
        m_i = state["mass"][neighbors]
        m_j = state["mass"][self_node]
        rho_i = state["rho"][neighbors]
        rho_j = state["rho"][self_node]

        volume_square = ((m_i / rho_i) ** 2 + (m_j / rho_j) ** 2) / m_i
        c = volume_square * grad_w_norm / (dist + EPS)
        a = c[:, None] * pb[neighbors][:, None] * dr      
        return bm.index_add(bm.zeros_like(state["tv"], dtype=bm.float64), neighbors, a, axis=0, alpha=1) 

    def heat_equation_solve(self, improve, state, dr, dist, neighbors, self_node, grad_w):
        """热量守恒方程求解"""
        if improve == 0:
            return self.t_tradition(state, dr, dist, neighbors, self_node, grad_w)
        else:
            # 可扩展改进方法
            raise NotImplementedError("Improved heat equation method not implemented yet.")

    def state_equation(self, method, state, **kwargs):
        """选择状态方程"""
        if method == "tait_eos":
            return self.tait_eos(state, **kwargs)
        elif method == "injection_molding":
            return self.fuild_p(state, **kwargs)
        else:
            # 在此添加其他本构方程
            raise NotImplementedError(f"Constitutive method '{method}' is not implemented.")

    def tait_eos(self, state, rho=None, c0=None, rho0=None, gamma=None, X=None):
        """状态方程计算压力"""
        rho = state['rho'] if rho is None else rho
        c0 = self.c0 if c0 is None else c0
        rho0 = self.rho0 if rho0 is None else rho0
        gamma = self.gamma if gamma is None else gamma
        X = self.X if X is None else X
        return gamma * c0**2 * ((rho / rho0)**gamma - 1) / rho0 + X

    def fuild_p(self, state, rho0=None, B=None, c1=None):
        """Updates the pressure of fluid particles"""
        rho0 = self.rho0 if rho0 is None else rho0
        B = self.B if B is None else B
        c1 = self.c1 if c1 is None else c1
        f_rho = state["rho"][(state["tag"] == 0)|(state["tag"] == 3)]
        fg_p = B * (bm.exp((1 - rho0 / f_rho) / c1) - 1)
        fg_indx = bm.where((state["tag"] == 0)|(state["tag"] ==3))[0]
        state["p"] = bm.set_at(state["p"], fg_indx, fg_p)
        return state["p"]

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

    def t_tradition(self, state, dr, dist, neighbors, self_node, grad_w):
        """传统温度变化计算"""
        k = (state["kappa"][neighbors] * state["kappa"][self_node]) / (state["kappa"][neighbors] + state["kappa"][self_node])
        a = bm.sum(dr * grad_w, axis=1)
        F_ab = a / ((dist * dist) + EPS)
        b = 4 * state["mass"][self_node] * k * (state["T"][neighbors] - state["T"][self_node])
        dTdt = (b * F_ab) / (state["Cp"][neighbors] * state["rho"][neighbors] * state["rho"][self_node])
        result = bm.index_add(bm.zeros_like(state["T"]), neighbors, dTdt, axis=0, alpha=1)
        return result 

    def tait_eos_p2rho(self, p, p0, rho0, gamma=1.0, X=0.0):
        """通过密度计算压力"""
        p_temp = p + p0 - X
        return rho0 * (p_temp / p0) ** (1 / gamma)

    def sound(self, state, rho0=None, B=None, c1=None):
        """Update sound speed"""
        rho0 = self.rho0 if rho0 is None else rho0
        B = self.B if B is None else B
        c1 = self.c1 if c1 is None else c1
        fg_tag = bm.where((state["tag"] == 0) | (state["tag"] == 3))[0]
        rho = state["rho"][fg_tag]
        value = bm.sqrt(B * bm.exp((1 - rho0 / rho) / c1) * (rho0 / (c1 * rho**2)))
        state["sound"] = bm.set_at(state["sound"], fg_tag, value)
        return state["sound"]
