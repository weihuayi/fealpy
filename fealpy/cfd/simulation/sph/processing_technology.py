from fealpy.backend import backend_manager as bm
from fealpy.cfd.simulation.sph.equation_solver import EquationSolver

EPS = bm.finfo(float).eps
class ProcessingTechnology:
    def __init__(self, mesh):
        self.mesh = mesh 

    @staticmethod
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
    def enforce_wall_boundary(state, p, g_ext, i_s, j_s, w_dist, dr_i_j, c0=10.0, rho0=1.0, X=5.0, \
        p0=100.0, with_temperature=False, dtype=bm.float64):
        """Enforce wall boundary conditions by treating boundary particles in a special way"""
        solver = EquationSolver()
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
        
        rho = solver.tait_eos_p2rho(p, p0, rho0, X=5.0)
        
        def compute_temperature():
            t_wall_unnorm = bm.index_add(bm.zeros(len(state["position"]), dtype=dtype), i_s, w_j_s_fluid * state["T"][j_s], axis=0, alpha=1)
            t_wall = t_wall_unnorm / (w_i_sum_wf + EPS)
            mask = bm.isin(state["tag"], bm.array([1, 2]))
            t_wall = bm.where(mask, t_wall, state["T"])
            return t_wall

        T = bm.where(bm.array(with_temperature), compute_temperature(), state["T"])
        return p, rho, mv, tv, T 

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
    def A_matrix(state, node_self, neighbors, dr, dw):
        mass_j = state["mass"][neighbors]
        rho_j = state["rho"][neighbors]
        value = (mass_j / rho_j)[:, None, None] * bm.einsum('ij, ik->ijk', -dr, dw) #外积
        A_s = bm.zeros((len(state["mu"]), 2, 2), dtype=bm.float64)
        A_s = bm.index_add(A_s, node_self, value, axis=0, alpha=1)
        return A_s