from fealpy.backend import TensorLike
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.typing import Index, _S

class PipeBendTurbulentFlow():
    def __init__(self):
        self.rho = 1.0
        self.mu = 0.003
        self.beta_s = 0.09
        self.beta = 0.079
        self.a1 = 0.31
        self.sigma_k = 1.0
        self.sigma_omega = 0.5
        self.sigma_omega2 = 0.856
        self.gamma = 0.5

    @cartesian
    def distance_t0_wallline(self, p: TensorLike) -> TensorLike:
        R_pipe = 0.5  # 管道半径
        R_bend = 2.8  # 弯管曲率半径

        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]

        # 1. 上游直管 (x <= 0)
        # 轴线在 (y=0, z=0)，点到轴线距离为 sqrt(y^2 + z^2)
        dist_to_axis_up = bm.sqrt(y**2 + z**2)
        d_up = R_pipe - dist_to_axis_up

        # 2. 弯管段 (x > 0 且 y < R_bend)
        # 轴线是以 (0, R_bend) 为圆心，R_bend 为半径的圆弧
        # 在 xy 平面上，点到圆心的距离：
        dist_to_center_xy = bm.sqrt(x**2 + (y - R_bend)**2)
        # 点到圆弧轴线的距离（考虑 z 轴）：
        dist_to_axis_bend = bm.sqrt((dist_to_center_xy - R_bend)**2 + z**2)
        d_bend = R_pipe - dist_to_axis_bend

        # 3. 下游直管 (y >= R_bend)
        # 假设下游沿 y 轴延伸，轴线在 (x=R_bend, z=0)
        # 注意：需根据你 ElbowPipeMesher 的实际生成坐标调整
        dist_to_axis_down = bm.sqrt((x - R_bend)**2 + z**2)
        d_down = R_pipe - dist_to_axis_down

        # 4. 平滑组合 (使用逻辑判断)
        # 修正：d 必须限制最小值为 0，防止数值越界进入壁面内部
        d = bm.where(x <= 0, d_up, 
                        bm.where(y >= R_bend, d_down, d_bend))

        # 限制范围，确保距离在 [0, R_pipe] 之间，防止 SST 模型崩溃
        return bm.maximum(d, 1e-15)
    
    @cartesian
    def strain_rate(self, u0, bcs, index):
            grad_u = u0.grad_value(bcs, index)
            grad_u_T = bm.swapaxes(grad_u, -1, -2)

            S_ij = 1/2 * (grad_u + grad_u_T)
            return S_ij
    
    @cartesian
    def tur_mu(self, u0, k0, omega0, points, bcs, index: Index = _S):
        beta_s = self.beta_s
        mu = self.mu
        rho = self.rho
        a1 = self.a1
        d = self.distance_t0_wallline(points)

        def shear_stress_limit_function():
            k0_value = bm.maximum(k0(bcs, index), 1e-10)
            arg2 = bm.maximum(2 * bm.sqrt(k0_value)/(beta_s * omega0(bcs, index) * d),
                            500 * mu/(d**2 * rho * omega0(bcs, index)))
            F2 = bm.tanh(arg2**2)
            return F2
        F2 = shear_stress_limit_function()

        S_ij = self.strain_rate(u0, bcs, index)
        S = bm.sqrt(2 * bm.sum(S_ij * S_ij, axis=(2, 3)))
        mu_t = a1 * k0(bcs, index)
        mu_t /= bm.maximum(a1 * omega0(bcs, index), S * F2)
        return mu_t
    
    @cartesian
    def is_inlet_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        atol = 1e-12
        on_boundary = (bm.abs(x + 10) < atol)
        return on_boundary
    
    @cartesian
    def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        atol = 1e-12
        on_boundary = (bm.abs(y - 17.8) < atol)
        return on_boundary
    
    @cartesian
    def is_wall_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]

        r = 0.5
        d = self.distance_t0_wallline(p)
        atol = 1e-12
        on_boundary = (bm.abs(d) < atol)
        return on_boundary
    
    # 动量方程
    @cartesian
    def inlet_velocity(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        R = 0.5
        d = self.distance_t0_wallline(p)
        u = bm.zeros(p.shape)
        u[..., 0] = 1.224*(1.0 - (0.5 - d)/R)**(1/7)
        u[..., 1] = 0.0
        u[..., 2] = 0.0
        return u
    
    @cartesian
    def outlet_velocity(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        R = 0.5
        d = self.distance_t0_wallline(p)
        u = bm.zeros(p.shape)
        u[..., 0] = 0.0
        u[..., 1] = 1.224*(1.0 - (0.5-d)/R)**(1/7)
        u[..., 2] = 0.0
        return u
    
    @cartesian
    def outlet_pressure(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pressure = bm.zeros(x.shape)
        return pressure
    
    @cartesian
    def wall_velocity(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        u = bm.zeros(p.shape)
        return u
    
    @cartesian
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
        return self.is_inlet_boundary(p) | self.is_wall_boundary(p)
        # return None
    
    @cartesian
    def is_pressure_boundary(self, p: TensorLike = None) -> TensorLike:
        # if p is None:
        #     return 1
        # return self.is_outlet_boundary(p)
        return 0
    
    @cartesian
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
        result = bm.zeros(p.shape)
        inlet = self.inlet_velocity(p)
        outlet = self.outlet_velocity(p)
        wall = self.wall_velocity(p)
        is_inlet = self.is_inlet_boundary(p)
        is_wall = self.is_wall_boundary(p)
        is_outlet = self.is_outlet_boundary(p)

        result[is_inlet] = inlet[is_inlet]
        result[is_wall] = wall[is_wall]
        result[is_outlet] = outlet[is_outlet]
        return result
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        return self.outlet_pressure(p)
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        result = bm.zeros(p.shape)
        return result
    
    # k 方程
    @cartesian
    def k_dirichlet(self, p: TensorLike) -> TensorLike:
        is_inlet = self.is_inlet_boundary(p)
        is_outlet = self.is_outlet_boundary(p)
        is_wall = self.is_wall_boundary(p)
        k = bm.zeros(p[..., 0].shape)
        k[is_inlet] = 0.00375
        k[is_outlet] = 0.00375
        k[is_wall] = 0
        return k
    
    @cartesian
    def is_k_boundary(self, p: TensorLike) -> TensorLike:
        is_inlet = self.is_inlet_boundary(p)
        is_outlet = self.is_outlet_boundary(p)
        is_wall = self.is_wall_boundary(p)
        return is_wall | is_inlet 
    
    @cartesian
    def production_k(self, u0, k0, omega0, mu_t, bcs, index) -> TensorLike:
        result_0 = self.production_omega(u0, k0, mu_t, bcs, index)
        result_1 = 10 * self.beta_s * self.rho * k0(bcs, index) * omega0(bcs, index)
        result = bm.minimum(result_0, result_1)
        return result
    
    # omega 方程
    @cartesian
    def omega_dirichlet(self, p: TensorLike) -> TensorLike:
        d = self.distance_t0_wallline(p)
        nu = self.mu/self.rho
        is_inlet = self.is_inlet_boundary(p)
        is_outlet = self.is_outlet_boundary(p)
        is_wall = self.is_wall_boundary(p)
        omega = bm.zeros(p[..., 0].shape)
        omega[is_inlet] = 1.597
        omega[is_wall] = bm.minimum((60 * nu / (self.beta * d**2)), 2e6)[is_wall]
        return omega
    
    @cartesian
    def is_omega_boundary(self, p: TensorLike) -> TensorLike:
        is_inlet = self.is_inlet_boundary(p)
        is_outlet = self.is_outlet_boundary(p)
        is_wall = self.is_wall_boundary(p)
        return is_inlet | is_wall
    
    @cartesian
    def production_omega(self, u0, k0, mu_t, bcs, index) -> TensorLike:
        S_ij = self.strain_rate(u0, bcs, index)
        grad_u = u0.grad_value(bcs, index)
        
        P = mu_t * bm.sum(S_ij * grad_u, axis=(2, 3))
        P -= 2/3 * k0(bcs, index) * bm.einsum("...ii -> ...", grad_u)
        return P
    
    @cartesian
    def cross_diffuison_f1(self, k1, omega0, points, bcs, index):
        d = self.distance_t0_wallline(p=points)
        rho = self.rho
        k1_value = k1(bcs, index)
        k1_value = bm.maximum(k1_value, 1e-10)
        arg1_11 = bm.sqrt(k1_value)
        arg1_11 /= self.beta_s * omega0(bcs, index) * d
        arg1_12 = 500 * self.mu / (d**2 * rho * omega0(bcs, index))
        arg1_1 = bm.maximum(arg1_11, arg1_12)

        def cross_diddusion():
            CD1 = 2 * rho * self.sigma_omega2
            reciprocal_omega0 = 1/omega0
            CD1 *= reciprocal_omega0(bcs, index)
            grad_k1 = k1.grad_value(bcs, index)
            grad_omega0 = omega0.grad_value(bcs, index)
            CD1 *= bm.sum(grad_k1 * grad_omega0, axis=(2))

            CD2 = 10e-10

            CD = bm.maximum(CD1, CD2)
            return CD
        CD = cross_diddusion()
        arg1_2 = 4 * rho * self.sigma_omega2 * k1(bcs, index)
        arg1_2 /= CD * d**2

        arg1 = bm.minimum(arg1_1, arg1_2)

        F1 = bm.tanh(arg1**4)
        return F1
    
    def pressure_integral_target(self):
        return 0.0

    

