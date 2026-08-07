from typing import Any, Optional, Union, List, Tuple

from fealpy.backend import bm
from fealpy.model import ComputationalModel
from fealpy.material import LinearElasticMaterial

from fealpy.csm.model.linear_elasticity import LinearElasticityPDEDataT
from fealpy.csm.model.model_manager import CSMModelManager

from fealpy.backend import TensorLike
from fealpy.decorator import cartesian
from fealpy.typing import Index, _S
from fealpy.cfd.simulation.fem.fem_base import FEM


class HydraulicPipeFlowModel2D(ComputationalModel):
    def __init__(self, options, mesh):
        self.options = options
        self.mesh = mesh
        self.rho = options.get('rho', 1.0)
        self.mu = options.get('mu', 0.003)

    @cartesian
    def distance_to_wallline(self, p: TensorLike) -> TensorLike:
        R_pipe = 0.5  # 管道半径
        R_bend = 2.8  # 弯管曲率半径

        x = p[..., 0]
        y = p[..., 1]

        # 1. 上游直管 (x <= 0)
        # 轴线在 (y=0, z=0)，点到轴线距离为 sqrt(y^2 + z^2)
        dist_to_axis_up = bm.abs(y)
        d_up = R_pipe - dist_to_axis_up

        # 2. 弯管段 (x > 0 且 y < R_bend)
        # 轴线是以 (0, R_bend) 为圆心，R_bend 为半径的圆弧
        # 在 xy 平面上，点到圆心的距离：
        dist_to_center_xy = bm.sqrt(x**2 + (y - R_bend)**2)
        # 点到圆弧轴线的距离（考虑 z 轴）：
        dist_to_axis_bend = bm.abs(dist_to_center_xy - R_bend)
        d_bend = R_pipe - dist_to_axis_bend

        # 3. 下游直管 (y >= R_bend)
        # 假设下游沿 y 轴延伸，轴线在 (x=R_bend, z=0)
        # 注意：需根据你 ElbowPipeMesher 的实际生成坐标调整
        dist_to_axis_down = bm.abs(x - R_bend)
        d_down = R_pipe - dist_to_axis_down

        # 4. 平滑组合 (使用逻辑判断)
        # 修正：d 必须限制最小值为 0，防止数值越界进入壁面内部
        d = bm.where(x <= 0, d_up, 
                        bm.where(y >= R_bend, d_down, d_bend))

        # 限制范围，确保距离在 [0, R_pipe] 之间，防止 SST 模型崩溃
        return bm.maximum(d, 1e-15)
    
    @cartesian
    def is_bd_dof(self, boundary_edge, space):

        edge = self.mesh.entity('edge')
        edge2dof = space.edge_to_dof()
        edge_np = bm.to_numpy(edge)
        be_np = bm.to_numpy(boundary_edge)

        edge_map = {tuple(e): i for i, e in enumerate(edge_np)}

        bd_edge_index = [edge_map[tuple(e)] for e in be_np]
        bd_edge_index = bm.array(bd_edge_index, dtype=bm.int64)

        bd_edge2dof = edge2dof[bd_edge_index]

        # 所有边界 dof
        bd_dof = bm.unique(bd_edge2dof.reshape(-1))

        gdof = int(bm.max(edge2dof)) + 1

        is_bd_dof = bm.zeros(gdof, dtype=bm.bool)
        is_bd_dof[bd_dof] = True

        return is_bd_dof
    
    @cartesian
    def is_inlet_boundary(self, space) -> TensorLike:
        inlet = self.mesh.inlet
        return self.is_bd_dof(inlet, space)
    
    @cartesian
    def is_outlet_boundary(self, space) -> TensorLike:
        outlet = self.mesh.outlet
        return self.is_bd_dof(outlet, space)
    
    @cartesian
    def is_wall_boundary(self, space) -> TensorLike:
        fsi = self.mesh.fsi
        return self.is_bd_dof(fsi, space)
    
    # 动量方程
    @cartesian
    def inlet_velocity(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        R = 0.5
        d = self.distance_to_wallline(p)
        u = bm.zeros(p.shape)
        u[..., 0] = 8*(0.5+y)*(0.5-y)
        u[..., 1] = 0.0
        # u = u.reshape(-1, order='F')
        return u
    
    @cartesian
    def outlet_velocity(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        R = 0.5
        d = self.distance_to_wallline(p)
        u = bm.zeros(p.shape)
        u[..., 0] = 0.0
        u[..., 1] = 1.224*(1.0 - (0.5-d)/R)**(1/7)
        # u = u.reshape(-1, order='F')
        return u
    
    @cartesian
    def outlet_pressure(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        pressure = bm.zeros(x.shape)
        return pressure
    
    @cartesian
    def wall_velocity(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        # u = bm.zeros(p.shape).reshape(-1, order='F')
        u = bm.zeros(p.shape)
        return u
    
    @cartesian
    def is_velocity_boundary(self, space = None) -> TensorLike:
        return self.is_inlet_boundary(space) | self.is_wall_boundary(space)
        # return None
    
    @cartesian
    def is_pressure_boundary(self,space = None) -> TensorLike:
        if space is None:
            return 1
        space = self.pspace
        is_p_bd_dof = bm.zeros(self.is_outlet_boundary(space).shape, dtype=bm.bool)
        return self.is_outlet_boundary(space)
        # return is_p_bd_dof
    
    @cartesian
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
        space = self.uspace
        gdof = space.number_of_global_dofs()
        result = bm.zeros(p.shape)
        inlet = self.inlet_velocity(p)
        outlet = self.outlet_velocity(p)
        wall = self.wall_velocity(p)
        is_inlet = self.is_inlet_boundary(space)[:gdof//2]
        # is_inlet = self.is_inlet_boundary(space)
        is_wall = self.is_wall_boundary(space)[:gdof//2]
        is_outlet = self.is_outlet_boundary(space)[:gdof//2]

        result[is_inlet] = inlet[is_inlet]
        # result[is_outlet] = outlet[is_outlet]
        result[is_wall] = wall[is_wall]
        return result
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        return self.outlet_pressure(p)
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape)
        return result
    
    @cartesian
    def pressure_integral_target(self):
        return 0.0
    
    