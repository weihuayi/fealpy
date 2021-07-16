
import numpy as np

from fealpy.decorator import cartesian, barycentric
from fealpy.functionspace import WedgeLagrangeFiniteElementSpace
from fealpy.timeintegratoralg.timeline import UniformTimeLine

from fealpy.boundarycondition import RobinBC, NeumannBC

from scipy.sparse.linalg import spsolve

class PlanetHeatConductionSimulator():
    def __init__(self, pde, mesh, args):

        self.args = args
        self.pde = pde
        self.mesh = mesh
        self.space = WedgeLagrangeFiniteElementSpace(mesh, p=args.degeree,
                q=3)
        self.S = self.space.stiff_matrix() # 刚度矩阵
        self.M = self.space.mass_matrix() # 质量矩阵

        omega = self.pde.options['omega']
        self.args.T *= 3600*24 # 换算成秒
        self.args.T *= omega # 归一化
        self.args.DT *= omega 
        NT = int(args.T/args.DT)
        self.timeline = UniformTimeLine(0, args.T, NT)

        self.uh0 = self.space.function() # 当前层的数值解
        self.uh0[:] = self.pde.options['theta']/self.pde.options['Tss']
        self.uh1 = self.space.function() # 下一层的数值解

        self.uh = self.space.function() # 临时数值解
        self.uh[:] = self.uh0

    def get_current_linear_system(self):

        t1 = self.timeline.next_time_level()

        S = self.S
        M = self.M 
        uh0 = self.uh0
        b = M@uh0[:]

        index = self.mesh.ds.exterior_boundary_tface_index()
        # 处理 Robin 边界条件
        R, b = self.space.set_tri_boundary_robin_bc(S, b, lambda x,
                n:self.pde.robin(x, n, t1),
                threshold=index, uh=self.uh, m=3)

        return R, b

    def picard_iteration(self, ctx=None):
        '''  piccard 迭代  向后欧拉方法 '''

        timeline = self.timeline
        i = timeline.current
        t1 = timeline.next_time_level()
        dt = timeline.current_time_step_length()

        S = self.S
        M = self.M
        F = self.apply_boundary_condition(S, F, timeline)
        
        error = 1
        xi_new = self.space.function()
        xi_new[:] = uh[:]
        while error > e:
            xi_tmp = xi_new.copy()
            b = M@uh[:] + dt*F
            R = M + dt*R

            xi_new[:] = spsolve(R, b).reshape(-1)
            error = np.max(np.abs(xi_tmp - xi_new[:]))
            print('error:', error)
        print('i:', i+1)
        uh[:] = xi_new

    def run():
        pass
