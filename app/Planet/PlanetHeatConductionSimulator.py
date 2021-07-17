
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
        self.space = WedgeLagrangeFiniteElementSpace(mesh, p=args.degree,
                q=args.integral)
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
        dt = self.timeline.current_time_step_length()

        S = self.S
        M = self.M 
        uh0 = self.uh0
        b = M@uh0[:]/dt

        index = self.mesh.ds.exterior_boundary_tface_index()
        # 处理 Robin 边界条件
        R, b = self.space.set_tri_boundary_robin_bc(S, b, lambda x,
                n:self.pde.robin(x, n, t1),
                threshold=index, uh=self.uh, m=3)
        return R, b

    def picard_iteration(self, ctx=None):
        '''  piccard 迭代  向后欧拉方法 '''

        timeline = self.timeline
        dt = timeline.current_time_step_length()

        S = self.S
        M = self.M

        uh0 = self.uh0
        uh1 = self.uh1
        uh = self.uh
        
        e = self.args.accuracy
        error = 1
        while error > e:
            uh[:] = uh1
            R, b = self.get_current_linear_system()
            R = M + dt*R
            uh1[:] = spsolve(R, dt*b).reshape(-1)
            error = np.max(np.abs(uh[:] - uh1[:]))
            print('error:', error)

    def run(self):
        """

        Notes
        -----

        计算所有时间层
        """
        timeline = self.timeline
        
        while not timeline.stop():
            i = timeline.current
            print('i:', i+1)
            self.picard_iteration()
            self.uh0[:] = self.uh1
            print(self.uh0)
            timeline.current +=1
