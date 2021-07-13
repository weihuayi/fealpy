
import numpy as np

from fealpy.decorator import cartesian, barycentric
from fealpy.functionspace import WedgeLagrangeFiniteElementSpace
from fealpy.timeintegratoralg.timeline import UniformTimeLine

from fealpy.boundarycondition import RobinBC, NeumannBC

from scipy.sparse.linalg import spsolve

class PlanetHeatConductionSimulator():
    def __init__(self, pde, mesh, p=1):

        self.pde = pde
        self.mesh = mesh
        self.space = WedgeLagrangeFiniteElementSpace(mesh, p=p, q=6)

        self.S = self.space.stiff_matrix() # 刚度矩阵
        self.M = self.space.mass_matrix() # 质量矩阵

    def time_mesh(self, T=10, NT=100):
        
        """ 
        Parameters
        ----------
            T: the final time (day)
        Notes
        ------
            无量纲化后 tau=omega*t 这里的时间层为 tau
        """

        omega = self.pde.options['omega']
        T *= 3600*24 # 换算成秒
        T *= omega # 归一化
        timeline = UniformTimeLine(0, T, NT)
        return timeline

    def init_solution(self):
        uh = self.space.function()
        uh[:] = self.pde.options['theta']/self.pde.options['Tss']
        return uh

    def apply_boundary_condition(self, A, b, timeline):
        t1 = timeline.next_time_level()
       
       # 对 neumann 边界条件进行处理
        bc = NeumannBC(self.space, lambda x, n:self.pde.neumann(x, n), threshold=self.pde.is_neumann_boundary())
        b = bc.apply(b) # 混合边界条件不需要输入矩阵A

        # 对 robin 边界条件进行处理
        bc = RobinBC(self.space, lambda x, n:self.pde.robin(x, n, t1), threshold=self.pde.is_robin_boundary())
        A0, b = bc.apply(A, b)
        return b

    def solve(self, uh, timeline, e=1e-10):
        '''  piccard 迭代  向后欧拉方法 '''

        i = timeline.current
        t1 = timeline.next_time_level()
        dt = timeline.current_time_step_length()
        F = self.pde.right_vector(uh)

        S = self.S
        M = self.M
        F = self.apply_boundary_condition(S, F, timeline)
        
        error = 1
        xi_new = self.space.function()
        xi_new[:] = uh[:]
        while error > e:
            xi_tmp = xi_new.copy()
            R = self.space.set_tri_boundary_robin_bc(S, F, lambda x,
                    n:self.pde.robin(x, n, t1),
                    threshold=self.pde.is_robin_boundary(), uh=xi_new, m=3)
            b = M@uh[:] + dt*F
            R = M + dt*R

            xi_new[:] = spsolve(R, b).reshape(-1)
            error = np.max(np.abs(xi_tmp - xi_new[:]))
            print('error:', error)
        print('i:', i+1)
        uh[:] = xi_new

