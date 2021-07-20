
import numpy as np

from fealpy.decorator import cartesian, barycentric
from fealpy.functionspace import ParametricLagrangeFiniteElementSpaceOnWedgeMesh
from fealpy.timeintegratoralg.timeline import UniformTimeLine

from scipy.sparse.linalg import spsolve

class PlanetHeatConductionSimulator():
    def __init__(self, pde, mesh, args):

        self.args = args
        self.pde = pde
        self.mesh = mesh
        self.space = ParametricLagrangeFiniteElementSpaceOnWedgeMesh(mesh,
                p=args.degree, q=args.nq)
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

    def nolinear_robin_boundary(self, bcs):
        uh = self.uh
        Phi = self.pde.options['Phi']
        t1 = self.timeline.next_time_level()

        qf = mesh.integrator(self.args.nq, 'tface')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = self.space.basis(bcs)

        index = self.mesh.ds.exterior_boundary_tface_index()
        face2dof = self.space.tri_face_to_dof()[index]

        val = np.einsum('qfi, fi->qf', phi, uh[face2dof])
        val **= 3

        return R, b

    def get_current_linear_system(self):

        t1 = self.timeline.next_time_level()

        S = self.S
        b = np.zeros(len(self.uh0), dtype=np.float64)

        index = self.mesh.ds.exterior_boundary_tface_index()
        # 处理 Robin 边界条件
        R, b = self.space.set_tri_boundary_robin_bc(S, b, lambda x,
                n:self.pde.robin(x, n, t1),
                threshold=index, uh=self.uh, m=3)
        return R, b

    def picard_iteration(self, ctx=None):
        '''  picard 迭代  向后欧拉方法 '''

        timeline = self.timeline
        dt = timeline.current_time_step_length()

        M = self.M

        uh0 = self.uh0
        uh1 = self.uh1
        uh = self.uh
        
        e = self.args.accuracy
        m = self.args.npicard
        k = 0
        error = 1.0
        while error > e:
            uh[:] = uh1
            
            R, b = self.get_current_linear_system()
            b *= dt
            b += M@uh0[:]
            R *= dt
            R += M
            
            uh1[:] = spsolve(R, b).reshape(-1)
            error = np.max(np.abs(uh[:] - uh1[:]))
            
            k += 1
            if k >= self.args.npicard: 
                print('picard iteration arrive max iteration with error:', error)
                break
    
    def update_mesh_data(self):
        """

        Notes
        -----
        更新 mesh 中的数据
        """
        mesh = self.mesh

        uh0 = self.uh0

        # 温度
        Tss = self.pde.options['Tss']
        T = uh0*Tss
        mesh.nodedata['uh'] = uh


    def run(self, ctx=None, queue=None, writer=None):
        """

        Notes
        -----

        计算所有时间层
        """
        timeline = self.timeline
        args = self.args
        
        if queue is not None:
            i = timeline.current
            fname = args.output + str(i).zfill(10) + '.vtu'
            self.update_mesh_data()
            data = {'name':fname, 'mesh':self.mesh}
            queue.put(data)

        if writer is not None:
            i = timeline.current
            fname = args.output + str(i).zfill(10) + '.vtu'
            self.update_mesh_data()
            writer(fname, self.mesh)

        while not timeline.stop():
            i = timeline.current
            print('i:', i+1)
            self.picard_iteration(ctx=ctx)
            self.uh0[:] = self.uh1
            
            timeline.advance()
            
            if timeline.current % args.step == 0:
                if queue is not None:
                    i = timeline.current
                    fname = args.output + str(i).zfill(10) + '.vtu'
                    self.update_mesh_data()
                    data = {'name':fname, 'mesh':self.mesh}
                    queue.put(data)

                if writer is not None:
                    i = timeline.current
                    fname = args.output + str(i).zfill(10) + '.vtu'
                    self.update_mesh_data()
                    writer(fname, self.mesh)
        
        if queue is not None:
            i = timeline.current
            fname = args.output + str(i).zfill(10) + '.vtu'
            self.update_mesh_data()
            data = {'name':fname, 'mesh':self.mesh}
            queue.put(data)
            queue.put(-1) # 发送模拟结束信号 

        if writer is not None:
            i = timeline.current
            fname = args.output + str(i).zfill(10) + '.vtu'
            self.update_mesh_data()
            writer(fname, self.mesh)

