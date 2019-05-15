import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import eye, csr_matrix, bmat
import scipy.io as sio
import pyamg
from timeit import default_timer as timer
from mpl_toolkits.mplot3d import Axes3D

from ..functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from ..solver.eigns import picard
from ..quadrature import FEMeshIntegralAlg
from ..mesh.adaptive_tools import mark


class EllipticEignvalueFEMModel:
    def __init__(self, pde, theta=0.2, maxit=30, step=0, n=3, p=1, q=3,
            resultdir='~/'):
        self.pde = pde
        self.step = step
        self.theta = theta
        self.maxit = maxit
        self.p = p
        self.q = q
        self.numrefine = n
        self.resultdir = resultdir

    def residual_estimate(self, uh):
        mesh = uh.space.mesh
        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()

        n = mesh.face_normal()
        bc = np.array([1/(GD+1)]*(GD+1), dtype=mesh.ftype)
        grad = uh.grad_value(bc)

        ps = mesh.bc_to_point(bc)
        try:
            d = self.pde.diffusion_coefficient(ps)
        except AttributeError:
            d = np.ones(NC, dtype=mesh.ftype)

        if isinstance(d, float):
            grad *= d
        elif len(d) == GD:
            grad = np.einsum('m, im->im', d, grad)
        elif isinstance(d, np.ndarray):
            if len(d.shape) == 1:
                grad = np.einsum('i, im->im', d, grad)
            elif len(d.shape) == 2:
                grad = np.einsum('im, im->im', d, grad)
            elif len(d.shape) == 3:
                grad = np.einsum('imn, in->in', d, grad)

        if GD == 2:
            face2cell = mesh.ds.edge_to_cell()
            h = np.sqrt(np.sum(n**2, axis=-1))
        elif GD == 3:
            face2cell = mesh.ds.face_to_cell()
            h = np.sum(n**2, axis=-1)**(1/4)

        J = h*np.sum((grad[face2cell[:, 0]] - grad[face2cell[:, 1]])*n, axis=-1)**2

        NC = mesh.number_of_cells()
        eta = np.zeros(NC, dtype=mesh.ftype)
        np.add.at(eta, face2cell[:, 0], J)
        np.add.at(eta, face2cell[:, 1], J)
        return np.sqrt(eta)

    def get_stiff_matrix(self, space, integrator, area):
        A = space.stiff_matrix(integrator, area)
        return A

    def get_mass_matrix(self, space, integrator, area):
        M = space.mass_matrix(integrator, area)
        return M

    def u(self, p):
        return np.sum(p, axis=-1)

    def alg_0(self):
        """
        1. 最粗网格上求解最小特征特征值问题，得到最小特征值 d_H 和特征向量 u_H
        2. 自适应求解  - \Delta u_h = d_H*u_H
            *  每层网格上求出的 u_h，插值到下一层网格上做为 u_H
            *  并更新 d_H = u_h@A@u_h/u_h@M@u_h， 其中 A 是当前网格层上的刚度矩
               阵，M 为当前网格层的质量矩阵。
        3. 最细网格层上求出的 uh 做为一个基函数，加入到最粗网格的有限元空间中，
           在最粗网格上求解最小特征值问题。
        """
        print("算法 0")
        start = timer()
        if self.step == 0:
            idx = []
        else:
            idx =list(range(0, self.maxit, self.step)) + [self.maxit-1]

        mesh = self.pde.init_mesh(n=self.numrefine)
        integrator = mesh.integrator(self.q)

        # 1. 粗网格上求解最小特征值问题
        space = LagrangeFiniteElementSpace(mesh, 1)
        gdof = space.number_of_global_dofs()
        print(0, ":", gdof)
        uh = np.zeros(gdof, dtype=np.float)
        area = mesh.entity_measure('cell')
        AH = self.get_stiff_matrix(space, integrator, area)
        MH = self.get_mass_matrix(space, integrator, area)
        isFreeHDof = ~(space.boundary_dof())
        A = AH[isFreeHDof, :][:, isFreeHDof].tocsr()
        M = MH[isFreeHDof, :][:, isFreeHDof].tocsr()
        uh[isFreeHDof], d = picard(A, M, np.ones(sum(isFreeHDof)))

        GD = mesh.geo_dimension()
        if (self.step > 0) and (0 in idx):
            NN = mesh.number_of_nodes()
            fig = plt.figure()
            fig.set_facecolor('white')
            if GD == 2:
                axes = fig.gca()
            else:
                axes = Axes3D(fig)
            mesh.add_plot(axes, cellcolor='w')
            fig.savefig(self.resultdir + 'mesh_0_0_' + str(NN) +'.pdf')
            plt.close()

        # 2. 以 u_h 为右端项自适应求解 -\Deta u = d*u_h
        I = eye(gdof)
        u0 = self.u(mesh.entity('node'))
        for i in range(self.maxit):
            uh = space.function(array=uh)
            eta = self.residual_estimate(uh)
            markedCell = mark(eta, self.theta)
            IM = mesh.bisect(markedCell, returnim=True)
            ui = self.u(mesh.entity('node'))
            u0 = IM@u0
            print("intetpolation error: ", np.max(np.abs(ui - u0)))

            if (self.step > 0) and (i in idx):
                NN = mesh.number_of_nodes()
                fig = plt.figure()
                fig.set_facecolor('white')
                if GD == 2:
                    axes = fig.gca()
                else:
                    axes = Axes3D(fig)
                mesh.add_plot(axes, cellcolor='w')
                fig.savefig(self.resultdir + 'mesh_0_' + str(i+1) + '_' + str(NN) +'.pdf')
                plt.close()

            I = IM@I
            uh = IM@uh
            space = LagrangeFiniteElementSpace(mesh, 1)
            gdof = space.number_of_global_dofs()
            print(i+1, ":", gdof)

            area = mesh.entity_measure('cell')
            A = self.get_stiff_matrix(space, integrator, area)
            M = self.get_mass_matrix(space, integrator, area)
            isFreeDof = ~(space.boundary_dof())
            b = d*M@uh
            ml = pyamg.ruge_stuben_solver(A[isFreeDof, :][:, isFreeDof].tocsr())
            uh[isFreeDof] = ml.solve(b[isFreeDof], x0=uh[isFreeDof], tol=1e-12, accel='cg').reshape((-1,))
            d = uh@A@uh/(uh@M@uh)

        # 3. 把 uh 加入粗网格空间, 组装刚度和质量矩阵

        w0 = uh@A
        w1 = w0@uh
        w2 = w0@I
        AA = bmat([[AH, w2.reshape(-1, 1)], [w2, w1]], format='csr')

        w0 = uh@M
        w1 = w0@uh
        w2 = w0@I
        MM = bmat([[MH, w2.reshape(-1, 1)], [w2, w1]], format='csr')

        isFreeDof = np.r_[isFreeHDof, True]

        u = np.zeros(len(isFreeDof))

        ## 求解特征值
        A = AA[isFreeDof, :][:, isFreeDof].tocsr()
        M = MM[isFreeDof, :][:, isFreeDof].tocsr()
        u[isFreeDof], d = picard(A, M, np.ones(sum(isFreeDof)))

        end = timer()

        print("smallest eigns:", d, "with time: ", end - start)

        uh *= u[-1]
        uh += I@u[:-1]
        uh /= np.max(np.abs(uh))
        uh = space.function(array=uh)
        return uh

    def alg_1(self):
        """
        1. 最粗网格上求解最小特征特征值问题，得到最小特征值 d_H 和特征向量 u_H
        2. 自适应求解  - \Delta u_h = u_H
            *  u_H 插值到下一层网格上做为新 u_H
        3. 最新网格层上求出的 uh 做为一个基函数，加入到最粗网格的有限元空间中，
           求解最小特征值问题。
        """
        print("算法 1")

        start = timer()
        if self.step == 0:
            idx = []
        else:
            idx =list(range(0, self.maxit, self.step)) + [self.maxit-1]


        mesh = self.pde.init_mesh(n=self.numrefine)
        integrator = mesh.integrator(self.q)

        # 1. 粗网格上求解最小特征值问题
        area = mesh.entity_measure('cell')
        space = LagrangeFiniteElementSpace(mesh, 1)
        AH = self.get_stiff_matrix(space, integrator, area)
        MH = self.get_mass_matrix(space, integrator, area)
        isFreeHDof = ~(space.boundary_dof())

        gdof = space.number_of_global_dofs()
        print(0, ":", gdof)
        uH = np.zeros(gdof, dtype=np.float)

        A = AH[isFreeHDof, :][:, isFreeHDof].tocsr()
        M = MH[isFreeHDof, :][:, isFreeHDof].tocsr()
        uH[isFreeHDof], d = picard(A, M, np.ones(sum(isFreeHDof)))

        uh = space.function()
        uh[:] = uH

        GD = mesh.geo_dimension()
        if (self.step > 0) and (0 in idx):
            NN = mesh.number_of_nodes()
            fig = plt.figure()
            fig.set_facecolor('white')
            if GD == 2:
                axes = fig.gca()
            else:
                axes = Axes3D(fig)
            mesh.add_plot(axes, cellcolor='w')
            fig.savefig(self.resultdir + 'mesh_1_0_' + str(NN) + '.pdf')
            plt.close()


        # 2. 以 u_H 为右端项自适应求解 -\Deta u = u_H
        I = eye(gdof)
        for i in range(self.maxit):

            eta = self.residual_estimate(uh)
            markedCell = mark(eta, self.theta)
            IM = mesh.bisect(markedCell, returnim=True)

            if (self.step > 0) and (i in idx):
                NN = mesh.number_of_nodes()
                fig = plt.figure()
                fig.set_facecolor('white')
                if GD == 2:
                    axes = fig.gca()
                else:
                    axes = Axes3D(fig)
                mesh.add_plot(axes, cellcolor='w')
                fig.savefig(self.resultdir + 'mesh_1_' + str(i+1) + '_' + str(NN) +'.pdf')
                plt.close()

            I = IM@I
            uH = IM@uH

            space = LagrangeFiniteElementSpace(mesh, 1)
            gdof = space.number_of_global_dofs()
            print(i+1, ":", gdof)

            area = mesh.entity_measure('cell')
            A = self.get_stiff_matrix(space, integrator, area)
            M = self.get_mass_matrix(space, integrator, area)
            isFreeDof = ~(space.boundary_dof())
            b = M@uH

            ml = pyamg.ruge_stuben_solver(A[isFreeDof, :][:, isFreeDof].tocsr())
            uh = space.function()
            uh[:] = uH
            uh[isFreeDof] = ml.solve(b[isFreeDof], x0=uh[isFreeDof], tol=1e-12, accel='cg').reshape((-1,))

        # 3. 把 uh 加入粗网格空间, 组装刚度和质量矩阵
        w0 = uh@A
        w1 = w0@uh
        w2 = w0@I
        AA = bmat([[AH, w2.reshape(-1, 1)], [w2, w1]], format='csr')

        w0 = uh@M
        w1 = w0@uh
        w2 = w0@I
        MM = bmat([[MH, w2.reshape(-1, 1)], [w2, w1]], format='csr')

        isFreeDof = np.r_[isFreeHDof, True]

        u = np.zeros(len(isFreeDof))
        ## 求解特征值
        A = AA[isFreeDof, :][:, isFreeDof].tocsr()
        M = MM[isFreeDof, :][:, isFreeDof].tocsr()
        u[isFreeDof], d = picard(A, M, np.ones(sum(isFreeDof)))
        end = timer()

        print("smallest eigns:", d, "with time: ", end - start)
        uh *= u[-1]
        uh += I@u[:-1]

        uh /= np.max(np.abs(uh))
        uh = space.function(array=uh)
        return uh

    def alg_2(self):
        """
        1. 最粗网格上求解最小特征特征值问题，得到最小特征值 d_H 和特征向量 u_H
        2. 自适应求解  - \Delta u_h = u_H
            *  u_H 插值到下一层网格上做为新 u_H
        3. 在最细网格上求解一次最小特征值问题
        """
        print("算法 2")
        start = timer()

        if self.step == 0:
            idx = []
        else:
            idx =list(range(0, self.maxit, self.step)) + [self.maxit-1]

        mesh = self.pde.init_mesh(n=self.numrefine)
        integrator = mesh.integrator(self.q)
        # 1. 粗网格上求解最小特征值问题
        area = mesh.entity_measure('cell')
        space = LagrangeFiniteElementSpace(mesh, 1)
        AH = self.get_stiff_matrix(space, integrator, area)
        MH = self.get_mass_matrix(space, integrator, area)
        isFreeHDof = ~(space.boundary_dof())

        gdof = space.number_of_global_dofs()
        uH = np.zeros(gdof, dtype=mesh.ftype)
        print(0, ":", gdof)

        A = AH[isFreeHDof, :][:, isFreeHDof].tocsr()
        M = MH[isFreeHDof, :][:, isFreeHDof].tocsr()
        uH[isFreeHDof], d = picard(A, M, np.ones(sum(isFreeHDof)))

        uh = space.function()
        uh[:] = uH

        GD = mesh.geo_dimension()
        if (self.step > 0) and (0 in idx):
            NN = mesh.number_of_nodes()
            fig = plt.figure()
            fig.set_facecolor('white')
            if GD == 2:
                axes = fig.gca()
            else:
                axes = Axes3D(fig)
            mesh.add_plot(axes, cellcolor='w')
            fig.savefig(self.resultdir + 'mesh_2_0_' + str(NN) + '.pdf')
            plt.close()

        # 2. 以 u_H 为右端项自适应求解 -\Deta u = u_H
        I = eye(gdof)
        for i in range(self.maxit):

            eta = self.residual_estimate(uh)
            markedCell = mark(eta, self.theta)
            IM = mesh.bisect(markedCell, returnim=True)
            if (self.step > 0) and (i in idx):
                NN = mesh.number_of_nodes()
                fig = plt.figure()
                fig.set_facecolor('white')
                if GD == 2:
                    axes = fig.gca()
                else:
                    axes = Axes3D(fig)
                mesh.add_plot(axes, cellcolor='w')
                fig.savefig(self.resultdir + 'mesh_2_' + str(i+1) + '_' + str(NN) +'.pdf')
                plt.close()

            I = IM@I
            uH = IM@uH

            space = LagrangeFiniteElementSpace(mesh, 1)
            gdof = space.number_of_global_dofs()
            print(i+1, ": ", gdof)

            area = mesh.entity_measure('cell')
            A = space.stiff_matrix(integrator, area)
            M = space.mass_matrix(integrator, area)
            isFreeDof = ~(space.boundary_dof())
            b = M@uH

            ml = pyamg.ruge_stuben_solver(A[isFreeDof, :][:, isFreeDof].tocsr())
            uh = space.function()
            uh[:] = uH
            uh[isFreeDof] = ml.solve(b[isFreeDof], x0=uh[isFreeDof], tol=1e-12, accel='cg').reshape((-1,))

        # 3. 在最细网格上求解一次最小特征值问题 
        A = A[isFreeDof, :][:, isFreeDof].tocsr()
        M = M[isFreeDof, :][:, isFreeDof].tocsr()
        uh[isFreeDof], d = picard(A, M, uh[isFreeDof], ml=ml)
        end = timer()

        print("smallest eigns:", d, "with time: ", end - start)
        return uh

    def alg_3(self):
        """
        1. 自适应在每层网格上求解最小特征值问题
        """
        print("算法 3")
        start = timer()
        if self.step == 0:
            idx = []
        else:
            idx =list(range(0, self.maxit, self.step)) + [self.maxit-1]

        mesh = self.pde.init_mesh(n=self.numrefine)
        integrator = mesh.integrator(self.q)

        space = LagrangeFiniteElementSpace(mesh, 1)
        isFreeDof = ~(space.boundary_dof())
        gdof = space.number_of_global_dofs()
        uh = np.ones(gdof, dtype=mesh.ftype)
        uh[~isFreeDof] = 0
        IM = eye(gdof)
        GD = mesh.geo_dimension()
        for i in range(self.maxit+1):
            print(i, ":", gdof)
            area = mesh.entity_measure('cell')
            A = self.get_stiff_matrix(space, integrator, area)
            M = self.get_mass_matrix(space, integrator, area)
            uh = IM@uh
            A = A[isFreeDof, :][:, isFreeDof].tocsr()
            M = M[isFreeDof, :][:, isFreeDof].tocsr()
            uh[isFreeDof], d = picard(A, M, uh[isFreeDof])

            if (self.step > 0) and (i in idx):
                NN = mesh.number_of_nodes()
                fig = plt.figure()
                fig.set_facecolor('white')
                if GD == 2:
                    axes = fig.gca()
                else:
                    axes = Axes3D(fig)
                mesh.add_plot(axes, cellcolor='w')
                fig.savefig(self.resultdir + 'mesh_3_' + str(i) + '_' + str(NN) + '.pdf')
                plt.close()

            if i < self.maxit:
                uh = space.function(array=uh)
                eta = self.residual_estimate(uh)
                markedCell = mark(eta, self.theta)
                IM = mesh.bisect(markedCell, returnim=True)
                space = LagrangeFiniteElementSpace(mesh, 1)
                isFreeDof = ~(space.boundary_dof())
                gdof = space.number_of_global_dofs()

        end = timer()
        print("smallest eigns:", d, "with time: ", end - start)
        uh = space.function(array=uh)
        return uh

    def alg_4(self):
        """
        1. 自适应求解 -\Delta u = 1。
        1. 在最细网格上求最小特征值和特征向量。
        """
        print("算法 4")
        start = timer()
        if self.step == 0:
            idx = []
        else:
            idx =list(range(0, self.maxit, self.step)) + [self.maxit-1]

        mesh = self.pde.init_mesh(n=self.numrefine)
        integrator = mesh.integrator(self.q)
        GD = mesh.geo_dimension()
        for i in range(self.maxit+1):
            space = LagrangeFiniteElementSpace(mesh, 1)
            gdof = space.number_of_global_dofs()
            print(i, ":", gdof)

            area = mesh.entity_measure('cell')
            A = self.get_stiff_matrix(space, integrator, area)
            M = self.get_mass_matrix(space, integrator, area)
            b = M@np.ones(gdof)

            isFreeDof = ~(space.boundary_dof())
            A = A[isFreeDof, :][:, isFreeDof].tocsr()
            M = M[isFreeDof, :][:, isFreeDof].tocsr()

            ml = pyamg.ruge_stuben_solver(A)
            uh = space.function()
            uh[isFreeDof] = ml.solve(b[isFreeDof], tol=1e-12, accel='cg').reshape((-1,))

            if (self.step > 0) and (i in idx):
                NN = mesh.number_of_nodes()
                fig = plt.figure()
                fig.set_facecolor('white')
                if GD == 2:
                    axes = fig.gca()
                else:
                    axes = Axes3D(fig)
                mesh.add_plot(axes, cellcolor='w')
                fig.savefig(self.resultdir + 'mesh_4_' + str(i) + '_' + str(NN) +'.pdf')
                plt.close()

            if i < self.maxit:
                eta = self.residual_estimate(uh)
                markedCell = mark(eta, self.theta)
                mesh.bisect(markedCell)

        uh[isFreeDof], d = picard(A, M, uh[isFreeDof], ml=ml)
        end = timer()

        print("smallest eigns:", d, "with time: ", end - start)
        return uh

    def savemesh(self, uh, fname):
        mesh = uh.space.mesh
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        data = {'node': node, 'elem': cell+1}
        sio.matlab.savemat(self.resultdir + fname, data)


    def savesolution(self, uh, fname):
        mesh = uh.space.mesh
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        data = {'node': node, 'elem': cell+1, 'solution': uh}
        sio.matlab.savemat(self.resultdir + fname, data)
