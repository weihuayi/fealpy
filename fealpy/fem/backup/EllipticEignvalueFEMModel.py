import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import eye, csr_matrix, bmat
from scipy.sparse.linalg import spsolve, eigs, LinearOperator
import scipy.io as sio
import pyamg
from timeit import default_timer as timer
from mpl_toolkits.mplot3d import Axes3D

from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.solver.eigns import picard
from fealpy.quadrature import FEMeshIntegralAlg
from fealpy.mesh.adaptive_tools import mark


class EllipticEignvalueFEMModel:
    def __init__(self, pde, theta=0.2, maxit=50, step=0, maxdof=1e5, n=3, p=1, q=3,
            sigma=None, multieigs=False, matlab=False, resultdir='~/'):
        self.multieigs = multieigs
        self.sigma = sigma
        self.pde = pde
        self.step = step
        self.theta = theta
        self.maxit = maxit
        self.maxdof = maxdof
        self.p = p
        self.q = q
        self.numrefine = n
        self.resultdir = resultdir
        self.picard = False
        self.matlab = matlab

    def residual_estimate(self, uh):
        mesh = uh.space.mesh
        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()

        n = mesh.face_unit_normal()
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

        cell2cell = mesh.ds.cell_to_cell()
        cell2face = mesh.ds.cell_to_face()
        measure = mesh.entity_measure('cell')
        J = 0
        for i in range(cell2cell.shape[1]):
            J += np.sum((grad - grad[cell2cell[:, i]])*n[cell2face[:, i]], axis=-1)**2
        return np.sqrt(J*measure)

    def get_stiff_matrix(self, space):
        try:
            A = space.stiff_matrix(cfun=self.pde.diffusion_coefficient)
        except AttributeError:
            A = space.stiff_matrix()

        try:
            A += space.mass_matrix(cfun=self.pde.reaction_coefficient)
        except AttributeError:
            pass

        return A

    def get_mass_matrix(self, space):
        M = space.mass_matrix()
        return M

    def linear_operator(self, x):
        y = self.M@x
        z = self.ml.solve(y, tol=1e-12, accel='cg').reshape(-1)
        return z

    def eigs(self, k=50):
        NN = self.A.shape[0]
        P = LinearOperator((NN, NN), matvec=self.linear_operator)
        vals, vecs = eigs(P, k=k)
        print(1/vals.real)

    def eig(self, A, M):
        NN = A.shape[0]
        self.M = M
        if self.sigma is None:
            self.ml = pyamg.ruge_stuben_solver(A)
        else:
            self.ml = pyamg.ruge_stuben_solver(A + self.sigma*M)
        P = LinearOperator((NN, NN), matvec=self.linear_operator)
        vals, vecs = eigs(P, k=1)
        if self.sigma is None:
            return vecs.reshape(-1).real, 1/vals.real[0]
        else:
            return vecs.reshape(-1).real, 1/vals.real[0] - self.sigma

    def psolve(self, A, b, M):
        if self.sigma is None:
            self.ml = pyamg.ruge_stuben_solver(A)
            return self.ml.solve(b, tol=1e-12, accel='cg').reshape((-1,))
        else:
            self.ml = pyamg.ruge_stuben_solver(A+self.sigma*M)
            return self.ml.solve(b, tol=1e-12, accel='cg').reshape((-1,))

    def deig(self, A, M):
        vals, vecs = eigs(A, k=1, M=M, which='SM')
        return vecs.reshape(-1).real, vals.real[0]

    def msolve(self, A, b):
        x = self.matlab.mldivide(A, b.reshape(-1, 1))
        return x.reshape(-1)

    def meigs(self, A, M):
        u, d, flag = self.matlab._call('eigs', [A + 100*M, M, 1, 'SM'])
        print("matlab eigs convergence:", flag)
        return u.reshape(-1), d - 100

    def alg_0(self, maxit=None):
        """
        1. 最粗网格上求解最小特征特征值问题，得到最小特征值 d_H 和特征向量 u_H
        2. 自适应求解  - \Delta u_h = d_H*u_H
            *  每层网格上求出的 u_h，插值到下一层网格上做为 u_H
            *  并更新 d_H = u_h@A@u_h/u_h@M@u_h， 其中 A 是当前网格层上的刚度矩
               阵，M 为当前网格层的质量矩阵。

        自适应 maxit， picard： 1
        """
        print("算法 0")
        if maxit is None:
            maxit = self.maxit

        start = timer()
        if self.step == 0:
            idx = []
        else:
            idx =list(range(0, self.maxit, self.step)) + [self.maxit-1]

        mesh = self.pde.init_mesh(n=self.numrefine)

        # 1. 粗网格上求解最小特征值问题
        space = LagrangeFiniteElementSpace(mesh, 1)
        gdof = space.number_of_global_dofs()
        print("initial mesh:", gdof)
        uh = np.zeros(gdof, dtype=np.float)
        AH = self.get_stiff_matrix(space)
        MH = self.get_mass_matrix(space)
        isFreeHDof = ~(space.boundary_dof())
        A = AH[isFreeHDof, :][:, isFreeHDof].tocsr()
        M = MH[isFreeHDof, :][:, isFreeHDof].tocsr()

        if self.picard is True:
            uh[isFreeHDof], d = picard(A, M, np.ones(sum(isFreeHDof)), sigma=self.sigma)
        else:
            uh[isFreeHDof], d = self.eig(A, M)

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
            self.savemesh(mesh, self.resultdir + 'mesh_0_0_' + str(NN) +'.mat')

        # 2. 以 u_h 为右端项自适应求解 -\Deta u = d*u_h
        I = eye(gdof)
        for i in range(maxit):
            uh = space.function(array=uh)
            eta = self.residual_estimate(uh)
            markedCell = mark(eta, self.theta)
            IM = mesh.bisect(markedCell, returnim=True)
            print(i+1, "refine: ", mesh.number_of_nodes())
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
                self.savemesh(
                        mesh,
                        self.resultdir + 'mesh_0_' + str(i+1) + '_' + str(NN) +'.mat')

            I = IM@I
            uh = IM@uh
            space = LagrangeFiniteElementSpace(mesh, 1)
            gdof = space.number_of_global_dofs()
            A = self.get_stiff_matrix(space)
            M = self.get_mass_matrix(space)
            isFreeDof = ~(space.boundary_dof())
            b = d*M@uh
            if self.sigma is None:
                ml = pyamg.ruge_stuben_solver(A[isFreeDof, :][:, isFreeDof].tocsr())
                uh[isFreeDof] = ml.solve(b[isFreeDof], x0=uh[isFreeDof], tol=1e-12, accel='cg').reshape((-1,))
            else:
                K = A[isFreeDof, :][:, isFreeDof].tocsr() + self.sigma*M[isFreeDof, :][:, isFreeDof].tocsr()
                b += self.sigma*M@uh
                ml = pyamg.ruge_stuben_solver(K)
                uh[isFreeDof] = ml.solve(b[isFreeDof], x0=uh[isFreeDof], tol=1e-12, accel='cg').reshape(-1)
                # uh[isFreeDof] = spsolve(A[isFreeDof, :][:, isFreeDof].tocsr(), b[isFreeDof])
            d = uh@A@uh/(uh@M@uh)

            if gdof > self.maxdof:
                break



        if self.multieigs is True:
            self.A = A[isFreeDof, :][:, isFreeDof].tocsr()
            self.M = M[isFreeDof, :][:, isFreeDof].tocsr()
            self.ml = pyamg.ruge_stuben_solver(self.A)
            self.eigs()

        end = timer()
        print("smallest eigns:", d, "with time: ", end - start)

        uh = space.function(array=uh)
        return uh

    def alg_3_1(self, maxit=None):
        """
        1. 自适应在每层网格上求解最小特征值问题

        refine: maxit, picard： maxit + 1
        """
        print("算法 3.1")

        if maxit is None:
            maxit = self.maxit

        start = timer()
        if self.step == 0:
            idx = []
        else:
            idx =list(range(0, self.maxit, self.step)) + [self.maxit-1]

        mesh = self.pde.init_mesh(n=self.numrefine)
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
            fig.savefig(self.resultdir + 'mesh_3_1_0_' + str(NN) + '.pdf')
            plt.close()
            self.savemesh(mesh,
                    self.resultdir + 'mesh_3_1_0_' + str(NN) + '.mat')

        space = LagrangeFiniteElementSpace(mesh, 1)
        isFreeDof = ~(space.boundary_dof())
        gdof = space.number_of_global_dofs()
        uh = np.ones(gdof, dtype=mesh.ftype)
        uh[~isFreeDof] = 0
        IM = eye(gdof)
        for i in range(maxit+1):
            area = mesh.entity_measure('cell')
            A = self.get_stiff_matrix(space)
            M = self.get_mass_matrix(space)
            uh = IM@uh
            A = A[isFreeDof, :][:, isFreeDof].tocsr()
            M = M[isFreeDof, :][:, isFreeDof].tocsr()

            if self.matlab is False:
                uh[isFreeDof], d = self.eig(A, M)
            else:
                uh[isFreeDof], d = self.meigs(A, M)

            if i < maxit:
                uh = space.function(array=uh)
                eta = self.residual_estimate(uh)
                markedCell = mark(eta, self.theta)
                IM = mesh.bisect(markedCell, returnim=True)
                print(i+1, "refine: ", mesh.number_of_nodes())

                if (self.step > 0) and (i in idx):
                    NN = mesh.number_of_nodes()
                    fig = plt.figure()
                    fig.set_facecolor('white')
                    if GD == 2:
                        axes = fig.gca()
                    else:
                        axes = Axes3D(fig)
                    mesh.add_plot(axes, cellcolor='w')
                    fig.savefig(self.resultdir + 'mesh_3_1_' + str(i+1) + '_' + str(NN) + '.pdf')
                    plt.close()
                    self.savemesh(mesh,
                            self.resultdir + 'mesh_3_1_' + str(i+1) + '_' + str(NN) + '.mat')

                space = LagrangeFiniteElementSpace(mesh, 1)
                isFreeDof = ~(space.boundary_dof())
                gdof = space.number_of_global_dofs()
            if gdof > self.maxdof:
                break

        if self.step > 0:
            NN = mesh.number_of_nodes()
            fig = plt.figure()
            fig.set_facecolor('white')
            if GD == 2:
                axes = fig.gca()
            else:
                axes = Axes3D(fig)
            mesh.add_plot(axes, cellcolor='w')
            fig.savefig(self.resultdir + 'mesh_3_1_' + str(i+1) + '_' + str(NN) + '.pdf')
            plt.close()
            self.savemesh(mesh,
                    self.resultdir + 'mesh_3_1_' + str(i+1) + '_' + str(NN) + '.mat')


        if self.multieigs is True:
            self.A = A
            self.M = M
            self.ml = pyamg.ruge_stuben_solver(self.A)
            self.eigs()

        end = timer()
        print("smallest eigns:", d, "with time: ", end - start)

        uh = space.function(array=uh)
        return uh

    def alg_3_2(self, maxit=None):
        """
        1. 自适应求解 -\Delta u = 1。
        1. 在最细网格上求最小特征值和特征向量。

        refine maxit， picard: 1
        """
        print("算法 3.2")
        if maxit is None:
            maxit = self.maxit

        start = timer()
        if self.step == 0:
            idx = []
        else:
            idx =list(range(0, self.maxit, self.step)) + [self.maxit-1]

        mesh = self.pde.init_mesh(n=self.numrefine)
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
            fig.savefig(self.resultdir + 'mesh_3_2_0_' + str(NN) +'.pdf')
            plt.close()
            self.savemesh(mesh,
                    self.resultdir + 'mesh_3_2_0_' + str(NN) +'.mat')

        final = 0
        integrator = mesh.integrator(self.q)
        for i in range(maxit):
            space = LagrangeFiniteElementSpace(mesh, 1)
            gdof = space.number_of_global_dofs()

            A = self.get_stiff_matrix(space)
            M = self.get_mass_matrix(space)
            b = M@np.ones(gdof)

            isFreeDof = ~(space.boundary_dof())
            A = A[isFreeDof, :][:, isFreeDof].tocsr()
            M = M[isFreeDof, :][:, isFreeDof].tocsr()

            uh = space.function()
            if self.matlab is False:
                uh[isFreeDof] = self.psolve(A, b[isFreeDof], M)
            else:
                uh[isFreeDof] = self.msolve(A, b[isFreeDof])


            eta = self.residual_estimate(uh)
            markedCell = mark(eta, self.theta)
            IM = mesh.bisect(markedCell, returnim=True)
            NN = mesh.number_of_nodes()
            print(i+1, "refine :", NN)
            if (self.step > 0) and (i in idx):
                fig = plt.figure()
                fig.set_facecolor('white')
                if GD == 2:
                    axes = fig.gca()
                else:
                    axes = Axes3D(fig)
                mesh.add_plot(axes, cellcolor='w')
                fig.savefig(self.resultdir + 'mesh_3_2_' + str(i+1) + '_' + str(NN) +'.pdf')
                plt.close()
                self.savemesh(mesh,
                        self.resultdir + 'mesh_3_2_' + str(i+1) + '_' + str(NN) +'.mat')
            final = i+1
            if NN > self.maxdof:
                break

        if self.step > 0:
            NN = mesh.number_of_nodes()
            fig = plt.figure()
            fig.set_facecolor('white')
            if GD == 2:
                axes = fig.gca()
            else:
                axes = Axes3D(fig)
            mesh.add_plot(axes, cellcolor='w')
            fig.savefig(self.resultdir + 'mesh_3_2_' + str(final) + '_' + str(NN) +'.pdf')
            plt.close()
            self.savemesh(mesh,
                    self.resultdir + 'mesh_3_2_' + str(final) + '_' + str(NN) +'.mat')

        space = LagrangeFiniteElementSpace(mesh, 1)
        gdof = space.number_of_global_dofs()

        A = self.get_stiff_matrix(space)
        M = self.get_mass_matrix(space)
        isFreeDof = ~(space.boundary_dof())
        A = A[isFreeDof, :][:, isFreeDof].tocsr()
        M = M[isFreeDof, :][:, isFreeDof].tocsr()


        if self.multieigs is True:
            self.A = A
            self.M = M
            self.ml = pyamg.ruge_stuben_solver(self.A)
            self.eigs()
        else:
            uh = IM@uh
            if self.matlab is False:
                uh[isFreeDof], d = self.eig(A, M)
            else:
                uh[isFreeDof], d = self.meigs(A, M)
            print("smallest eigns:", d)
            end = timer()
            print("with time: ", end - start)
            return space.function(array=uh)

    def alg_3_3(self, maxit=None):
        """
        1. 最粗网格上求解最小特征特征值问题，得到最小特征值 d_H 和特征向量 u_H
        2. 自适应求解  - \Delta u_h = u_H
            *  u_H 插值到下一层网格上做为新 u_H
        3. 在最细网格上求解一次最小特征值问题

        自适应 maxit, picard:2
        """
        print("算法 3.3")

        if maxit is None:
            maxit = self.maxit

        start = timer()

        if self.step == 0:
            idx = []
        else:
            idx =list(range(0, self.maxit, self.step))

        mesh = self.pde.init_mesh(n=self.numrefine)
        # 1. 粗网格上求解最小特征值问题
        space = LagrangeFiniteElementSpace(mesh, 1)
        AH = self.get_stiff_matrix(space)
        MH = self.get_mass_matrix(space)
        isFreeHDof = ~(space.boundary_dof())

        gdof = space.number_of_global_dofs()
        uH = np.zeros(gdof, dtype=mesh.ftype)
        print("initial mesh :", gdof)

        A = AH[isFreeHDof, :][:, isFreeHDof].tocsr()
        M = MH[isFreeHDof, :][:, isFreeHDof].tocsr()
        if self.matlab is False:
            uH[isFreeHDof], d = self.eig(A, M)
        else:
            uH[isFreeHDof], d = self.meigs(A, M)

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
            fig.savefig(self.resultdir + 'mesh_3_3_0_' + str(NN) + '.pdf')
            plt.close()
            self.savemesh(mesh, self.resultdir + 'mesh_3_3_0_' + str(NN) + '.mat')

        # 2. 以 u_H 为右端项自适应求解 -\Deta u = u_H
        I = eye(gdof)
        final = 0
        for i in range(maxit-1):
            eta = self.residual_estimate(uh)
            markedCell = mark(eta, self.theta)
            IM = mesh.bisect(markedCell, returnim=True)
            NN = mesh.number_of_nodes()
            print(i+1, "refine : ", NN)
            if (self.step > 0) and (i in idx):
                NN = mesh.number_of_nodes()
                fig = plt.figure()
                fig.set_facecolor('white')
                if GD == 2:
                    axes = fig.gca()
                else:
                    axes = Axes3D(fig)
                mesh.add_plot(axes, cellcolor='w')
                fig.savefig(self.resultdir + 'mesh_3_3_' + str(i+1) + '_' + str(NN) +'.pdf')
                plt.close()
                self.savemesh(mesh,
                        self.resultdir + 'mesh_3_3_' + str(i+1) + '_' + str(NN) +'.mat')
            final = i+1

            I = IM@I
            uH = IM@uH

            if NN > self.maxdof:
                break

            space = LagrangeFiniteElementSpace(mesh, 1)
            gdof = space.number_of_global_dofs()

            A = self.get_stiff_matrix(space)
            M = self.get_mass_matrix(space)
            isFreeDof = ~(space.boundary_dof())
            b = M@uH

            uh = space.function()
            if self.matlab is False:
                uh[isFreeDof] = self.psolve(
                        A[isFreeDof, :][:, isFreeDof].tocsr(),
                        b[isFreeDof],
                        M[isFreeDof, :][:, isFreeDof].tocsr())
            else:
                uh[isFreeDof] = self.msolve(A[isFreeDof, :][:, isFreeDof].tocsr(), b[isFreeDof])


        # 3. 在最细网格上求解一次最小特征值问题 

        if self.step > 0:
            NN = mesh.number_of_nodes()
            fig = plt.figure()
            fig.set_facecolor('white')
            if GD == 2:
                axes = fig.gca()
            else:
                axes = Axes3D(fig)
            mesh.add_plot(axes, cellcolor='w')
            fig.savefig(self.resultdir + 'mesh_3_3_' + str(final) + '_' + str(NN) +'.pdf')
            plt.close()
            self.savemesh(mesh, self.resultdir + 'mesh_3_3_' + str(final) + '_' + str(NN) +'.mat')

        space = LagrangeFiniteElementSpace(mesh, 1)
        gdof = space.number_of_global_dofs()
        A = self.get_stiff_matrix(space)
        M = self.get_mass_matrix(space)
        isFreeDof = ~(space.boundary_dof())
        uh = space.function(array=uh)
        Ah = A[isFreeDof, :][:, isFreeDof].tocsr()
        Mh = M[isFreeDof, :][:, isFreeDof].tocsr()
        uh = space.function()

        if self.multieigs is True:
            self.A = Ah
            self.M = Mh
            self.ml = pyamg.ruge_stuben_solver(self.A)
            self.eigs()
        else:
            if self.matlab is False:
                uh[isFreeDof], d = self.eig(Ah, Mh)
            else:
                uh[isFreeDof], d = self.meigs(Ah, Mh)
            print("smallest eigns:", d)
            end = timer()
            print("with time: ", end - start)

            if False:
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

                if self.matlab is False:
                    u[isFreeDof], d = self.eig(A, M)
                else:
                    u[isFreeDof], d = self.meigs(A, M)

                print("new smallest eigns:", d)

            return uh

    def alg_3_4(self, maxit=None):
        """
        1. 最粗网格上求解最小特征特征值问题，得到最小特征值 d_H 和特征向量 u_H
        2. 自适应求解  - \Delta u_h = u_H
            *  u_H 插值到下一层网格上做为新 u_H
        3. 最新网格层上求出的 uh 做为一个基函数，加入到最粗网格的有限元空间中，
           求解最小特征值问题。

        自适应 maxit， picard:2
        """
        print("算法 3.4")
        if maxit is None:
            maxit = self.maxit

        start = timer()
        if self.step == 0:
            idx = []
        else:
            idx =list(range(0, self.maxit, self.step)) + [self.maxit-1]

        mesh = self.pde.init_mesh(n=self.numrefine)
        integrator = mesh.integrator(self.q)

        # 1. 粗网格上求解最小特征值问题
        space = LagrangeFiniteElementSpace(mesh, 1)
        AH = self.get_stiff_matrix(space)
        MH = self.get_mass_matrix(space)
        isFreeHDof = ~(space.boundary_dof())

        gdof = space.number_of_global_dofs()
        print("initial mesh :", gdof)

        uH = np.zeros(gdof, dtype=np.float)

        A = AH[isFreeHDof, :][:, isFreeHDof].tocsr()
        M = MH[isFreeHDof, :][:, isFreeHDof].tocsr()

        if self.matlab is False:
            uH[isFreeHDof], d = self.eig(A, M)
        else:
            uH[isFreeHDof], d = self.meigs(A, M)

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
            fig.savefig(self.resultdir + 'mesh_3_4_0_' + str(NN) + '.pdf')
            plt.close()
            self.savemesh(mesh,
                    self.resultdir + 'mesh_3_4_0_' + str(NN) + '.mat')


        # 2. 以 u_H 为右端项自适应求解 -\Deta u = u_H
        I = eye(gdof)
        final = 0
        for i in range(maxit):
            eta = self.residual_estimate(uh)
            markedCell = mark(eta, self.theta)
            IM = mesh.bisect(markedCell, returnim=True)
            print(i+1, "refine :", mesh.number_of_nodes())
            if (self.step > 0) and (i in idx):
                NN = mesh.number_of_nodes()
                fig = plt.figure()
                fig.set_facecolor('white')
                if GD == 2:
                    axes = fig.gca()
                else:
                    axes = Axes3D(fig)
                mesh.add_plot(axes, cellcolor='w')
                fig.savefig(self.resultdir + 'mesh_3_4_' + str(i+1) + '_' + str(NN) +'.pdf')
                plt.close()
                self.savemesh(mesh,
                        self.resultdir + 'mesh_3_4_' + str(i+1) + '_' + str(NN) +'.mat')

            I = IM@I
            uH = IM@uH

            space = LagrangeFiniteElementSpace(mesh, 1)
            gdof = space.number_of_global_dofs()

            A = self.get_stiff_matrix(space)
            M = self.get_mass_matrix(space)
            isFreeDof = ~(space.boundary_dof())
            b = M@uH

            uh = space.function()
            if self.matlab is False:
                uh[isFreeDof] = self.psolve(
                        A[isFreeDof, :][:, isFreeDof].tocsr(),
                        b[isFreeDof],
                        M[isFreeDof, :][:, isFreeDof].tocsr())
            else:
                uh[isFreeDof] = self.msolve(A[isFreeDof, :][:, isFreeDof].tocsr(), b[isFreeDof])
            final = i+1
            if gdof > self.maxdof:
                break


        if self.step > 0:
            NN = mesh.number_of_nodes()
            fig = plt.figure()
            fig.set_facecolor('white')
            if GD == 2:
                axes = fig.gca()
            else:
                axes = Axes3D(fig)
            mesh.add_plot(axes, cellcolor='w')
            fig.savefig(self.resultdir + 'mesh_3_4_' + str(final) + '_' + str(NN) +'.pdf')
            plt.close()
            self.savemesh(mesh,
                    self.resultdir + 'mesh_3_4_' + str(final) + '_' + str(NN) +'.mat')

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

        if self.multieigs is True:
            self.A = A
            self.M = M
            self.ml = pyamg.ruge_stuben_solver(self.A)
            self.eigs()
        else:
            if self.matlab is False:
                u[isFreeDof], d = self.eig(A, M)
            else:
                u[isFreeDof], d = self.meigs(A, M)

            print("smallest eigns:", d)
            end = timer()
            print("with time: ", end - start)
            uh *= u[-1]
            uh += I@u[:-1]

            uh /= np.max(np.abs(uh))
            uh = space.function(array=uh)
            return uh

    def alg_3_5(self, maxit=None):
        """
        1. 最粗网格上求解最小特征特征值问题，得到最小特征值 d_H 和特征向量 u_H
        2. 自适应求解  - \Delta u_h = u_H
            *  u_H 插值到下一层网格上做为新 u_H
        3. 在最细网格上求解一次最小特征值问题

        自适应 maxit, picard:2
        """
        print("算法 3.5")

        if maxit is None:
            maxit = self.maxit

        start = timer()

        if self.step == 0:
            idx = []
        else:
            idx =list(range(0, self.maxit, self.step))

        mesh = self.pde.init_mesh(n=self.numrefine)
        # 1. 粗网格上求解最小特征值问题
        space = LagrangeFiniteElementSpace(mesh, 1)
        AH = self.get_stiff_matrix(space)
        MH = self.get_mass_matrix(space)
        isFreeHDof = ~(space.boundary_dof())

        gdof = space.number_of_global_dofs()
        uH = np.zeros(gdof, dtype=mesh.ftype)
        print("initial mesh :", gdof)

        A = AH[isFreeHDof, :][:, isFreeHDof].tocsr()
        M = MH[isFreeHDof, :][:, isFreeHDof].tocsr()
        if self.matlab is False:
            uH[isFreeHDof], d = self.eig(A, M)
        else:
            uH[isFreeHDof], d = self.meigs(A, M)

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
            fig.savefig(self.resultdir + 'mesh_3_5_0_' + str(NN) + '.pdf')
            plt.close()
            self.savemesh(mesh, self.resultdir + 'mesh_3_5_0_' + str(NN) + '.mat')

        # 2. 以 u_H 为右端项自适应求解 -\Deta u = u_H

        I = eye(gdof)
        final = 0
        while True:
            eta = self.residual_estimate(uh)
            markedCell = mark(eta, self.theta)
            IM = mesh.bisect(markedCell, returnim=True)
            print(final+1, "refine :", mesh.number_of_nodes())
            if (self.step > 0) and (final in idx):
                NN = mesh.number_of_nodes()
                fig = plt.figure()
                fig.set_facecolor('white')
                if GD == 2:
                    axes = fig.gca()
                else:
                    axes = Axes3D(fig)
                mesh.add_plot(axes, cellcolor='w')
                fig.savefig(self.resultdir + 'mesh_3_5_' + str(final+1) + '_' + str(NN) +'.pdf')
                plt.close()
                self.savemesh(mesh,
                        self.resultdir + 'mesh_3_5_' + str(final+1) + '_' + str(NN) +'.mat')

            I = IM@I
            uH = IM@uH

            space = LagrangeFiniteElementSpace(mesh, 1)
            gdof = space.number_of_global_dofs()

            A = self.get_stiff_matrix(space)
            M = self.get_mass_matrix(space)
            isFreeDof = ~(space.boundary_dof())
            b = M@uH

            uh = space.function()
            if self.matlab is False:
                uh[isFreeDof] = self.psolve(
                        A[isFreeDof, :][:, isFreeDof].tocsr(),
                        b[isFreeDof],
                        M[isFreeDof, :][:, isFreeDof].tocsr())
            else:
                uh[isFreeDof] = self.msolve(A[isFreeDof, :][:, isFreeDof].tocsr(), b[isFreeDof])
            final += 1
            if gdof > 4e+4:
                break

        # 1. 自适应粗网格上求解最小特征值问题
        space = LagrangeFiniteElementSpace(mesh, 1)
        AH = self.get_stiff_matrix(space)
        MH = self.get_mass_matrix(space)
        isFreeHDof = ~(space.boundary_dof())

        gdof = space.number_of_global_dofs()
        print("initial mesh :", gdof)

        uH = np.zeros(gdof, dtype=np.float)

        A = AH[isFreeHDof, :][:, isFreeHDof].tocsr()
        M = MH[isFreeHDof, :][:, isFreeHDof].tocsr()

        if self.matlab is False:
            uH[isFreeHDof], d = self.eig(A, M)
        else:
            uH[isFreeHDof], d = self.meigs(A, M)

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
            fig.savefig(self.resultdir + 'mesh_3_5_' + str(final) + '_' + str(NN) + 'H.pdf')
            plt.close()
            self.savemesh(mesh, self.resultdir + 'mesh_3_5_' + str(final) + '_' + str(NN) + 'H.mat')

        I = eye(gdof)
        for i in range(final, maxit-1):
            eta = self.residual_estimate(uh)
            markedCell = mark(eta, self.theta)
            IM = mesh.bisect(markedCell, returnim=True)
            NN = mesh.number_of_nodes()
            print(i+1, "refine : ", NN)
            if (self.step > 0) and (i in idx):
                NN = mesh.number_of_nodes()
                fig = plt.figure()
                fig.set_facecolor('white')
                if GD == 2:
                    axes = fig.gca()
                else:
                    axes = Axes3D(fig)
                mesh.add_plot(axes, cellcolor='w')
                fig.savefig(self.resultdir + 'mesh_3_5_' + str(i+1) + '_' + str(NN) +'.pdf')
                plt.close()
                self.savemesh(mesh,
                        self.resultdir + 'mesh_3_3_' + str(i+1) + '_' + str(NN) +'.mat')
            final = i+1

            I = IM@I
            uH = IM@uH

            if NN > self.maxdof:
                break

            space = LagrangeFiniteElementSpace(mesh, 1)
            gdof = space.number_of_global_dofs()

            A = self.get_stiff_matrix(space)
            M = self.get_mass_matrix(space)
            isFreeDof = ~(space.boundary_dof())
            b = M@uH

            uh = space.function()
            if self.matlab is False:
                uh[isFreeDof] = self.psolve(
                        A[isFreeDof, :][:, isFreeDof].tocsr(),
                        b[isFreeDof],
                        M[isFreeDof, :][:, isFreeDof].tocsr())
            else:
                uh[isFreeDof] = self.msolve(A[isFreeDof, :][:, isFreeDof].tocsr(), b[isFreeDof])


        # 3. 在最细网格上求解一次最小特征值问题 

        if self.step > 0:
            NN = mesh.number_of_nodes()
            fig = plt.figure()
            fig.set_facecolor('white')
            if GD == 2:
                axes = fig.gca()
            else:
                axes = Axes3D(fig)
            mesh.add_plot(axes, cellcolor='w')
            fig.savefig(self.resultdir + 'mesh_3_5_' + str(final) + '_' + str(NN) +'f.pdf')
            plt.close()
            self.savemesh(mesh, self.resultdir + 'mesh_3_5_' + str(final) + '_' + str(NN) +'f.mat')

        space = LagrangeFiniteElementSpace(mesh, 1)
        gdof = space.number_of_global_dofs()
        A = self.get_stiff_matrix(space)
        M = self.get_mass_matrix(space)
        isFreeDof = ~(space.boundary_dof())
        uh = space.function(array=uh)
        Ah = A[isFreeDof, :][:, isFreeDof].tocsr()
        Mh = M[isFreeDof, :][:, isFreeDof].tocsr()
        uh = space.function()

        if self.multieigs is True:
            self.A = Ah
            self.M = Mh
            self.ml = pyamg.ruge_stuben_solver(self.A)
            self.eigs()
        else:
            if self.matlab is False:
                uh[isFreeDof], d = self.eig(Ah, Mh)
            else:
                uh[isFreeDof], d = self.meigs(Ah, Mh)
            print("smallest eigns:", d)
            end = timer()
            print("with time: ", end - start)

            return uh

    def alg_3_6(self, maxit=None):
        """
        1. 最粗网格上求解最小特征特征值问题，得到最小特征值 d_H 和特征向量 u_H
        2. 自适应求解  - \Delta u_h = u_H
            *  u_H 插值到下一层网格上做为新 u_H
        3. 最新网格层上求出的 uh 做为一个基函数，加入到最粗网格的有限元空间中，
           求解最小特征值问题。

        自适应 maxit， picard:2
        """
        print("算法 3.6")
        if maxit is None:
            maxit = self.maxit

        start = timer()
        if self.step == 0:
            idx = []
        else:
            idx =list(range(0, self.maxit, self.step)) + [self.maxit-1]

        mesh = self.pde.init_mesh(n=self.numrefine)
        integrator = mesh.integrator(self.q)

        # 1. 粗网格上求解最小特征值问题
        space = LagrangeFiniteElementSpace(mesh, 1)
        AH = self.get_stiff_matrix(space)
        MH = self.get_mass_matrix(space)
        isFreeHDof = ~(space.boundary_dof())

        gdof = space.number_of_global_dofs()
        print("initial mesh :", gdof)

        uH = np.zeros(gdof, dtype=np.float)

        A = AH[isFreeHDof, :][:, isFreeHDof].tocsr()
        M = MH[isFreeHDof, :][:, isFreeHDof].tocsr()

        if self.matlab is False:
            uH[isFreeHDof], d = self.eig(A, M)
        else:
            uH[isFreeHDof], d = self.meigs(A, M)

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
            fig.savefig(self.resultdir + 'mesh_3_6_0_' + str(NN) + '.pdf')
            plt.close()
            self.savemesh(mesh,
                    self.resultdir + 'mesh_3_6_0_' + str(NN) + '.mat')


        # 2. 以 u_H 为右端项自适应求解 -\Deta u = u_H
        I = eye(gdof)
        final = 0

        while True:
            eta = self.residual_estimate(uh)
            markedCell = mark(eta, self.theta)
            IM = mesh.bisect(markedCell, returnim=True)
            print(final+1, "refine :", mesh.number_of_nodes())
            if (self.step > 0) and (final in idx):
                NN = mesh.number_of_nodes()
                fig = plt.figure()
                fig.set_facecolor('white')
                if GD == 2:
                    axes = fig.gca()
                else:
                    axes = Axes3D(fig)
                mesh.add_plot(axes, cellcolor='w')
                fig.savefig(self.resultdir + 'mesh_3_6_' + str(final+1) + '_' + str(NN) +'.pdf')
                plt.close()
                self.savemesh(mesh,
                        self.resultdir + 'mesh_3_6_' + str(final+1) + '_' + str(NN) +'.mat')

            I = IM@I
            uH = IM@uH

            space = LagrangeFiniteElementSpace(mesh, 1)
            gdof = space.number_of_global_dofs()

            A = self.get_stiff_matrix(space)
            M = self.get_mass_matrix(space)
            isFreeDof = ~(space.boundary_dof())
            b = M@uH

            uh = space.function()
            if self.matlab is False:
                uh[isFreeDof] = self.psolve(
                        A[isFreeDof, :][:, isFreeDof].tocsr(),
                        b[isFreeDof],
                        M[isFreeDof, :][:, isFreeDof].tocsr())
            else:
                uh[isFreeDof] = self.msolve(A[isFreeDof, :][:, isFreeDof].tocsr(), b[isFreeDof])
            final += 1
            if gdof > 4e+4:
                break

        # 1. 自适应粗网格上求解最小特征值问题
        space = LagrangeFiniteElementSpace(mesh, 1)
        AH = self.get_stiff_matrix(space)
        MH = self.get_mass_matrix(space)
        isFreeHDof = ~(space.boundary_dof())

        gdof = space.number_of_global_dofs()
        print("initial mesh :", gdof)

        uH = np.zeros(gdof, dtype=np.float)

        A = AH[isFreeHDof, :][:, isFreeHDof].tocsr()
        M = MH[isFreeHDof, :][:, isFreeHDof].tocsr()

        if self.matlab is False:
            uH[isFreeHDof], d = self.eig(A, M)
        else:
            uH[isFreeHDof], d = self.meigs(A, M)

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
            fig.savefig(self.resultdir + 'mesh_3_6_' + str(final) + '_' + str(NN) + 'H.pdf')
            plt.close()
            self.savemesh(mesh, self.resultdir + 'mesh_3_6_' + str(final) + '_' + str(NN) + 'H.mat')

        I = eye(gdof)
        for i in range(final, maxit):
            eta = self.residual_estimate(uh)
            markedCell = mark(eta, self.theta)
            IM = mesh.bisect(markedCell, returnim=True)
            print(i+1, "refine :", mesh.number_of_nodes())
            if (self.step > 0) and (i in idx):
                NN = mesh.number_of_nodes()
                fig = plt.figure()
                fig.set_facecolor('white')
                if GD == 2:
                    axes = fig.gca()
                else:
                    axes = Axes3D(fig)
                mesh.add_plot(axes, cellcolor='w')
                fig.savefig(self.resultdir + 'mesh_3_4_' + str(i+1) + '_' + str(NN) +'.pdf')
                plt.close()
                self.savemesh(mesh,
                        self.resultdir + 'mesh_3_4_' + str(i+1) + '_' + str(NN) +'.mat')

            I = IM@I
            uH = IM@uH

            space = LagrangeFiniteElementSpace(mesh, 1)
            gdof = space.number_of_global_dofs()

            A = self.get_stiff_matrix(space)
            M = self.get_mass_matrix(space)
            isFreeDof = ~(space.boundary_dof())
            b = M@uH

            uh = space.function()
            if self.matlab is False:
                uh[isFreeDof] = self.psolve(
                        A[isFreeDof, :][:, isFreeDof].tocsr(),
                        b[isFreeDof],
                        M[isFreeDof, :][:, isFreeDof].tocsr())
            else:
                uh[isFreeDof] = self.msolve(A[isFreeDof, :][:, isFreeDof].tocsr(), b[isFreeDof])
            final = i+1
            if gdof > self.maxdof:
                break

        if self.step > 0:
            NN = mesh.number_of_nodes()
            fig = plt.figure()
            fig.set_facecolor('white')
            if GD == 2:
                axes = fig.gca()
            else:
                axes = Axes3D(fig)
            mesh.add_plot(axes, cellcolor='w')
            fig.savefig(self.resultdir + 'mesh_3_6_' + str(final) + '_' +
                    str(NN) +'f.pdf')
            plt.close()
            self.savemesh(mesh,
                    self.resultdir + 'mesh_3_6_' + str(final) + '_' + str(NN)
                    +'f.mat')

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

        if self.multieigs is True:
            self.A = A
            self.M = M
            self.ml = pyamg.ruge_stuben_solver(self.A)
            self.eigs()
        else:
            if self.matlab is False:
                u[isFreeDof], d = self.eig(A, M)
            else:
                u[isFreeDof], d = self.meigs(A, M)

            print("smallest eigns:", d)
            end = timer()
            print("with time: ", end - start)
            uh *= u[-1]
            uh += I@u[:-1]

            uh /= np.max(np.abs(uh))
            uh = space.function(array=uh)
            return uh





    def savemesh(self, mesh, fname):
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        data = {'node': node, 'elem': cell+1}
        sio.matlab.savemat(fname, data)


    def savesolution(self, uh, fname):
        mesh = uh.space.mesh
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        data = {'node': node, 'elem': cell+1, 'solution': uh}
        sio.matlab.savemat(self.resultdir + fname, data)
