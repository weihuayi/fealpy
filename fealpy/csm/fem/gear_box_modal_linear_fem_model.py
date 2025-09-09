
from typing import Any, Optional, Union

from fealpy.backend import bm
from fealpy.typing import TensorLike
from fealpy.decorator import variantmethod
from fealpy.model import ComputationalModel

from fealpy.mesh import Mesh
from fealpy.functionspace import functionspace 

from fealpy.fem import (
        BilinearForm,
        LinearElasticityIntegrator, 
        ScalarMassIntegrator as MassIntegrator
        )

from fealpy.sparse import coo_matrix

from ..model import CSMModelManager

class GearBoxModalLinearFEMModel(ComputationalModel):
    """
    A computational model for the gearbox modal analysis using linear finite
    element method (LFEM).

    Parameters:
        options (dict): A dictionary containing the options for the model.
    """
    def __init__(self, options):
        self.options = options
        super().__init__(
            pbar_log=options['pbar_log'],
            log_level=options['log_level']
        )

        self.set_pde(options['pde'])
        mesh = self.pde.init_mesh()
        self.set_mesh(mesh)

        self.itype = bm.int32
        self.ftype = bm.float64

    def set_pde(self, pde = 2) -> None:
        if isinstance(pde, int):
            self.pde = CSMModelManager("linear_elasticity").get_example(
                    pde, 
                    mesh_file=self.options['mesh_file'])
        else:
            self.pde = pde
        self.logger.info(self.pde)
        self.logger.info(self.pde.material)

    def set_mesh(self, mesh: Mesh) -> None:
        self.mesh = mesh
        self.logger.info(self.mesh)

    def rbe2_matrix(self): 
        """
        Construct the RBE2 matrix for the gearbox model.

        Notes:
            1. The RBE2 matrix is used to couple the gearbox shell nodes with
               the shaft system nodes.
            2. The RBE2 matrix is constructed based on the reference nodes and
               coupling nodes defined in the mesh data.
        """

        NN = self.mesh.number_of_nodes()
        redges, rnodes = self.mesh.data.get_rbe2_edge()

        # the flag of nodes coupling with the shaft system 
        isCNode = self.mesh.data.get_node_data('isCNode')
        self.mesh.nodedata['isCNode'] = isCNode

        # the flag of reference nodes
        isRNode = bm.zeros(NN, dtype=bm.bool)
        isRNode = bm.set_at(isRNode, rnodes, True)

        # the flag of gearbox nodes, which are not reference nodes 
        isGBNode = bm.logical_not(isRNode)

        # the flag of surface nodes, which are the gearbox nodes that maybe 
        # constrained by the reference nodes (notice that some reference nodes
        # maybe not in the shaft system)
        isSNode = bm.zeros(NN, dtype=bm.bool)
        isSNode = bm.set_at(isSNode, redges[:, 1], True)

        # the flag of RBE2 nodes, which are the coupling nodes that are
        # constrained by the nodes in the shaft system.
        isRBE2 = isCNode[redges[:, 0]]
        isCSNode = bm.zeros(NN, dtype=bm.bool)
        isCSNode = bm.set_at(isCSNode, redges[isRBE2, 1], True)
        
        # the flag of fixed gear box nodes, which are the nodes that are fixed 
        # TODO: we should update the code to support genearal case of boundary
        # conditions
        name = self.mesh.data['boundary_conditions'][0][0]
        nset = self.mesh.data.get_node_set(name)
        isFNode = bm.zeros(NN, dtype=bm.bool)
        isFNode[nset] = True

        # add above flags to the mesh data 
        self.mesh.data.add_node_data('isRNode', isRNode)
        self.mesh.data.add_node_data('isGBNode', isGBNode)
        self.mesh.data.add_node_data('isSNode', isSNode)
        self.mesh.data.add_node_data('isCSNode', isCSNode)
        self.mesh.data.add_node_data('isFNode', isFNode)

        # put the flags into mesh.nodedata for visualization
        self.mesh.nodedata['isRNode'] = isRNode
        self.mesh.nodedata['isGBNode'] = isGBNode
        self.mesh.nodedata['isSNode'] = isSNode
        self.mesh.nodedata['isCSNode'] = isCSNode
        self.mesh.nodedata['isFNode'] = isFNode

        # construct the RBE2 matrix 
        ridx = bm.where(isCNode)[0]
        NC = ridx.shape[0] # number of coupling nodes

        sidx = bm.where(isCSNode)[0]
        NS = sidx.shape[0] # number of surface nodes

        rmap = bm.full(NN, -1, dtype=self.itype)
        rmap = bm.set_at(rmap, ridx, bm.arange(NC, dtype=self.itype))

        smap = bm.full(NN, -1, dtype=self.itype)
        smap = bm.set_at(smap, sidx, bm.arange(NS, dtype=self.itype))


        I = smap[redges[isRBE2, 1]]
        J = rmap[redges[isRBE2, 0]]

        assert bm.all(I >= 0) and bm.all(J >= 0), "RBE2: I/J 含有 -1，映射未覆盖所有耦合边"

        node = self.mesh.entity('node')
        v = node[redges[isRBE2, 0]] - node[redges[isRBE2, 1]]

        self.logger.info(f"RBE2 matrix: {NS} surface nodes, {NC} reference nodes")

        kwargs = {'shape':(3*NS, 6*NC), 'itype': self.itype, 'dtype': self.ftype}
        ones = bm.ones(NS, dtype=self.ftype)

        G  = coo_matrix((    ones, (3*I+0, 6*J+0)), **kwargs)
        G += coo_matrix((-v[:, 2], (3*I+0, 6*J+4)), **kwargs)
        G += coo_matrix(( v[:, 1], (3*I+0, 6*J+5)), **kwargs)

        G += coo_matrix((    ones, (3*I+1, 6*J+1)), **kwargs)
        G += coo_matrix(( v[:, 2], (3*I+1, 6*J+3)), **kwargs)
        G += coo_matrix((-v[:, 0], (3*I+1, 6*J+5)), **kwargs)

        G += coo_matrix((    ones, (3*I+2, 6*J+2)), **kwargs)
        G += coo_matrix((-v[:, 1], (3*I+2, 6*J+3)), **kwargs)
        G += coo_matrix(( v[:, 0], (3*I+2, 6*J+4)), **kwargs)

        self.logger.info(f"RBE2 matrix shape: {G.shape}, nnz: {G.nnz}")
        G = G.coalesce() 
        self.logger.info(f"RBE2 matrix after coalesce: {G.shape}, nnz: {G.nnz}")

        self.G = G.tocsr().to_scipy()


    def box_linear_system(self):
        """
        Construct the linear system for the gearbox shell model.
        """

        GD = self.mesh.geo_dimension()
        self.space = functionspace(self.mesh, ('Lagrange', 1), shape=(-1, GD))

        bform = BilinearForm(self.space)
        integrator = LinearElasticityIntegrator(self.pde.material)
        integrator.assembly.set('fast')
        bform.add_integrator(integrator)
        S = bform.assembly()

        bform = BilinearForm(self.space)
        integrator = MassIntegrator(self.pde.material.density)
        bform.add_integrator(integrator)
        M = bform.assembly()

        S = S.to_scipy()
        M = M.to_scipy()

        S = (S + S.T)/2.0
        M = (M + M.T)/2.0


        # the flag of free nodes 
        isRNode = self.mesh.data.get_node_data('isRNode')
        isFNode = self.mesh.data.get_node_data('isFNode')
        isCSNode = self.mesh.data.get_node_data('isCSNode')
        isFreeNode = ~(isRNode | isFNode | isCSNode) 
        self.mesh.data.add_node_data('isFreeNode', isFreeNode)

        # the flag of free dofs
        isFreeDof = bm.repeat(isFreeNode, 3)
        isCSDof = bm.repeat(isCSNode, 3)

        S0 = S[isFreeDof, :]
        S1 = S[isCSDof, :]

        S00 = S0[:, isFreeDof]
        S01 = S0[:, isCSDof] @ self.G
        S11 = S1[:, isCSDof] @ self.G 
        S11 = self.G.T @ S11  


        M0 = M[isFreeDof, :]
        M1 = M[isCSDof, :]

        M00 = M0[:, isFreeDof]
        M01 = M0[:, isCSDof] @ self.G
        M11 = M1[:, isCSDof] @ self.G
        M11 = self.G.T @ M11 

        #self._check_symmetry_spd(S11, M11)

        S11 = (S11 + S11.T)/2.0  # ensure symmetry
        M11 = (M11 + M11.T)/2.0  # ensure symmetry

        return [[S00, S01], [S01.T, S11]], [[M00, M01], [M01.T, M11]] 

    def shaft_linear_system(self):
        """
        Load the shaft system from the .mat file.
        Notes:
            1. 这里假设轴系耦合节点的排序与整体轴系节点的排序一致。
        """
        from scipy.io import loadmat
        from scipy.sparse import csr_matrix

        shaft_system = loadmat(self.options['shaft_system_file'])
        S = shaft_system['stiffness_total_system']
        M = shaft_system['mass_total_system']
        layout = shaft_system['component_bearing_layout'][:, 1:]
        section = shaft_system['order_section_shaft']

        self.logger.info(
                f"Sucsess load shaft system from {self.options['shaft_system_file']}\n"
                f"shaft system stiffness matrix: {S.shape}\n"
                f"shaft system mass matrix: {M.shape}")
        # the total number of nodes in the shaft system
        NN = int(section[1].sum())  

        # offset for the nodes in each shaft section 
        offset = [0] + [int(a) for a in bm.cumsum(section[1])]  
        self.logger.info(
                f"Number of nodes in the shaft system: {NN}\n"
                f"Offset for the nodes: {offset}")

        nidmap = self.mesh.data['nidmap']
        # the mapping from the original node index to the natural number index
        imap = {f"{a[0]:.1f}" : a[1] for a in zip(section[0], range(section.shape[1]))}
        self.logger.info(f"Node map: {imap}")
        # the flag of the nodes in the shaft system which coupling with the
        # gearbox shell model
        isCNode = bm.zeros(NN, dtype=bm.bool)
        for i, j in layout[:, :2]:
            idx = offset[imap[f"{i:.1f}"]] + int(j) - 1
            isCNode[idx] = True
        self.logger.info(f"Number of coupling nodes: {bm.sum(isCNode)}")

        isCDof = bm.repeat(isCNode, 6)

        # 构造轴系的分块矩阵
        S0 = S[bm.logical_not(isCDof), :]
        S1 = S[isCDof, :]

        S00 = S0[:, bm.logical_not(isCDof)]
        S01 = S0[:, isCDof]
        S11 = S1[:, isCDof]

        M0 = M[bm.logical_not(isCDof), :]
        M1 = M[isCDof, :]

        M00 = M0[:, bm.logical_not(isCDof)]
        M01 = M0[:, isCDof]
        M11 = M1[:, isCDof]
        self.logger.info(
                f"shaft system stiffness matrix: {S00.shape}, {S01.shape}, {S11.shape}\n"
                f"shaft system mass matrix: {M00.shape}, {M01.shape}, {M11.shape}")

        # the flag of the coupling nodes in the gearbox shell model
        cnode = nidmap[layout[:, -1].astype(self.itype)]
        isCNode = bm.zeros(self.mesh.number_of_nodes(), dtype=bm.bool)
        isCNode = bm.set_at(isCNode, cnode, True)
        self.mesh.data.add_node_data('isCNode', isCNode)

        # resort the columns of S01, S11, M01, M11 according to the cnode order
        re = bm.argsort(cnode)
        idx = bm.arange(6 * cnode.shape[0], dtype=self.itype).reshape(-1, 6)
        idx = idx[re, :].flatten()

        S01 = csr_matrix(S01[:, idx])
        S11 = csr_matrix(S11[idx, :][:, idx])
        M01 = csr_matrix(M01[:, idx])
        M11 = csr_matrix(M11[idx, :][:, idx])

        return [[S00, S01], [S01.T, S11]], [[M00, M01], [M01.T, M11]]


    @variantmethod('direct')
    def set_ksp(self, ksp):
        """
        Set the KSP solver for the eigenvalue problem using direct method.
        """
        ksp.setType('preonly')
        pc = ksp.getPC()
        pc.setType('lu')
        try:
            pc.setFactorSolverType('mumps')
        except Exception:
            pc.setFactorSolverType('superlu')

    @set_ksp.register('iterative')
    def set_ksp(self, ksp):
        """
        Set the KSP solver for the eigenvalue problem using iterative method.
        """
        ksp.setTolerances(rtol=1e-8, atol=1e-08, max_it=1000)
        ksp.setType('gmres')  # 或 'gmres'
        def my_ksp_monitor(ksp, its, rnorm):
            print(f"KSP iter {its}, residual norm = {rnorm}")
        ksp.setMonitor(my_ksp_monitor)
        pc = ksp.getPC()
        pc.setType('gamg')  # 或 'gamg' 若使用 AMG



    @variantmethod('slepc')
    def solve(self, which: str ='SM', fname: str="eigen.vtu") -> None:
        """
        Solve the eigenvalue problem using SLEPc.
        """
        from petsc4py import PETSc
        from slepc4py import SLEPc
        from scipy.sparse import bmat

        S0, M0 = self.shaft_linear_system()
        #self._check_symmetry_spd(bmat(S0).tocsr(), bmat(M0).tocsr())

        self.rbe2_matrix()
        S1, M1 = self.box_linear_system()
        self._check_symmetry_spd(bmat(S1).tocsr(), bmat(M1).tocsr())

        self._check_G(S0, M0, S1, M1)
        self._check_flags()


        self.mesh.to_vtk(fname=fname)


        N0 = S0[0][0].shape[0]  # number of free dofs in the shaft system
        N1 = S1[0][0].shape[0]  # number of free dofs in the gearbox shell model
        N2 = S0[1][1].shape[0]  # number of coupling dofs

        S = bmat([[S0[0][0], None, S0[0][1]],
                  [None, S1[0][0], S1[0][1]],
                  [S0[1][0], S1[1][0], S0[1][1] + S1[1][1]]]).tocsr()
        
        M = bmat([[M0[0][0], None, M0[0][1]],
                  [None, M1[0][0], M1[0][1]],
                  [M0[1][0], M1[1][0], M0[1][1] + M1[1][1]]]).tocsr()


        self._check_system_singular(S, M, 0.0)


        self.logger.info(f"Global system: {S.shape}, {M.shape}")

        PS = PETSc.Mat().createAIJ(
                size=S.shape, 
                csr=(S.indptr, S.indices, S.data))
        PS.assemble()
        PM = PETSc.Mat().createAIJ(
                size=M.shape, 
                csr=(M.indptr, M.indices, M.data))
        PM.assemble()

        eps = SLEPc.EPS().create()
        eps.setOperators(PS, PM)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)

        sigma = 0.0  # 更贴近你的目标最小特征值（基于经验）
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
        eps.setTarget(sigma)  # ← 显式设置目标

        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(sigma)

        ksp = st.getKSP()
        self.set_ksp['direct'](ksp)

        vec = PS.getVecRight()
        vec.setRandom()
        eps.setInitialSpace([vec])

        k = self.options.get('neigen', 2)
        eps.setDimensions(nev=k, ncv=min(8*k, 200))
        eps.setTolerances(tol=1e-6, max_it=10000)
        eps.solve()

        eigvals = []
        eigvecs = []

        vr, vi = eps.getOperators()[0].getVecs()
        print(f"Number of eigenvalues converged: {eps.getConverged()}")
        for i in range(min(k, eps.getConverged())):
            val = eps.getEigenpair(i, vr, vi)
            eigvals.append(val.real)
            eigvecs.append(vr.getArray().copy())
        eigvals = bm.array(eigvals)
        self.logger.info(f"Eigenvalues: {eigvals}")

        # transform the eigenvectors to the original node index
        NN = self.mesh.number_of_nodes()
        isFreeNode = self.mesh.data.get_node_data('isFreeNode')
        isFreeDof = bm.repeat(isFreeNode, 3)
        isCSNode = self.mesh.data.get_node_data('isCSNode')
        isCSDof = bm.repeat(isCSNode, 3)
        vec = [] 
        for i, val in enumerate(eigvecs):
            phi = bm.zeros((NN * 3,), dtype=self.ftype)
            idx, = bm.where(isFreeDof)
            phi = bm.set_at(phi, idx, val[N0:N0 + N1])
            # transform the dof values of coupling nodes to surface nodes
            # constrained by the coupling nodes.
            idx, = bm.where(isCSDof)
            phi = bm.set_at(phi, idx, self.G @ val[N0 + N1:])
            phi = phi.reshape((NN, 3))
            # put into the mesh node data 
            self.mesh.nodedata[f'eigenvalue-{i}-{eigvals[i]:0.5e}'] = phi

        # check the correctness of the eigenvectors
        ne = len(eigvals)
        for i in range(len(eigvecs)):
            for j in range(i+1):
                v1 = bm.array(eigvecs[i], dtype=self.ftype)
                v2 = bm.array(eigvecs[j], dtype=self.ftype)
                print(f"Eigenvector {i} and {j} dot product: {bm.dot(v1, M @ v2)}")

        self.mesh.to_vtk(fname=fname)

    def post_process(self, fname: str="eigen.vtu") -> None:
        """
        Post-process the results, such as saving to files or visualizing.
        """

        self.mesh.to_vtk(fname=fname)



    def _check_flags(self):
        """
        """
        NN = self.mesh.number_of_nodes()
        isRNode = self.mesh.data.get_node_data('isRNode') # reference node
        isCSNode = self.mesh.data.get_node_data('isCSNode') # surface coupling node
        isFNode = self.mesh.data.get_node_data('isFNode') # fixed node

        assert isRNode.shape[0] == NN and isCSNode.shape[0] == NN and isFNode.shape[0] == NN

        # 互斥（参考节点不能同时是表面耦合/固定）
        assert not bm.any(isRNode & isCSNode)
        assert not bm.any(isRNode & isFNode)

        isFreeNode = ~(isRNode | isFNode | isCSNode)
        self.logger.info(
            f"counts - Reference:{bm.sum(isRNode)}, CSurface:{bm.sum(isCSNode)}, Fiexed:{bm.sum(isFNode)}, "
            f"Free:{bm.sum(isFreeNode)}, Total:{NN}")
        
    def _check_G(self, S0, M0, S1, M1):
        """
        """
        # S0 来自 shaft_linear_system，S1 来自 box_linear_system
        # 块意义：[[.., ..],[.., S11]] 中右下角维度就是耦合自由度数
        n2_shaft = S0[1][1].shape[0]  # 应当等于 6 * NC
        n2_box   = S1[1][1].shape[0]  # 也应是 6 * NC
        assert n2_shaft == n2_box, f"耦合块维度不一致: shaft {n2_shaft} vs box {n2_box}"

        # 检查 G 形状与 6*NC
        assert self.G.shape[1] == n2_box, f"G 列数 {self.G.shape[1]} != 耦合自由度 {n2_box}"

        # G 是否明显降秩（抽样）
        import numpy as np
        from scipy.sparse import csc_matrix
        col_norm = np.sqrt(self.G.power(2).sum(axis=0)).A.ravel()
        assert np.all(col_norm > 0), "G 存在全零列（降秩）"


    def _check_symmetry_spd(self, S, M):
        """
        """
        import numpy as np
        ST = S - S.T
        MT = M - M.T
        assert ST.nnz == 0 or np.max(np.abs(ST.data)) < 1e-10, f"S 非对称, {ST.nnz}, {np.max(np.abs(ST.data))}"
        assert MT.nnz == 0 or np.max(np.abs(MT.data)) < 1e-10, f"M 非对称, {MT.nnz}, {np.max(np.abs(MT.data))}"

        # M 不应有零行/零对角
        import numpy as np
        zrow = np.where(M.getnnz(axis=1) == 0)[0]
        assert zrow.size == 0, f"M 存在零行: {zrow[:10]}"
        d = M.diagonal()
        assert np.all(d > 0), "M 对角存在非正项（可能含零质量自由度）"


    def _check_system_singular(self, S, M, sigma=0.0):
        """
        """
        from scipy.sparse.linalg import spsolve, LinearOperator
        import numpy as np

        A = S - sigma * M

        rhs = np.random.rand(A.shape[0])
        x = spsolve(A, rhs)  # 若崩/警告或残差巨大 => (A-σM) 奇异或预处理不当
        res = np.linalg.norm(A @ x - rhs) / np.linalg.norm(rhs)
        self.logger.info(f"Test (S - σM) solve residual ~ {res: .2e}")

