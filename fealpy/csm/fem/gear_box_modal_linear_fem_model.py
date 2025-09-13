
from typing import Any, Optional, Union, Callable, Tuple, Sequence, Literal
import os

from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
from scipy.sparse import bmat

from scipy.io import loadmat
from scipy.sparse import csr_matrix

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


    @variantmethod('all')
    def construct_system(self):
        """
        Construct the global linear system for the gearbox modal analysis.
        """

        S0, M0 = self.shaft_linear_system()
        self.rbe2_matrix()
        S1, M1 = self.box_linear_system()

        N0 = S0[0][0].shape[0]  # number of free dofs in the shaft system
        N1 = S1[0][0].shape[0]  # number of free dofs in the gearbox shell model
        N2 = S0[1][1].shape[0]  # number of coupling dofs

        S = bmat([[S0[0][0],     None,            S0[0][1]],
                  [    None, S1[0][0],            S1[0][1]],
                  [S0[1][0], S1[1][0], S0[1][1] + S1[1][1]]]).tocsr()
        
        M = bmat([[M0[0][0],     None,            M0[0][1]],
                  [    None, M1[0][0],            M1[0][1]],
                  [M0[1][0], M1[1][0], M0[1][1] + M1[1][1]]]).tocsr()

        self.S = S
        self.M = M


        self.logger.info(f"Global system: {S.shape}, {M.shape}")

        PS = PETSc.Mat().createAIJ(
                size=S.shape, 
                csr=(S.indptr, S.indices, S.data))
        PS.assemble()
        PM = PETSc.Mat().createAIJ(
                size=M.shape, 
                csr=(M.indptr, M.indices, M.data))
        PM.assemble()

        NN0 = N0//6
        NN1 = N1//3
        NN2 = N2//6
        dof_nodes = bm.concat((bm.arange(N0)//6, bm.arange(N1)//3 + NN0, bm.arange(N2)//6 + NN0 + NN1))
        dof_comps = bm.concat((bm.arange(N0)%6, bm.arange(N1)%3, bm.arange(N2)%6))

        return PS, PM, N0, N1, N2, dof_nodes, dof_comps

    @construct_system.register('box')
    def construct_system(self):
        """
        Construct the global linear system for the gearbox modal analysis.
        """

        S0, M0 = self.shaft_linear_system()
        self.rbe2_matrix()
        S1, M1 = self.box_linear_system()

        N0 = 0  # without shaft system
        N1 = S1[0][0].shape[0]  # number of free dofs in the gearbox shell model
        N2 = S1[1][1].shape[0]  # number of coupling dofs

        S = bmat(S1, format='csr')
        M = bmat(M1, format='csr')

        self.S = S
        self.M = M


        self.logger.info(f"Global system: {S.shape}, {M.shape}")

        PS = PETSc.Mat().createAIJ(
                size=S.shape, 
                csr=(S.indptr, S.indices, S.data))
        PS.assemble()
        PM = PETSc.Mat().createAIJ(
                size=M.shape, 
                csr=(M.indptr, M.indices, M.data))
        PM.assemble()

        NN0 = N0//6
        NN1 = N1//3
        NN2 = N2//6
        dof_nodes = bm.concat((bm.arange(N0)//6, bm.arange(N1)//3 + NN0, bm.arange(N2)//6 + NN0 + NN1))
        dof_comps = bm.concat((bm.arange(N0)%6, bm.arange(N1)%3, bm.arange(N2)%6))

        return PS, PM, N0, N1, N2, dof_nodes, dof_comps



    @variantmethod('slepc')
    def solve(self, PS, PM) -> None:
        """
        Solve the eigenvalue problem using SLEPc.
        """
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
        return eps


    def post_process(self, eps, N0, N1, N2) -> None:
        """
        Post-process the eigenvalue problem solution.
        """
        k = self.options.get('neigen', 2)

        eigvals = []
        eigvecs = []

        vr, vi = eps.getOperators()[0].getVecs()
        print(f"Number of eigenvalues converged: {eps.getConverged()}")
        for i in range(min(k, eps.getConverged())):
            val = eps.getEigenpair(i, vr, vi)
            eigvals.append(val.real)
            eigvecs.append(vr.getArray().copy())
        eigvals = bm.array(eigvals)
        freqs = bm.sqrt(eigvals) / (2 * bm.pi)
        self.logger.info(f"Eigenvalues: {eigvals}")
        self.logger.info(f"Frequencies: {freqs}")

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
                print(f"Eigenvector {i} and {j} dot product: {bm.dot(v1, self.M @ v2)}")

    def to_mtx(self,
               fname: str = "system.mtx",
               S: Union[PETSc.Mat, "sp.csr_matrix"] = None,
               M: Optional[Union[PETSc.Mat, "sp.csr_matrix"]] = None,
               sigma: Optional[complex] = None) -> None:
        """
        Export the final discrete system to Matrix Market (.mtx) files.

        This method supports three modes:
          1) Only stiffness: write S to <fname>.
          2) Stiffness + Mass: write S to <stem>_S.mtx and M to <stem>_M.mtx.
          3) Shifted operator: form A = S - sigma * M and write A to <fname>.

        Parameters
            fname (str, optional, default='system.mtx'):
                Output file name or stem. If both S and M are exported separately,
                '<stem>_S.mtx' and '<stem>_M.mtx' will be used.
            S (petsc4py.PETSc.Mat | scipy.sparse.csr_matrix):
                Stiffness matrix.
            M (petsc4py.PETSc.Mat | scipy.sparse.csr_matrix | None, optional):
                Mass matrix. Required if `sigma` is provided.
            sigma (complex | float | None, optional):
                Shift parameter. If provided, exports A = S - sigma * M.

        Notes
            - Parallel ASCII (.mtx) I/O is functional but slow for large problems.
              For big runs, prefer: (i) write PETSc binary in parallel, then
              (ii) convert to .mtx on a single rank.
            - Matrix Market stores a single matrix per file; when `S` and `M`
              are requested together, two files are produced.

        Returns
            None
        """
        if S is None:
            raise ValueError("`S` must be provided.")

        comm = PETSc.COMM_WORLD
        rank = comm.getRank()

        # --- helpers ---------------------------------------------------------
        def _as_petsc_mat(X: Union[PETSc.Mat, "sp.csr_matrix"]) -> PETSc.Mat:
            if isinstance(X, PETSc.Mat):
                return X
            if _HAS_SCIPY and sp.isspmatrix_csr(X):
                A = PETSc.Mat().createAIJ(size=X.shape, csr=(X.indptr, X.indices, X.data), comm=comm)
                A.assemble()
                return A
            raise TypeError("Matrix must be PETSc.Mat or SciPy CSR matrix.")



        def _write_mtx(
            mat: PETSc.Mat,
            path: str,
            *,
            sig_digits: int = 17,                      # 有效数字位数（17~20 常用）
            symmetric: bool = False,                   # 是否只写一个三角，并在头部标记 symmetric
            triangular: Literal["lower","upper"] = "lower",
            gzip_output: bool = False,                 # True 则输出 .mtx.gz
            comm = PETSc.COMM_WORLD,
        ) -> None:
            """
            Write PETSc.Mat to Matrix Market (.mtx) with controllable precision (MPI-safe).

            Header:
              %%MatrixMarket matrix coordinate {real|complex} {general|symmetric}
              M N NNZ
            Body (1-based):
              i  j  value                  (real)
              i  j  real_value  imag_value (complex)

            Parameters:
                mat : Assembled PETSc.Mat (SeqAIJ / MPIAIJ / ...).
                path : Output file path ('.mtx' recommended; '.gz' will be added if gzip_output=True).
                sig_digits : Significant digits; 17 generally round-trips IEEE-754 double.
                symmetric : If True, write only one triangular part and mark header 'symmetric'.
                triangular : 'lower' or 'upper' when symmetric=True.
                gzip_output : If True, produce gzip file.
                comm : PETSc communicator (will be bridged to mpi4py).
            """
            # ---- mpi4py 通信器 ----
            mpicomm = comm.tompi4py() if hasattr(comm, "tompi4py") else MPI.COMM_WORLD
            rank = mpicomm.Get_rank()
            size = mpicomm.Get_size()

            self.logger.info(f"[to_mtx] rank {rank}/{size} writing to {path}")

            # ---- 基本信息 ----
            M, N = mat.getSize()
            istart, iend = mat.getOwnershipRange()

            # 有效数字 -> 科学计数法小数位（'%.Ne' 中 N = sig_digits - 1）
            dec = max(0, sig_digits - 1)
            fmt = f"{{0:.{dec}e}}"

            # 分片文件名（基于最终输出名构造）
            out_path = path
            if gzip_output and not out_path.endswith(".gz"):
                out_path += ".gz"
            base = os.path.basename(out_path)
            part = f".{base}.part{rank:05d}"

            # ---- 本地写分片 ----
            local_nnz = 0
            is_complex_local = False

            with open(part, "wt", encoding="utf-8") as f:
                for i in range(istart, iend):
                    cols, vals = mat.getRow(i)   # petsc4py：无需 restoreRow
                    for j, v in zip(cols, vals):
                        if symmetric:
                            if triangular == "lower" and j > i:
                                continue
                            if triangular == "upper" and j < i:
                                continue
                        ii = i + 1
                        jj = j + 1
                        if isinstance(v, complex):
                            is_complex_local = True
                            f.write(f"{ii} {jj} {fmt.format(v.real)} {fmt.format(v.imag)}\n")
                        else:
                            f.write(f"{ii} {jj} {fmt.format(float(v))}\n")
                        local_nnz += 1

            # ---- 归约统计 ----
            nnz = mpicomm.allreduce(local_nnz, op=MPI.SUM)
            any_complex = mpicomm.allreduce(1 if is_complex_local else 0, op=MPI.SUM) > 0

            # ---- rank 0 写头并串接 ----
            if rank == 0:
                field = "complex" if any_complex else "real"
                symmetry = "symmetric" if symmetric else "general"
                header = [
                    f"%%MatrixMarket matrix coordinate {field} {symmetry}",
                    f"{M} {N} {nnz}",
                ]

                opener = (lambda p, m: gzip.open(p, m, encoding="utf-8")) if gzip_output else (lambda p, m: open(p, m, encoding="utf-8"))
                with opener(out_path, "wt") as out:
                    out.write("\n".join(header) + "\n")
                    for r in range(size):
                        rp = f".{base}.part{r:05d}"
                        with open(rp, "rt", encoding="utf-8") as pf:
                            for line in pf:
                                out.write(line)

                # 清理分片
                for r in range(size):
                    os.remove(f".{base}.part{r:05d}")

            mpicomm.Barrier()




        # --- normalize inputs ------------------------------------------------
        S_petsc = _as_petsc_mat(S)
        S_petsc.assemble()

        if sigma is not None:
            if M is None:
                raise ValueError("`sigma` is provided but `M` is None; need M to form A = S - sigma*M.")
            M_petsc = _as_petsc_mat(M)
            M_petsc.assemble()

            # A = S - sigma * M  （不修改原 S）
            A = S_petsc.copy()
            # 允许不同稀疏模式，稳妥一些
            A.axpy(-sigma, M_petsc, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            A.assemble()

            # 输出单个矩阵
            out_path = fname if fname.endswith(".mtx") else (fname + ".mtx")
            if rank == 0:
                print(f"[to_mtx] Writing A = S - sigma*M to {out_path}")
            _write_mtx(A, out_path)
            A.destroy()
            return

        # 若不需要组合，判断是否同时输出 S 与 M
        if M is None:
            # 仅输出 S
            out_path = fname if fname.endswith(".mtx") else (fname + ".mtx")
            if rank == 0:
                print(f"[to_mtx] Writing S to {out_path}")
            _write_mtx(S_petsc, out_path)
            return
        else:
            # 分别输出 S 与 M
            stem, ext = os.path.splitext(fname)
            if ext and ext != ".mtx":
                # 若给了其它后缀，仍按 _S/_M.mtx 命名
                stem = stem + ext  # 保留用户自定义的“后缀感”作为 stem
            S_path = f"{stem}_S.mtx"
            M_path = f"{stem}_M.mtx"
            M_petsc = _as_petsc_mat(M)
            M_petsc.assemble()

            if rank == 0:
                print(f"[to_mtx] Writing S to {S_path} and M to {M_path}")
            _write_mtx(S_petsc, S_path)
            _write_mtx(M_petsc, M_path)
            return


    def to_abaqus(
        self,
        fname_stem: str = "system",
        S: PETSc.Mat = None,
        M: Optional[PETSc.Mat] = None,
        *,
        dof_mapper: Optional[Callable[[int], Tuple[int, int]]] = None,
        dof_nodes: Optional[Sequence[int]] = None,
        dof_comps: Optional[Sequence[int]] = None,
        symmetric: bool = True,
        triangular: str = "lower",  # 'lower' | 'upper' | 'full'
        float_fmt: str = "%.16e",
        gzip_output: bool = False,
        temp_dir: str = ".",
    ) -> None:
        """
        Export PETSc matrices to Abaqus matrix-input text files.

        Each nonzero entry is written as a 5-tuple line:
        "row_node, row_dof, col_node, col_dof, value" (1-based node & DOF IDs).

        This method supports exporting:
          - Stiffness matrix S -> <stem>_K.mtx
          - Optional mass matrix M -> <stem>_M.mtx
          - Symmetric matrices with triangular filtering ('lower' or 'upper'), or full output

        Parameters
            fname_stem (str, optional, default='system'):
                Output file stem. Produces '<stem>_K.mtx' for S and, if provided,
                '<stem>_M.mtx' for M. If `gzip_output=True`, '.gz' is appended.
            S (petsc4py.PETSc.Mat):
                Stiffness matrix (AIJ/SeqAIJ/MPIAIJ).
            M (petsc4py.PETSc.Mat | None, optional):
                Mass matrix. If provided, a second file for M is written.
            dof_mapper (callable | None, optional):
                Callable that maps a global DOF index (0-based) to
                (node_id, comp_id). Return values must be 1-based as expected by Abaqus.
                Provide this OR both `dof_nodes` and `dof_comps`.
            dof_nodes (Sequence[int] | None, optional):
                Alternative to `dof_mapper`: node ID per global DOF (length = ndof).
                If 0-based, it will be automatically shifted to 1-based.
            dof_comps (Sequence[int] | None, optional):
                Alternative to `dof_mapper`: component ID per global DOF (length = ndof).
                Typically 1..6 for structural DOFs. If 0-based, it will be shifted to 1-based.
            symmetric (bool, optional, default=True):
                If True, treat matrices as symmetric and write only the chosen triangular part.
                If False, write all entries ('full').
            triangular (str, optional, default='lower'):
                Triangular part to export when `symmetric=True`. One of
                {'lower', 'upper', 'full'}. Ignored if `symmetric=False`.
            float_fmt (str, optional, default='%.16e'):
                Floating-point format for values.
            gzip_output (bool, optional, default=False):
                If True, write gzip-compressed files ('.gz').
            temp_dir (str, optional, default='.'):
                Directory for rank-local temporary chunk files prior to concatenation.

        Notes
            - Node and DOF IDs in the written files must be 1-based to conform to Abaqus.
              If your internal IDs are 0-based, provide a mapper that shifts them or rely
              on the auto-shift for `dof_nodes`/`dof_comps`.
            - For symmetric matrices, writing only the lower (or upper) triangular part
              significantly reduces file size. In Abaqus, use:
              `*MATRIX INPUT, TYPE=SYMMETRIC, INPUT=<file>`.
            - MPI text I/O can be heavy. This method writes per-rank chunks and concatenates
              them on rank 0 to avoid interleaving; consider disabling compression during
              development for faster turnaround.

        Returns
            None
        """
        if S is None:
            raise ValueError("S (stiffness) must be provided.")

        comm = PETSc.COMM_WORLD
        rank = comm.getRank()
        size = comm.getSize()

        # ---- build mapper ----
        if dof_mapper is None:
            if dof_nodes is None or dof_comps is None:
                raise ValueError("Provide `dof_mapper` or both `dof_nodes` and `dof_comps`.")
            if len(dof_nodes) != len(dof_comps):
                raise ValueError("`dof_nodes` and `dof_comps` length mismatch.")
            node_shift = 1 if min(dof_nodes) == 0 else 0
            comp_shift = 1 if min(dof_comps) == 0 else 0

            def dof_mapper(i: int) -> Tuple[int, int]:
                return dof_nodes[i] + node_shift, dof_comps[i] + comp_shift

        if symmetric is False:
            triangular = "full"
        else:
            triangular = triangular.lower()
            if triangular not in ("lower", "upper", "full"):
                raise ValueError("`triangular` must be 'lower'|'upper'|'full'.")

        def _export_one(mat: PETSc.Mat, out_path: str) -> None:
            istart, iend = mat.getOwnershipRange()
            base = os.path.basename(out_path)
            tmp = os.path.join(temp_dir, f".{base}.part{rank:05d}")

            with open(tmp, "wt", encoding="utf-8") as f:
                for i in range(istart, iend):
                    cols, vals = mat.getRow(i)
                    for j, v in zip(cols, vals):
                        if symmetric:
                            if triangular == "lower" and j > i:
                                continue
                            if triangular == "upper" and j < i:
                                continue
                        ni, di = dof_mapper(i)
                        nj, dj = dof_mapper(j)
                        f.write(f"{ni}, {di}, {nj}, {dj}, {float_fmt % float(v)}\n")

            comm.Barrier()
            if rank == 0:
                if gzip_output:
                    with gzip.open(out_path, "wt", encoding="utf-8") as g:
                        for r in range(size):
                            part = os.path.join(temp_dir, f".{base}.part{r:05d}")
                            with open(part, "rt", encoding="utf-8") as pf:
                                for line in pf:
                                    g.write(line)
                            os.remove(part)
                else:
                    with open(out_path, "wt", encoding="utf-8") as out:
                        for r in range(size):
                            part = os.path.join(temp_dir, f".{base}.part{r:05d}")
                            with open(part, "rt", encoding="utf-8") as pf:
                                for line in pf:
                                    out.write(line)
                            os.remove(part)
            comm.Barrier()

        # ---- write K (S) ----
        k_path = f"{fname_stem}_K.mtx"
        if gzip_output:
            k_path += ".gz"
        _export_one(S, k_path)
        if rank == 0:
            print(f"[to_abaqus] Wrote stiffness to {k_path} "
                  f"(symmetric={symmetric}, triangular='{triangular}')")

        # ---- write M (optional) ----
        if M is not None:
            m_path = f"{fname_stem}_M.mtx"
            if gzip_output:
                m_path += ".gz"
            _export_one(M, m_path)
            if rank == 0:
                print(f"[to_abaqus] Wrote mass to {m_path}")

    def write_abaqus_frequency_input(
        self,
        inp_path: str = "matrix_modes.inp",
        *,
        fname_stem: str = "system",
        k_file: Optional[str] = None,
        m_file: Optional[str] = None,
        symmetric: bool = True,
        eigensolver: str = "LANCZOS",          # or "AMS"
        modes: int = 10,
        lower: Optional[float] = None,         # lower frequency bound (Hz)
        upper: Optional[float] = None,         # upper frequency bound (Hz)
        shift: Optional[float] = None,         # frequency shift (Hz)
        normalization: str = "MASS",           # MASS | DISPLACEMENT | DEFAULT
        heading: Optional[str] = None,
        boundary_lines: Optional[Sequence[str]] = None,  # raw '*BOUNDARY' lines (optional)
    ) -> None:
        """
        Generate a minimal Abaqus input deck for eigenfrequency extraction using matrix input.

        The deck references stiffness/mass matrices written in Abaqus "matrix input text" format,
        then assembles and runs a *FREQUENCY step (Lanczos/AMS).

        Parameters:
            inp_path (str, optional, default='matrix_modes.inp'):
                Output path for the generated Abaqus .inp file.
            fname_stem (str, optional, default='system'):
                Stem used when `k_file` or `m_file` are not provided. Defaults to
                '<stem>_K.mtx' and '<stem>_M.mtx'.
            k_file (str | None, optional):
                Path to the stiffness matrix text file. If None, uses '<stem>_K.mtx'.
            m_file (str | None, optional):
                Path to the mass matrix text file. If None, uses '<stem>_M.mtx'.
                If truly no mass is available, you may set this to an empty string or
                omit *MASS in assembly (see Notes).
            symmetric (bool, optional, default=True):
                Whether matrices are symmetric. Controls TYPE=SYMMETRIC vs TYPE=UNSYMMETRIC
                in *MATRIX INPUT.
            eigensolver (str, optional, default='LANCZOS'):
                Eigensolver choice for *FREQUENCY. Typical options: 'LANCZOS', 'AMS'.
            modes (int, optional, default=10):
                Number of eigenmodes to extract.
            lower (float | None, optional):
                Lower frequency bound (Hz) for *FREQUENCY data line (optional).
            upper (float | None, optional):
                Upper frequency bound (Hz) for *FREQUENCY data line (optional).
            shift (float | None, optional):
                Frequency shift (Hz) for *FREQUENCY data line (optional).
            normalization (str, optional, default='MASS'):
                NORMALIZATION keyword for *FREQUENCY, e.g., 'MASS' or 'DISPLACEMENT'.
            heading (str | None, optional):
                Optional text placed under *HEADING.
            boundary_lines (Sequence[str] | None, optional):
                Plain lines to be placed under a '*BOUNDARY' block (optional).
                Each item should be a complete Abaqus boundary line, e.g., '123, 1, 3'.

        Notes:
            - The referenced matrix files must be Abaqus "matrix input text" format with
              1-based node/DOF IDs and five-column lines: row_node, row_dof, col_node, col_dof, value.
            - If `symmetric=True`, the files should contain only one triangular part; use
              '*MATRIX INPUT, TYPE=SYMMETRIC'. Otherwise use TYPE=UNSYMMETRIC and export full entries.
            - Abaqus does not support gzip-compressed INPUT files. Ensure `to_abaqus(..., gzip_output=False)`.
            - If no mass matrix is available, the input deck will assemble only STIFFNESS=K. Abaqus
              *FREQUENCY typically expects both K and M for physical units; use with care.
            - All paths in `INPUT=` are interpreted relative to the working directory of the job.

        Returns
            None
        """
        type_str = "SYMMETRIC" if symmetric else "UNSYMMETRIC"
        k_path = k_file if k_file is not None else f"{fname_stem}_K.mtx"
        m_path = m_file if m_file is not None else f"{fname_stem}_M.mtx"

        # Build *FREQUENCY data line: n_modes, lower, upper, shift (only include numbers that are provided)
        freq_parts = [str(int(modes))]
        for x in (lower, upper, shift):
            if x is not None:
                freq_parts.append(f"{float(x)}")
        freq_line = ", ".join(freq_parts) + ","

        lines = []
        lines.append("*HEADING")
        lines.append(heading.strip() if heading else "**Matrix-eigenvalue job (auto-generated).")

        # Matrix inputs
        lines.append(f"*MATRIX INPUT, NAME=K, TYPE={type_str}, INPUT={k_path}")
        if m_path and m_path.strip():
            lines.append(f"*MATRIX INPUT, NAME=M, TYPE={type_str}, INPUT={m_path}")

        # Optional boundary block (works even if no node coordinates are defined)
        if boundary_lines:
            lines.append("*BOUNDARY")
            lines.extend(boundary_lines)

        # Assemble usage model
        if m_path and m_path.strip():
            lines.append("*MATRIX ASSEMBLE, STIFFNESS=K, MASS=M")
        else:
            lines.append("*MATRIX ASSEMBLE, STIFFNESS=K")

        # Frequency extraction step
        norm_kw = f", NORMALIZATION={normalization.upper()}" if normalization else ""
        lines.append("*STEP, NAME=MODES")
        lines.append(f"*FREQUENCY, EIGENSOLVER={eigensolver.upper()}{norm_kw}")
        lines.append(freq_line)
        lines.append("*END STEP")

        # Write file
        os.makedirs(os.path.dirname(inp_path) or ".", exist_ok=True)
        with open(inp_path, "wt", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        # Optional: small log on rank 0
        comm = PETSc.COMM_WORLD
        if comm.getRank() == 0:
            print(f"[write_abaqus_frequency_input] Wrote input deck to {inp_path}")
            print(f"  - K: {k_path}")
            if m_path and m_path.strip():
                print(f"  - M: {m_path}")
            print(f"  - Eigensolver: {eigensolver}, Modes: {modes}, Symmetric: {symmetric}")
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

