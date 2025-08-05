
from typing import Any, Optional, Union
from scipy.sparse.linalg import eigsh

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

class GearBoxModalLFEMModel(ComputationalModel):

    def __init__(self, options):
        self.options = options
        super().__init__(
            pbar_log=options['pbar_log'],
            log_level=options['log_level']
        )
        self.set_pde(options['pde'])
        GD = self.pde.geo_dimension()

        mesh = self.pde.init_mesh()
        self.set_mesh(mesh)

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
        Set node flags for the gear box model.

        the node flags is the base for the dofs flag
        """
        NN = self.mesh.number_of_nodes()
        redges, rnodes = self.mesh.data.get_rbe2_edge()

        # 耦合节点
        isCNode = self.mesh.data.get_node_data('isCNode')

        # 参考节点, 注意部分参考节点可能不是耦合节点
        isRNode = bm.zeros(NN, dtype=bm.bool)
        isRNode = bm.set_at(isRNode, rnodes, True)

        # 齿轮箱节点
        isGBNode = bm.logical_not(isRNode)

        # 参考节点约束的齿轮箱面节点
        isSNode = bm.zeros(NN, dtype=bm.bool)
        isSNode = bm.set_at(isSNode, redges[:, 1], True)

        # 实际耦合节点对应的齿轮箱面节点
        isRBE2 = isCNode[redges[:, 0]]
        isCSNode = bm.zeros(NN, dtype=bm.bool)
        isCSNode = bm.set_at(isCSNode, redges[isRBE2, 1], True)
        
        # 齿轮箱固定节点, 注意这里只是临时处理 
        name = self.mesh.data['boundary_conditions'][0][0]
        nset = self.mesh.data.get_node_set(name)
        isFNode = bm.zeros(NN, dtype=bm.bool)
        isFNode[nset] = True

        # 记录节点的标志
        self.mesh.data.add_node_data('isRNode', isRNode)
        self.mesh.data.add_node_data('isGBNode', isGBNode)
        self.mesh.data.add_node_data('isSNode', isSNode)
        self.mesh.data.add_node_data('isFNode', isFNode)


        ridx = bm.where(isCNode)[0]
        sidx = bm.where(isCSNode)[0]
        rmap = bm.zeros(NN, dtype=bm.int32)
        smap = bm.zeros(NN, dtype=bm.int32)
        rmap = bm.set_at(rmap, ridx, bm.arange(ridx.shape[0], dtype=bm.int32))
        smap = bm.set_at(smap, sidx, bm.arange(sidx.shape[0], dtype=bm.int32))

        I = smap[redges[isRBE2, 1]]
        J = rmap[redges[isRBE2, 0]]
        

        node = self.mesh.entity('node')
        # 依赖点指向参考点 
        v = node[redges[redges[isRBE2, 0]] - node[redges[isRBE2, 1]]

        NS = isCSNode.sum() # number of surface nodes
        NR = isCNode.sum() # number of reference nodes

        self.logger.info(f"RBE2 matrix: {NS} surface nodes, {NR} reference nodes")

        G = coo_matrix((3*NS, 6*NR), dtype=bm.float64)
        ones = bm.ones(NS, dtype=bm.float64)

        G += coo_matrix((    ones, (3*I+0, 6*J+0)), shape=G.shape, dtype=bm.float64)
        G += coo_matrix((-v[:, 2], (3*I+0, 6*J+4)), shape=G.shape, dtype=bm.float64)
        G += coo_matrix(( v[:, 1], (3*I+0, 6*J+5)), shape=G.shape, dtype=bm.float64)

        G += coo_matrix((    ones, (3*I+1, 6*J+1)), shape=G.shape, dtype=bm.float64)
        G += coo_matrix(( v[:, 2], (3*I+1, 6*J+3)), shape=G.shape, dtype=bm.float64)
        G += coo_matrix((-v[:, 0], (3*I+1, 6*J+5)), shape=G.shape, dtype=bm.float64)

        G += coo_matrix((    ones, (3*I+2, 6*J+2)), shape=G.shape, dtype=bm.float64)
        G += coo_matrix((-v[:, 1], (3*I+2, 6*J+3)), shape=G.shape, dtype=bm.float64)
        G += coo_matrix(( v[:, 0], (3*I+2, 6*J+4)), shape=G.shape, dtype=bm.float64)

        return G.tocsr().to_scipy()


    def box_linear_system(self):
        """
        """
        # Implementation of the linear system construction goes here
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

        # 自由点不是参考点，不是固定点，不是曲面上的点（依赖点） 
        isRNode = self.mesh.data.get_node_data('isRNode')
        isFNode = self.mesh.data.get_node_data('isFNode')
        isCSNode = self.mesh.data.get_node_data('isCSNode')
        isFreeNode = ~(isRNode | isFNode | isCSNode) 
        isFreeDof = bm.repeat(isFreeNode, 3)

        isCSDof = bm.repeat(isCSNode, 3)

        S0 = S[isFreeDof, :]
        S1 = S[isCSDof, :]

        S00 = S0[:, isFreeDof]
        S01 = S0[:, isCSDof] @ self.G
        S11 = self.G.T @ S1[:, isCSDof] @ self.G 

        M0 = M[isFreeDof, :]
        M1 = M[isCSDof, :]
        M00 = M0[:, isFreeDof]
        M01 = M0[:, isCSDof] @ self.G
        M11 = self.G.T @ M1[:, isCSDof] @ self.G

        return [[S00, S01], [S01.T, S11]], [[M00, M01], [M01.T, M11]] 

    def shaft_linear_system(self):
        """
        轴系的线性系统对应的自由度，可以分为耦合与非耦合两部分。
        耦合部分是指轴系的耦合节点，这些节点的自由度需要与齿轮箱的自由度进行耦合。
        这里的问题是，轴系的耦合节点的排序与齿轮箱对应节点排序不一致。
        因此需要对轴系的线性系统进行处理，将耦合节点的自由度与齿轮箱的自由度进行匹配。

        1. 轴系的矩阵进行重排序
        2. 非耦合节点的排序不变
        3. 耦合节点的排序要与齿轮箱的节点排序一致
        """
        from scipy.io import loadmat
        from scipy.sparse import csr_matrix

        shaft_system = loadmat(self.options['shaft_system_file'])
        S = shaft_system['stiffness_total_system_spectrum']
        M = shaft_system['mass_total_system']

        self.logger.info(
                f"Sucsess load shaft system from {self.options['shaft_system_file']}"
                f"shaft system stiffness matrix: {S.shape}"
                f"shaft system mass matrix: {M.shape}")

        d0 = [
            [1,  5,  21.5, -1.1, 350250],
            [1, 13,  75.5, -1.1, 350251],
            [3,  4,  12.5, -1.1, 350247],
            [3, 17,  129, -1.1, 350248],
            [4,  7,  100.8, -1.1, 350243],
            [4, 25,  253.6, -1.1, 350244],
            [4,  4,  70, -1.1, 350241],
        ]
        d0 = bm.array(d0, dtype=bm.float64)
        d1 = [[1, 1.1, 1.2, 2, 2.1, 2.2, 3, 4, 5], [17, 5, 3, 7, 3, 3, 18, 26, 8]]
        d1 = bm.array(d1).T
        NN = int(d1[:, 1].sum())
        offset = [0] + [int(a) for a in bm.cumsum(d1[:, 1])] 
        self.logger.info(
                f"Number of nodes in the shaft system: {NN}"
                f"Offset for the nodes: {offset}")

        nidmap = self.mesh.data['nidmap']
        imap = {str(a[0]) : a[1] for a in zip(d1[:, 0], range(d1.shape[0]))}
        # Create a boolean array to mark coupling nodes
        isCouplingNodes = bm.zeros(NN, dtype=bm.bool)
        for i, j in d0[:, :2]:
            idx = imap[str(i)] + int(j) - 1
            isCouplingNodes[idx] = True
        isCouplingDofs = bm.repeat(isCouplingNodes, 6)
        self.logger.info(f"Number of coupling nodes: {bm.sum(isCouplingNodes)}")

        S0 = S[bm.logical_not(isCouplingDofs), :]
        S1 = S[isCouplingDofs, :]

        S00 = S0[:, bm.logical_not(isCouplingDofs)]
        S01 = S0[:, isCouplingDofs]
        S11 = S1[:, isCouplingDofs]

        M0 = M[bm.logical_not(isCouplingDofs), :]
        M1 = M[isCouplingDofs, :]

        M00 = M0[:, bm.logical_not(isCouplingDofs)]
        M01 = M0[:, isCouplingDofs]
        M11 = M1[:, isCouplingDofs]
        self.logger.info(
                f"shaft system stiffness matrix: {S00.shape}, {S01.shape}, {S11.shape}"
                f"shaft system mass matrix: {M00.shape}, {M01.shape}, {M11.shape}")

        # 重新对耦合节点的自由度进行排序

        cnode = nidmap[d0[:, -1].astype(bm.int32)]
        isCouplingNodes = bm.zeros(self.mesh.number_of_nodes(), dtype=bm.bool)
        isCouplingNodes = bm.set_at(isCouplingNodes, cnode, True)
        self.mesh.data.add_node_data('isCouplingNodes', isCouplingNodes)
        re = bm.argsort(cnode)
        idx = bm.arange(6 * cnode.shape[0], dtype=bm.int32).reshape(-1, 6)
        idx = idx[re, :].flatten()
        S01 = csr_matrix(S01[:, idx])
        S11 = csr_matrix(S11[idx, :][:, idx])
        M01 = csr_matrix(M01[:, idx])
        M11 = csr_matrix(M11[idx, :][:, idx])

        return [[S00, S01], [S01.T, S11]], [[M00, M01], [M01.T, M11]]



    @variantmethod('scipy')
    def solve(self, which: str ='SM'):
        """Solve the eigenvalue problem using SLEPc.
        
        """
        from petsc4py import PETSc
        from slepc4py import SLEPc

        self.shaft_linear_system()
        #S, M = self.linear_system()
        #S, M = self.apply_bc(S, M)


        

