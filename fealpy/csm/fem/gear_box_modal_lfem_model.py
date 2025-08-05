
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
        self.set_node_flag()
        self.set_space_degree(options['space_degree'])

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

    def set_node_flag(self): 
        """
        Set node flags for the gear box model.

        the node flags is the base for the dofs flag
        """
        redges, rnodes = self.mesh.data.get_rbe2_edge()

        isRefNodes = bm.zeros(self.mesh.number_of_nodes(), dtype=bm.bool)
        isRefNodes[rnodes] = True

        isGearNodes = bm.logical_not(isRefNodes)
        isSurfaceNodes = bm.zeros(self.mesh.number_of_nodes(), dtype=bm.bool)
        isSurfaceNodes[redges[:, 1]] = True
        
        # TODO: get the fixed nodes from the boundary conditions
        name = self.mesh.data['boundary_conditions'][0][0]
        nset = self.mesh.data.get_node_set(name)

        isFixedNodes = bm.zeros(self.mesh.number_of_nodes(), dtype=bm.bool)
        isFixedNodes[nset] = True

        self.mesh.data.add_node_data('isRefNodes', isRefNodes)
        self.mesh.data.add_node_data('isGearNodes', isGearNodes)
        self.mesh.data.add_node_data('isSurfaceNodes', isSurfaceNodes)
        self.mesh.data.add_node_data('isFixedNodes', isFixedNodes)

        self.G = self.rbe2_matrix(redges, rnodes)


    def rbe2_matrix(self, redges, rnodes):
        """
        """
        NN = self.mesh.number_of_nodes()

        isRefNodes = self.mesh.data.get_node_data('isRefNodes') 
        isSurfaceNodes = self.mesh.data.get_node_data('isSurfaceNodes')
        ridx = bm.where(isRefNodes)[0]
        sidx = bm.where(isSurfaceNodes)[0]
        rmap = bm.zeros(NN, dtype=bm.int32)
        smap = bm.zeros(NN, dtype=bm.int32)
        rmap[ridx] = bm.arange(ridx.shape[0], dtype=bm.int32)
        smap[sidx] = bm.arange(sidx.shape[0], dtype=bm.int32)

        I = smap[redges[:, 1]]
        J = rmap[redges[:, 0]]
        

        node = self.mesh.entity('node')
        # surface node point to reference node
        v = node[redges[:, 0]] - node[redges[:, 1]]

        NS = redges.shape[0] # number of surface nodes
        NR = rnodes.shape[0] # number of reference nodes

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


    def linear_system(self):
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

        return S, M

    def apply_bc(self, S, M):
        """
        """

        S = S.to_scipy()
        M = M.to_scipy()
        # 自由点不是参考点，不是固定点，不是曲面上的点（依赖点） 
        isRefNodes = self.mesh.data.get_node_data('isRefNodes')
        isFixedNodes = self.mesh.data.get_node_data('isFixedNodes')
        isSurfaceNodes = self.mesh.data.get_node_data('isSurfaceNodes')
        isFreeNodes = ~(isRefNodes | isFixedNodes | isSurfaceNodes) 
        isFreeDofs = bm.repeat(isFreeNodes, 3)

        isSurfaceDofs = bm.repeat(isSurfaceNodes, 3)

        S0 = S[isFreeDofs, :]
        S1 = S[isSurfaceDofs, :]
        M0 = M[isFreeDofs, :]
        M1 = M[isSurfaceDofs, :]

        S00 = S0[:, isFreeDofs]
        S01 = S0[:, isSurfaceDofs] @ self.G
        S11 = self.G.T @ S1[:, isSurfaceDofs] @ self.G 

        M00 = M0[:, isFreeDofs]
        M01 = M0[:, isSurfaceDofs] @ self.G
        M11 = self.G.T @ M1[:, isSurfaceDofs] @ self.G

        return [[S00, S01], [S01.T, S11]], [[M00, M01], [M01.T, M11]] 

    def load_shaft_system(self):
        """
        """
        from scipy.io import loadmat
        from scipy.sparse import csr_matrix
        shaft_system = loadmat(self.options['shaft_system_file'])
        self.logger.info(f"Sucsess load shaft system from {self.options['shaft_system_file']}")

        S = shaft_system['stiffness_total_system_spectrum']
        M = shaft_system['mass_total_system']

        self.logger.info(f"shaft system stiffness matrix: {S.shape}")
        self.logger.info(f"shaft system mass matrix: {M.shape}")

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
        self.logger.info(f"Number of nodes in the shaft system: {NN}")

        offset = [0] + [int(a) for a in bm.cumsum(d1[:, 1])] 
        self.logger.info(f"Offset for the nodes: {offset}")

        nidmap = self.mesh.data['nidmap']

        imap = {str(a[0]) : a[1] for a in zip(d1[:, 0], range(d1.shape[0]))}

        self.logger.info(imap)
        
        # Create a boolean array to mark coupling nodes
        isCouplingNodes = bm.zeros(NN, dtype=bm.bool)
        for i, j in d0[:, :2]:
            idx = imap[str(i)] + int(j) - 1
            isCouplingNodes[idx] = True

        isCouplingDofs = bm.repeat(isCouplingNodes, 6)

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

        cnode = nidmap[d0[:, -1].astype(bm.int32)]
        self.logger.info(f"Coupling nodes: {cnode}")



    @variantmethod('scipy')
    def solve(self, which: str ='SM'):
        """Solve the eigenvalue problem using SLEPc.
        
        """
        from petsc4py import PETSc
        from slepc4py import SLEPc
        #S, M = self.linear_system()
        #S, M = self.apply_bc(S, M)

        self.load_shaft_system()

        

