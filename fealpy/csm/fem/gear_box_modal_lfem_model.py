
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
        DirichletBC
        )
from fealpy.fem import ScalarMassIntegrator as MassIntegrator

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

        self.logger.info(self.G)


    def rbe2_matrix(self, redges, rnodes):
        """
        """
        NN = self.mesh.number_of_nodes()

        isRefNodes = self.mesh.data.get_node_data('isRefNodes') 
        isSurfaceNodes = self.mesh.data.get_node_data('isSurfaceNodes')
        # 顺序一致性
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

        G = coo_matrix((3*NS, 6*NR), ftype=bm.float64)
        ones = bm.ones(NS, dtype=bm.float64)

        G += coo_matrix((    ones, (3*I+0, 6*J+0)), shape=G.shape, ftype=bm.float64)
        G += coo_matrix((-v[:, 2], (3*I+0, 6*J+4)), shape=G.shape, ftype=bm.float64)
        G += coo_matrix(( v[:, 1], (3*I+0, 6*J+5)), shape=G.shape, ftype=bm.float64)

        G += coo_matrix((    ones, (3*I+1, 6*J+1)), shape=G.shape, ftype=bm.float64)
        G += coo_matrix(( v[:, 2], (3*I+1, 6*J+3)), shape=G.shape, ftype=bm.float64)
        G += coo_matrix((-v[:, 0], (3*I+1, 6*J+5)), shape=G.shape, ftype=bm.float64)

        G += coo_matrix((    ones, (3*I+2, 6*J+2)), shape=G.shape, ftype=bm.float64)
        G += coo_matrix((-v[:, 1], (3*I+2, 6*J+3)), shape=G.shape, ftype=bm.float64)
        G += coo_matrix(( v[:, 0], (3*I+2, 6*J+4)), shape=G.shape, ftype=bm.float64)

        return G.tocsr()



        

