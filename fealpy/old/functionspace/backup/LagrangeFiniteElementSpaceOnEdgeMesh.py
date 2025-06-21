import numpy as np
from .Function import Function
from ..decorator import barycentric


class CEdgeMeshLFEMDof():
    """
    @brief EdgeMesh 上的分片 p 次连续元的自由度管理类
    """
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = mesh.multi_index_matrix(p) 

    def is_boundary_dof(self, threshold=None):
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_node_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('node', index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[index] = True
        return isBdDof

    def entity_to_dof(self, etype='cell', index=np.s_[:]):
        if etype in {'cell', 'edge', 1}:
            return self.cell_to_dof()[index]
        elif etype in {'node', 0}:
            NN = self.mesh.number_of_nodes()
            return np.arange(NN)[index]

    def cell_to_dof(self, index=np.s_[:]):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')

        if p == 1:
            return cell[index]
        else:
            NN = mesh.number_of_nodes()
            NC = mesh.number_of_cells()
            ldof = self.number_of_local_dofs()
            cell2dof = np.zeros((NC, ldof), dtype=mesh.itype)
            cell2dof[:, [0, -1]] = cell
            cell2dof[:, 1:-1] = NN + np.arange(NC*(p-1)).reshape(NC, p-1)
            return cell2dof[index]

    def number_of_local_dofs(self, doftype='cell'):
        if doftype in {'cell', 'edge', 1}:
            return self.p + 1
        elif doftype in {'face', 'node', 0}:
            return 1

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        gdof = mesh.number_of_nodes()
        if p > 1:
            NC = mesh.number_of_cells()
            gdof += NC*(p-1)
        return gdof

    def interpolation_points(self):
        return self.mesh.interpolation_points(self.p)

class DEdgeMeshLFEMDof():
    """
    @brief EdgeMesh 上的分片 p 次间断元的自由度管理类
    """
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = mesh.multi_index_matrix() 

    def is_boundary_dof(self, threshold=None):
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_node_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('node', index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[index] = True
        return isBdDof

    def entity_to_dof(self, etype='cell', index=np.s_[:]):
        return self.cell_to_dof()[index]

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')
        ldof = self.number_of_local_dofs()
        cell2dof = np.arange(NC*(p+1)).reshape(NC, p+1)
        return cell2dof

    def number_of_local_dofs(self, _):
        return self.p + 1

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        gdof = NC*(p+1)
        return gdof

    def interpolation_points(self):
        return self.mesh.interpolation_points(self.p)

class LagrangeFiniteElementSpaceOnEdgeMesh():
    def __init__(self, mesh, p=1, spacetype='C', dof=None, doforder='nodes'):
        """
        @param[in] doforder 向量空间自由度排序的约定，'nodes'(default) and 'vdims'

        @note  'nodes': x_0, x_1, x_2, ..., y_0, y_1, ..., z_0, z_1, ...
               'vdims': x_0, y_0, z_0, x_1, y_1, z_1, ...
        """
        self.mesh = mesh
        self.cellmeasure = mesh.entity_measure('cell')
        self.p = p
        self.doforder = doforder # 扩展为向量空间时, 自由度的排序规则 
        if dof is None:
            if spacetype == 'C':
                self.dof = CEdgeMeshLFEMDof(mesh, p)
            elif spacetype == 'D':
                self.dof = DEdgeMeshLFEMDof(mesh, p)
        else:
            self.dof = dof

        self.TD = mesh.top_dimension() 
        self.GD = mesh.geo_dimension()

        self.spacetype = spacetype
        self.itype = mesh.itype
        self.ftype = mesh.ftype

    def __str__(self):
        return "Lagrange finite element space on edge mesh, which can be used on the structure of Truss and Frame simulation!"

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='cell'):
        return self.dof.number_of_local_dofs(doftype=doftype)

    def cell_to_dof(self):
        return self.dof.cell_to_dof()

    @barycentric
    def basis(self, bc, index=np.s_[:]):
        p = self.p
        phi = self.mesh.shape_function(bc, p=p)
        return phi[..., None, :]

    @barycentric
    def grad_basis(self, bc, index=np.s_[:]):
        return self.mesh.grad_shape_function(bc, p=self.p, index=index)

    @barycentric
    def value(self, uh, bc, index=np.s_[:]):
        gdof = self.number_of_global_dofs()
        phi = self.basis(bc, index=index) # (NQ, NC, ldof)
        cell2dof = self.dof.cell_to_dof(index=index)

        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        if self.doforder == 'nodes':
            # phi.shape == (NQ, NC, ldof)
            # uh.shape == (..., gdof)
            # uh[..., cell2dof].shape == (..., NC, ldof)
            # val.shape == (NQ, ..., NC)
            s1 = '...ci, {}ci->...{}c'.format(s0[:dim], s0[:dim])
            val = np.einsum(s1, phi, uh[..., cell2dof])
        elif self.doforder == 'vdims':
            # phi.shape == (NQ, NC, ldof)
            # uh.shape == (gdof, ...)
            # uh[cell2dof, ...].shape == (NC, ldof, ...)
            # val.shape == (NQ, NC, ...)
            s1 = '...ci, ci{}->...c{}'.format(s0[:dim], s0[:dim])
            val = np.einsum(s1, phi, uh[cell2dof, ...])
            return val

    @barycentric
    def grad_value(self, uh, bc, index=np.s_[:]):
        """
        """
        gdof = self.number_of_global_dofs()
        gphi = self.grad_basis(bc, index=index) # (NQ, NC, ldof, GD)
        cell2dof = self.dof.cell_to_dof(index=index)
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        if self.doforder == 'nodes':
            # gphi.shape == (NQ, NC, ldof, GD)
            # uh.shape == (..., gdof)
            # uh[..., cell2dof].shape == (..., NC, ldof)
            # val.shape == (NQ, ..., GD, NC)
            s1 = '...cim, {}ci->...{}mc'.format(s0[:dim], s0[:dim])
            val = np.einsum(s1, gphi, uh[..., cell2dof])
        elif self.doforder == 'vdims':
            # gphi.shape == (NQ, NC, ldof, GD)
            # uh.shape == (gdof, ...)
            # uh[cell2dof, ...].shape == (NC, ldof, ...)
            # val.shape == (NQ, NC, ..., GD)
            s1 = '...cim, ci{}->...c{}m'.format(s0[:dim], s0[:dim])
            val = np.einsum(s1, gphi, uh[cell2dof[index], ...])
        return val

    def function(self, dim=None, array=None, dtype=np.float64):
        return Function(self, dim=dim, array=array, 
                coordtype='barycentric', dtype=dtype)

    def array(self, dim=None, dtype=np.float64):
        gdof = self.number_of_global_dofs()
        if (dim is None):
            dim = tuple() 
        if type(dim) is int:
            dim = (dim, )

        if self.doforder == 'nodes':
            shape = dim + (gdof, )
        elif self.doforder == 'vdims':
            shape = (gdof, ) + dim

        return np.zeros(shape, dtype=dtype)
