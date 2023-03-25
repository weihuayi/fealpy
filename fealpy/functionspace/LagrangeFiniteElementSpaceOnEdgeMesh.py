import numpy as np


class CEdgeMeshFEMDof():
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = mesh.multi_index_matrix() 
        self.cell2dof = self.cell_to_dof()

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

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')

        if p == 1:
            return cell
        else:
            NN = mesh.number_of_nodes()
            NC = mesh.number_of_cells()
            ldof = self.number_of_local_dofs()
            cell2dof = np.zeros((NC, ldof), dtype=mesh.itype)
            cell2dof[:, [0, -1]] = cell
            cell2dof[:, 1:-1] = NN + np.arange(NC*(p-1)).reshape(NC, p-1)
            return cell2dof

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
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell
        node = mesh.node

        if p == 1:
            return node
        else:
            NN = mesh.number_of_nodes()
            gdof = self.number_of_global_dofs()
            shape = (gdof,) + node.shape[1:]
            ipoint = np.zeros(shape, dtype=np.float64)
            ipoint[:NN] = node
            NC = mesh.number_of_cells()
            cell = mesh.ds.cell
            w = np.zeros((p-1,2), dtype=np.float64)
            w[:,0] = np.arange(p-1, 0, -1)/p
            w[:,1] = w[-1::-1, 0]
            GD = mesh.geo_dimension()
            ipoint[NN:NN+(p-1)*NC] = np.einsum('ij, kj...->ki...', w,
                    node[cell]).reshape(-1, GD)

            return ipoint

class LagrangeFiniteElementSpaceOnEdgeMesh():
    def __init__(self, mesh, p=1, spacetype='C', dof=None):
        self.mesh = mesh
        self.cellmeasure = mesh.entity_measure('cell')
        self.p = p
        if dof is None:
            if spacetype == 'C':
            elif spacetype == 'D':
        else:
            self.dof = dof
            self.TD = mesh.top_dimension() 

        self.GD = mesh.geo_dimension()

        self.spacetype = spacetype
        self.itype = mesh.itype
        self.ftype = mesh.ftype

        self.multi_index_matrix = multi_index_matrix 

    def __str__(self):
        return "Lagrange finite element space on edge mesh!"

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='cell'):
        if self.spacetype == 'C':
            return self.dof.number_of_local_dofs(doftype=doftype)
        elif self.spacetype == 'D':
            return self.dof.number_of_local_dofs()
