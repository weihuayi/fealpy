import numpy as np
import jax
import jax.numpy as jnp

class LinearMeshCFEDof():
    def __init__(self, mesh, p):
        TD = mesh.top_dimension()
        self.mesh = mesh
        self.p = p
        self.multiIndex = mesh.multi_index_matrix(p, TD) 
        self.cell2dof = self.cell_to_dof()

    def is_boundary_dof(self, threshold=None):
        TD = self.mesh.top_dimension()
        gdof = self.number_of_global_dofs()
        if type(threshold) is np.ndarray:
            index = threshold
            if (index.dtype == np.bool_) and (len(index) == gdof):
                return index
        else:
            index = self.mesh.ds.boundary_face_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter(TD-1, index=index)
                flag = threshold(bc)
                index = index[flag]

        face2dof = self.face_to_dof(index=index) # 只获取指定的面的自由度信息
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[face2dof] = True
        return isBdDof

    def face_to_dof(self, index=np.s_[:]):
        return self.mesh.face_to_ipoint(self.p, index=index)

    def edge_to_dof(self, index=np.s_[:]):
        return self.mesh.edge_to_ipoint(self.p, index=index)

    def cell_to_dof(self, index=np.s_[:]):
        return self.mesh.cell_to_ipoint(self.p, index=index)

    def interpolation_points(self, index=np.s_[:]):
        return self.mesh.interpolation_points(self.p, index=index)

    def number_of_global_dofs(self):
        return self.mesh.number_of_global_ipoints(self.p)

    def number_of_local_dofs(self, doftype='cell'):
        return self.mesh.number_of_local_ipoints(self.p, iptype=doftype)

class LagrangeFESpace():

    def __init__(self, mesh, p=1, ctype='C'):
        self.mesh = mesh
        self.p = p

        assert ctype in {'C', 'D'}
        self.ctype = ctype # 空间连续性类型
        if ctype == 'C':
            self.dof = LinearMeshCFEDof(mesh, p)

        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()


    def basis(self, bc, index=np.s_[:]):
        p = self.p
        phi = self.mesh.shape_function(bc, p=p)
        return phi[..., None, :]

    def grad_basis(self, bc, index=np.s_[:], variables='x'):
        """
        @brief
        """
        return self.mesh.grad_shape_function(bc, p=self.p, index=index, variables=variables)


    def value(self, uh, bc, index=np.s_[:]):
        """
        @brief
        """
        pass
