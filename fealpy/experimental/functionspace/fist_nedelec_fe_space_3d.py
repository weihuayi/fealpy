

    @property
    def cell2dof(self):
        return self.cell_to_dof()

    def boundary_dof(self):
        eidx = self.mesh.ds.boundary_edge_index()
        e2d = self.edge_to_dof(index=eidx)
        return e2d.reshape(-1)

    def is_boundary_dof(self):
        bddof = self.boundary_dof()

        gdof = self.number_of_global_dofs()
        flag = np.zeros(gdof, dtype=np.bool_)

        flag[bddof] = True
        return flag
    
mesh = TriangleMesh.from_box(nx= 1,ny = 1)
p = 4
a = FirstNedelecDof2d(mesh,p)
k = a.cell2dof
print((k,))





