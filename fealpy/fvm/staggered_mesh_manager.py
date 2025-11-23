from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh
class StaggeredMeshManager:
    

    def __init__(self, domain, nx, ny):
        x_left = domain[0]
        x_right = domain[1]
        y_bottom = domain[2]
        y_top = domain[3]
        hx = (x_right-x_left)/ nx
        hy = (y_top-y_bottom)/ ny

        self.umesh = QuadrangleMesh.from_box(
            box=[-hx / 2 + x_left, hx / 2 + x_right, y_bottom, y_top],
            nx=nx + 1, ny=ny
        )
        self.vmesh = QuadrangleMesh.from_box(
            box=[x_left, x_right, -hy / 2 + y_bottom, hy / 2 + y_top],
            nx=nx, ny=ny + 1
        )
        self.pmesh = QuadrangleMesh.from_box(box=domain, nx=nx, ny=ny)

    def get_dof_mapping_ucell2pedge(self):
        pc2e = self.pmesh.cell_to_edge()
        ucell2pedge = bm.unique(bm.concatenate((pc2e[:,3],pc2e[:,1])))
        return ucell2pedge
    
    def get_dof_mapping_vcell2pedge(self):
        pc2e = self.pmesh.cell_to_edge()
        vcell2pedge = bm.unique(bm.concatenate((pc2e[:,2],pc2e[:,0])))
        return vcell2pedge
    
    def get_dof_mapping_pcell2uedge(self):
        import numpy as np 
        uc2e = self.umesh.cell_to_edge()
        pcell2uedge = np.intersect1d(uc2e[:,1],uc2e[:,3])
        return pcell2uedge
    
    def get_dof_mapping_pcell2vedge(self):
        import numpy as np 
        vc2e = self.vmesh.cell_to_edge()
        pcell2vedge = np.intersect1d(vc2e[:,0],vc2e[:,2])
        return pcell2vedge 
    
    def get_dof_mapping_uedge2vedge(self):
        import numpy as np 
        uc2e = self.umesh.cell_to_edge()
        vc2e = self.vmesh.cell_to_edge()
        a = bm.unique(bm.concatenate((uc2e[:,2],uc2e[:,0])))
        b = bm.unique(bm.concatenate((vc2e[:,3],vc2e[:,1])))
        c = np.intersect1d(uc2e[:,3],uc2e[:,1])
        d = np.intersect1d(vc2e[:,2],vc2e[:,0])
        NE = self.umesh.number_of_edges()
        uedge2vedge = bm.zeros(NE)  
        uedge2vedge[b] = a
        uedge2vedge[d] = c
        return uedge2vedge
    
    def get_dof_mapping_vedge2uedge(self):
        import numpy as np 
        uc2e = self.umesh.cell_to_edge()
        vc2e = self.vmesh.cell_to_edge()
        a = bm.unique(bm.concatenate((uc2e[:,2],uc2e[:,0])))
        b = bm.unique(bm.concatenate((vc2e[:,3],vc2e[:,1])))
        c = np.intersect1d(uc2e[:,3],uc2e[:,1])
        d = np.intersect1d(vc2e[:,2],vc2e[:,0])
        NE = self.umesh.number_of_edges()
        vedge2uedge = bm.zeros(NE)  
        vedge2uedge[a] = b
        vedge2uedge[c] = d
        return vedge2uedge

    def map_velocity_uvcell_to_pedge(self, u_cell, v_cell, ap_u, ap_v):
        ucell2pedge = self.get_dof_mapping_ucell2pedge()
        vcell2pedge = self.get_dof_mapping_vcell2pedge()

        NE = self.pmesh.number_of_edges()
        p_edge_velocity = bm.zeros(NE)
        ap_edge = bm.zeros(NE)
        
        p_edge_velocity[ucell2pedge] = u_cell
        p_edge_velocity[vcell2pedge] = v_cell
        ap_edge[ucell2pedge] = ap_u
        ap_edge[vcell2pedge] = ap_v

        return p_edge_velocity, ap_edge
    
    def map_pressure_pcell_to_uvedge(self, p):
        ucell2pedge = self.get_dof_mapping_ucell2pedge()
        vcell2pedge = self.get_dof_mapping_vcell2pedge()

        pe2c = self.pmesh.edge_to_cell()[:,:2]
        p_e = (p[pe2c[:, 0]] + p[pe2c[:, 1]]) / 2
        p_u = p_e[ucell2pedge]
        p_v = p_e[vcell2pedge]
        return p_u, p_v