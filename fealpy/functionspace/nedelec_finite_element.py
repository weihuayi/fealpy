import numpy as np
from .function import Function
from .lagrange_fem_space import LagrangeFiniteElementSpace

class NedelecFiniteElement2d():
    def __init__(self, mesh, p, flag=0):
        self.mesh = mesh
        self.p = p
        self.flag = flag # 0: the first kind of Nedelec element
                         # 1: the second kind of Nedelec element

        self.itype = mesh.itype
        self.ftype = mesh.ftype

    def basis(self, bc):

        mesh = self.mesh
        NC = mesh.number_of_cells()
        GD = mesh.geom_dimension()
        p = self.p

        ldof = self.number_of_local_dofs()
        phi = np.zeros((NC, ldof, GD), dtype=self.ftype)

    def number_of_local_dofs():
        GD = mesh.geom_dimension()
        if GD == 2:
            if self.flag == 0: 
                return 3
            else:
                return 6
        else:
            if self.flag == 0:
                return 6
            else:
                return 12


        

class FirstNedelecFiniteElement2d():
    def __init__(self, mesh, p=0):

        self.mesh = mesh
        self.p = p

        self.itype = mesh.itype
        self.ftype = mesh.ftype

    def cell_to_dof(self):
        pass
    
    def cell_to_edge_sign(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        edge2cell = mesh.ds.edge2cell
        cell2edgeSign = -np.ones((NC, 3), dtype=np.int)
        cell2edgeSign[edge2cell[:, 0], edge2cell[:, 2]] = 1
        return cell2edgeSign

    def basis(self, bc):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        dim = mesh.geom_dimension()
        p = self.p
        ldof = self.number_of_local_dofs()
        phi = np.zeros((NC, ldof, dim), dtype=self.dtype)
        Dlambda, _ = mesh.grad_lambda() 
        cell2edgeSign = self.cell_to_edge_sign()
        if p == 0:
            phi[:, 0, :] = bc[1]*Dlambda[:, 2, :] - bc[2]*Dlambda[:, 1, :]
            phi[:, 1, :] = bc[2]*Dlambda[:, 0, :] - bc[0]*Dlambda[:, 2, :]
            phi[:, 2, :] = bc[0]*Dlambda[:, 1, :] - bc[1]*Dlambda[:, 0, :]
            phi *= cell2edgeSign.reshape(-1, 3, 1)
        else:
            #TODO: raise a error
            print("error!")

        return phi

    def grad_basis(self, bc):
        mesh = self.mesh

        NC = mesh.number_of_cells()
        dim = mesh.geom_dimension()

        ldof = self.number_of_local_dofs()

        gradPhi = np.zeros((NC, ldof, dim, dim), dtype=self.dtype)

        cell2edgeSign = self.cell_to_edge_sign()
        Dlambda, _ = mesh.grad_lambda(bc)
        if p == 0:
            A = np.einsum('...i, ...j->...ij', Dlambda[:, 2, :], Dlambda[:, 1, :])
            B = np.einsum('...i, ...j->...ij', Dlambda[:, 1, :], Dlambda[:, 2, :]) 
            gradPhi[:, 0, :, :] = A - B 

            A = np.einsum('...i, ...j->...ij', Dlambda[:, 0, :], Dlambda[:, 2, :])
            B = np.einsum('...i, ...j->...ij', Dlambda[:, 2, :], Dlambda[:, 0, :])
            gradPhi[:, 1, :, :] = A - B 

            A = np.einsum('...i, ...j->...ij', Dlambda[:, 1, :], Dlambda[:, 0, :])
            B = np.einsum('...i, ...j->...ij', Dlambda[:, 0, :], Dlambda[:, 1, :])
            gradPhi[:, 2, :, :] = A - B 

            gradPhi *= cell2edgeSign.reshape(-1, 3, 1, 1) 
        else:
            #TODO:raise a error
            print("error")

        return gradPhi 

    def div_basis(self, bc):
        mesh = self.mesh
        p = self.p

        divPhi = np.zeors((NC, ldof), dtype=self.dtype)

        cell2edgeSign = self.cell_to_edge_sign()
        Dlambda, _ = mesh.grad_lambda()
        if p == 0:
            divPhi[:, 0] = np.sum(Dlambda[:, 1, :]*Dlambda[:, 2, :] - Dlambda[:, 2, :]*Dlambda[:, 1, :], axia=1)
            divPhi[:, 1] = np.sum(Dlambda[:, 2, :]*Dlambda[:, 0, :] - Dlambda[:, 0, :]*Dlambda[:, 2, :], axis=1)
            divPhi[:, 2] = np.sum(Dlambda[:, 0, :]*Dlambda[:, 1, :] - Dlambda[:, 1, :]*Dlambda[:, 0, :], axis=1)
            divPhi *= cell2edgeSign 
        else:
            #TODO:raise a error
            print("error")

        return divPhi 

    def dual_basis(self, u):
        pass

    def array(self):
        pass

    def value(self, uh, bc):
        pass

    def grad_value(self, uh, bc):
        pass

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        if p == 0:
            return NE
        else:
            #TODO: raise a error
            print("error!")

    def number_of_local_dofs(self):
        p = self.p
        if p==0:
            return 3
        else:
            #TODO: raise a error
            print("error!")

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell

        N = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()
        if p == 0:
            cell2dof = mesh.ds.cell2edge
        else:
            #TODO: raise a error 
            print('error!')

        return cell2dof

class SecondNedelecFiniteElementTwo2d():
    def __init__(self, mesh, p=1, dtype=np.float):
        self.mesh=mesh
        self.p = p
    
    def cell_to_edge_sign(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        edge2cell = mesh.ds.edge2cell
        cell2edgeSign = -np.ones((NC, 3), dtype=np.int)
        cell2edgeSign[edge2cell[:, 0], edge2cell[:, 2]] = 1
        return cell2edgeSign

    def basis(self, bc):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        dim = mesh.geom_dimentsion()
        p = self.p
        ldofs = mesh.number_of_local_dofs()
        phi = np.zeros((NC, ldofs, dim), dtype=self.dtype)
        Dlambda, _ = mesh.grad_lambda() 
        cell2edgeSign = self.cell_to_edge_sign()
        if p == 1:
            phi[:, 0, :] = bc[1]*Dlambda[:, 2, :] - bc[2]*Dlambda[:, 1, :]
            phi[:, 1, :] = bc[1]*Dlambda[:, 2, :] + bc[2]*Dlambda[:, 1, :]
            phi[:, 2, :] = bc[2]*Dlambda[:, 0, :] - bc[0]*Dlambda[:, 2, :]
            phi[:, 3, :] = bc[2]*Dlambda[:, 0, :] + bc[0]*Dlambda[:, 2, :]
            phi[:, 4, :] = bc[0]*Dlambda[:, 1, :] - bc[1]*Dlambda[:, 0, :]
            phi[:, 5, :] = bc[0]*Dlambda[:, 1, :] + bc[1]*Dlambda[:, 0, :]
            phi[:, 0:6:2] *=cell2edgeSign
            phi[:, 1:6:2] *=cell2edgeSign
        else:
            #TODO: raise a error
            print("error!")

        return phi

    def grad_basis(self, bc):
        mesh = self.mesh

        NC = mesh.number_of_cells()
        dim = mesh.geom_dimension()

        ldof = self.number_of_local_dofs()

        gradPhi = np.zeros((NC, ldof, dim, dim), dtype=self.dtype)

        cell2edgeSign = self.cell_to_edge_sign()
        Dlambda, _ = mesh.grad_lambda()
        if p == 1:
            A = np.einsum('...i, ...j->...ij', Dlambda[:, 2, :], Dlambda[:, 1, :])
            B = np.einsum('...i, ...j->...ij', Dlambda[:, 1, :], Dlambda[:, 2, :]) 
            gradPhi[:, 0, :, :] = A - B 
            gradPhi[:, 1, :, :] = A + B 

            A = np.einsum('...i, ...j->...ij', Dlambda[:, 0, :], Dlambda[:, 2, :])
            B = np.einsum('...i, ...j->...ij', Dlambda[:, 2, :], Dlambda[:, 0, :])
            gradPhi[:, 2, :, :] = A - B 
            gradPhi[:, 3, :, :] = A + B 

            A = np.einsum('...i, ...j->...ij', Dlambda[:, 1, :], Dlambda[:, 0, :])
            B = np.einsum('...i, ...j->...ij', Dlambda[:, 0, :], Dlambda[:, 1, :])
            gradPhi[:, 4, :, :] = A - B 
            gradPhi[:, 5, :, :] = A + B 

            gradPhi[:, 0:6:2, :, :] *= cell2edgeSign.reshape(-1, 3, 1, 1) 
            gradPhi[:, 1:6:2, :, :] *= cell2edgeSign.reshape(-1, 3, 1, 1) 

        else:
            #TODO:raise a error
            print("error")

        return gradPhi 

    def div_basis(self, bc):
        mesh = self.mesh
        p = self.p

        divPhi = np.zeors((NC, ldof), dtype=self.dtype)

        Dlambda, _ = mesh.grad_lambda()
        if p == 1:
            divPhi[:, 0] = np.sum(Dlambda[:, 1, :]*Dlambda[:, 2, :] - Dlambda[:, 2, :]*Dlambda[:, 1, :], axia=1)
            divPhi[:, 1] = np.sum(Dlambda[:, 1, :]*Dlambda[:, 2, :] + Dlambda[:, 2, :]*Dlambda[:, 1, :], axia=1)
            divPhi[:, 2] = np.sum(Dlambda[:, 2, :]*Dlambda[:, 0, :] - Dlambda[:, 0, :]*Dlambda[:, 2, :], axis=1)
            divPhi[:, 3] = np.sum(Dlambda[:, 2, :]*Dlambda[:, 0, :] + Dlambda[:, 0, :]*Dlambda[:, 2, :], axis=1)
            divPhi[:, 4] = np.sum(Dlambda[:, 0, :]*Dlambda[:, 1, :] - Dlambda[:, 1, :]*Dlambda[:, 0, :], axis=1)
            divPhi[:, 5] = np.sum(Dlambda[:, 0, :]*Dlambda[:, 1, :] + Dlambda[:, 1, :]*Dlambda[:, 0, :], axis=1)
            divPhi[:, 0:6:2] *=cell2edgeSign
            divPhi[:, 1:6:2] *=cell2edgeSign
        else:
            #TODO:raise a error
            print("error")

        return divPhi 

    def dual_basis(self, u):
        pass

    def array(self):
        pass

    def value(self, uh, bc):
        pass

    def grad_value(self, uh, bc):
        pass

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        if p == 1:
            return 2*NE
        else:
            #TODO: raise a error
            print("error!")

    def number_of_local_dofs(self):
        p = self.p
        if p==1:
            return 6
        else:
            #TODO: raise a error
            print("error!")

    def edge_to_dof(self):
        p = self.p
        mesh = self.mesh

        NE = mesh.number_of_edges()

        if p == 1:
            edge2dof = np.arange(2*NE).reshape(NE, 2)
        else:
            #TODO: raise error
            print('error!')

        return edge2dof

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell

        N = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()
        edge2dof = self.edge_to_dof()

        cell2edgeSign = mesh.ds.cell_to_edge_sign()
        cell2edge = mesh.ds.cell_to_edge()

        if p == 1:
            cell2dof = np.zeros((NC, ldof), dtype=np.int)

            cell2dof[cell2edgeSign[:, 0], 0:2]= edge2dof[cell2edge[cell2edgeSign[:, 0], 0], :]  
            cell2dof[~cell2edgeSign[:, 0], 0:2]= edge2dof[cell2edge[~cell2edgeSign[:, 0], 0], -1::-1]  

            cell2dof[cell2edgeSign[:, 1], 2:4]= edge2dof[cell2edge[cell2edgeSign[:, 1], 1], :]  
            cell2dof[~cell2edgeSign[:, 1], 2:4]= edge2dof[cell2edge[~cell2edgeSign[:, 1], 1], -1::-1]  

            cell2dof[cell2edgeSign[:, 2], 4:6]= edge2dof[cell2edge[cell2edgeSign[:, 2], 2], :]  
            cell2dof[~cell2edgeSign[:, 2], 4:6]= edge2dof[cell2edge[~cell2edgeSign[:, 2], 2], -1::-1]  

        return cell2dof
