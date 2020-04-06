import numpy as np
from .function import Function
from .ScaledMonomialSpace2d import ScaledMonomialSpace2d

class RTDof2d:
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p # 默认的空间次数
        self.multiIndex = self.multi_index_matrix() # 默认的多重指标
        self.cell2dof = self.cell_to_dof() # 默认的自由度数组

    def cell_to_dof(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        NE = mesh.
        ldof = self.number_of_local_dofs(p=p)
        cell2dof = np.arange(NC*ldof).reshape(NC, ldof)
        return cell2dof

    def number_of_local_dofs(self):
        p = self.p
        return p*(p+2) 

    def number_of_global_dofs(self):
        p = self.p
        ldof = self.number_of_local_dofs(p=p)
        NC = self.mesh.number_of_cells()
        NE = self.mesh.number_of_edges()
        gdof = NE*p
        if p > 1:
            gdof += NC*(p-1)*p
        return gdof 

class RaviartThomasFiniteElementSpace2d:
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.smspace = ScaledMonomialSpace2d(mesh, p)
        self.p = p

    def basis(self, bc):
        mesh = self.mesh
        p = self.p
        ldof = self.number_of_local_dofs()
        NC = mesh.number_of_cells()
        Rlambda = mesh.rot_lambda()
        cell2edgeSign = self.cell_to_edge_sign()
        shape = bc.shape[:-1] + (NC, ldof, 2)
        phi = np.zeros(shape, dtype=np.float)
        if p == 0:
            phi[..., 0, :] = bc[..., 1, np.newaxis, np.newaxis]*Rlambda[:, 2, :] - bc[..., 2, np.newaxis, np.newaxis]*Rlambda[:, 1, :]
            phi[..., 1, :] = bc[..., 2, np.newaxis, np.newaxis]*Rlambda[:, 0, :] - bc[..., 0, np.newaxis, np.newaxis]*Rlambda[:, 2, :]
            phi[..., 2, :] = bc[..., 0, np.newaxis, np.newaxis]*Rlambda[:, 1, :] - bc[..., 1, np.newaxis, np.newaxis]*Rlambda[:, 0, :]
            phi *= cell2edgeSign.reshape(-1, 3, 1)
        elif p == 1:
            pass
        else:
            raise ValueError('p')

        return phi

    def grad_basis(self, bc):
        mesh = self.mesh
        p = self.p

        ldof = self.number_of_local_dofs()
        NC = mesh.number_of_cells()
        shape = (NC, ldof, 2, 2)
        gradPhi = np.zeros(shape, dtype=np.float)

        cell2edgeSign = self.cell_to_edge_sign()
        W = np.array([[0, 1], [-1, 0]], dtype=np.float)
        Rlambda= mesh.rot_lambda()
        Dlambda = mesh.grad_lambda()
        if p == 0:
            A = np.einsum('...i, ...j->...ij', Rlambda[:, 2, :], Dlambda[:, 1, :])
            B = np.einsum('...i, ...j->...ij', Rlambda[:, 1, :], Dlambda[:, 2, :]) 
            gradPhi[:, 0, :, :] = A - B

            A = np.einsum('...i, ...j->...ij', Rlambda[:, 0, :], Dlambda[:, 2, :])
            B = np.einsum('...i, ...j->...ij', Rlambda[:, 2, :], Dlambda[:, 0, :])
            gradPhi[:, 1, :, :] = A - B

            A = np.einsum('...i, ...j->...ij', Rlambda[:, 1, :], Dlambda[:, 0, :])
            B = np.einsum('...i, ...j->...ij', Rlambda[:, 0, :], Dlambda[:, 1, :])
            gradPhi[:, 2, :, :] = A - B

            gradPhi *= cell2edgeSign.reshape(-1, 3, 1, 1)
        else:
            #TODO:raise a error
            print("error")

        return gradPhi 

    def div_basis(self, bc):
        mesh = self.mesh
        p = self.p

        ldof = self.number_of_local_dofs()
        NC = mesh.number_of_cells()
        divPhi = np.zeros((NC, ldof), dtype=np.float)
        cell2edgeSign = self.cell_to_edge_sign()
        W = np.array([[0, 1], [-1, 0]], dtype=np.float)

        Rlambda = mesh.rot_lambda()
        Dlambda = mesh.grad_lambda()
        if p == 0:
            divPhi[:, 0] = np.sum(Dlambda[:, 1, :]*Rlambda[:, 2, :], axis=1) - np.sum(Dlambda[:, 2, :]*Rlambda[:, 1, :], axis=1)
            divPhi[:, 1] = np.sum(Dlambda[:, 2, :]*Rlambda[:, 0, :], axis=1) - np.sum(Dlambda[:, 0, :]*Rlambda[:, 2, :], axis=1)
            divPhi[:, 2] = np.sum(Dlambda[:, 0, :]*Rlambda[:, 1, :], axis=1) - np.sum(Dlambda[:, 1, :]*Rlambda[:, 0, :], axis=1)
            divPhi *= cell2edgeSign
        else:
            #TODO:raise a error
            print("error")

        return divPhi

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell

        N = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()
        if p == 0:
            cell2dof = mesh.ds.cell_to_edge()
        else:
            #TODO: raise a error 
            print('error!')

        return cell2dof

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
            print("error!")

    def value(self, uh, bc, cellidx=None):
        phi = self.basis(bc)
        cell2dof = self.dof.cell2dof
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        if cellidx is None:
            val = np.einsum(s1, phi, uh[cell2dof])
        else:
            val = np.einsum(s1, phi, uh[cell2dof[cellidx]])
        return val

    def grad_value(self, uh, bc, cellidx=None):
        gphi = self.grad_basis(bc, cellidx=cellidx)
        cell2dof = self.dof.cell2dof
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijmn, ij{}->...i{}mn'.format(s0[:dim], s0[:dim])
        if cellidx is None:
            val = np.einsum(s1, gphi, uh[cell2dof])
        else:
            val = np.einsum(s1, gphi, uh[cell2dof[cellidx]])
        return val

    def div_value(self, uh, bc, cellidx=None):
        val = self.grad_value(uh, bc, cellidx=None)
        return val.trace(axis1=-2, axis2=-1)

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def interpolation(self, u, returnfun=False):
        mesh = self.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        NE = mesh.number_of_edges()
        n = mesh.edge_unit_normal()
        l = mesh.entity_measure('edge')

        qf = IntervalQuadrature(3)
        bcs, ws = qf.get_quadrature_points_and_weights()
        points = np.einsum('kj, ijm->kim', bcs, node[edge])
        val = u(points)
        uh = np.einsum('k, kim, im, i->i', ws, val, n, l)

        if returnfun is True:
            return Function(self, array=uh)
        else:
            return uh

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=np.float)
