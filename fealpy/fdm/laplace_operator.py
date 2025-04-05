from ..backend import backend_manager as bm
from ..sparse import COOTensor
from ..solver import spsolve, cg

class Laplace:
    def __init__(self, mesh, pde=None):
        """
        @brief a operator to solving the Laplace equation: - ∆u = f

        Parameters:
            mesh: A structured mesh such as UniformMesh1d, UniformMesh2d, UniformMesh3d
            pde: A pde solve class.
        """
        self.device = mesh.device

        self.mesh = mesh
        self.pde = pde
        self.h = mesh.h
        
        if not isinstance(mesh.h, (list, tuple)):
            self.h = [mesh.h]

        self.dimension = mesh.GD

        self.h = self.h + [0] * (3 - len(self.h)) # 标准化 h

        self.nx = getattr(mesh, 'nx', 0) 
        self.ny = getattr(mesh, 'ny', 0) 
        self.nz = getattr(mesh, 'nz', 0)

    def lplace_uh(self, dtype=None):
        """
        @brief: Initialize the solution uh.
        """
        nx = self.nx
        ny = self.ny
        nz = self.nz

        if dtype==None:
            dtype = bm.float64

        uh = bm.zeros((nx + 1)*(ny + 1)*(nz + 1), dtype=dtype, device=self.device)
        return uh

    def assembly(self):
        """
        @brief: Assemble the matrix A.
        """
        n0 = self.nx + 1
        n1 = self.ny + 1
        n2 = self.nz + 1

        hx = self.h[0]
        hy = self.h[1]
        hz = self.h[2]
        
        cx = 1 / (hx ** 2) if hx != 0 else 0
        cy = 1 / (hy ** 2) if hy != 0 else 0
        cz = 1 / (hz ** 2) if hz != 0 else 0

        NN = n0*n1*n2
        
        k = bm.arange(NN, device=self.device).reshape(n0, n1, n2)
        K = bm.arange(NN, device=self.device)

        val_diag = bm.broadcast_to(2 * bm.tensor((cx + cy + cz), dtype=bm.float64, device=self.device), (NN,))
        
        val_x = bm.broadcast_to(bm.tensor(-cx, dtype=bm.float64, device=self.device), (NN - n1 * n2,))
        val_y = bm.broadcast_to(bm.tensor(-cy, dtype=bm.float64, device=self.device), (NN - n0 * n2,))
        val_z = bm.broadcast_to(bm.tensor(-cz, dtype=bm.float64, device=self.device), (NN - n0 * n1,))

        I_x = k[1:, :, :].ravel()
        J_x = k[0:-1, :, :].ravel()
        
        I_y = k[:, 1:, :].ravel()
        J_y = k[:, 0:-1, :].ravel()
        
        I_z = k[:, :, 1:].ravel()
        J_z = k[:, :, 0:-1].ravel()

        col = bm.concat([K, I_x, J_x, I_y, J_y, I_z, J_z], axis=-1)
        row = bm.concat([K, J_x, I_x, J_y, I_y, J_z, I_z], axis=-1)
        values = bm.concat([val_diag, val_x, val_x, val_y, val_y, val_z, val_z], axis=-1) 
        A = COOTensor(bm.stack([col, row]), values, spshape=(NN, NN)).tocsr()

        return A

    def interpolate_node(self, f):
        """
        @brief: Evaluate the function f at all mesh nodes.
        """
        mesh = self.mesh
        node = mesh.node

        return f(node)

    def apply_dirichlet_bc(self, gD, A, f, uh=None, threshold=None):
        mesh = self.mesh
        f = f.reshape(-1, )  
        if uh is None:
            uh = self.lplace_uh().reshape(-1)
        else:
            uh = uh.reshape(-1)  

        node = mesh.node
        if threshold is None:
            isBdNode = mesh.boundary_node_flag()
        elif isinstance(threshold, int):
            isBdNode = (bm.arange(node.shape[0], device=self.device) == threshold)
        elif callable(threshold):
            isBdNode = threshold(node)
        else:
            raise ValueError(f"Invalid threshold: {threshold}")
        
        uh[isBdNode] = gD(node[isBdNode]).reshape(-1)

        f -= A @ uh

        f[isBdNode] = uh[isBdNode]
        
        bdIdx = bm.zeros(A.shape[0], dtype=A.dtype, device=self.device)
        bdIdx[isBdNode] = 1
        I = bm.arange(A.shape[0], dtype=mesh.itype, device=self.device)
        D0 = COOTensor(bm.stack([I,I]), bm.ones(A.shape[0], dtype=A.dtype, device=self.device) - bdIdx).tocsr() #boundary
        D1 = COOTensor(bm.stack([I,I]), bdIdx).tocsr()

        A = D0@A@D0 + D1
        return A, f
    
    
    def error(self, u, uh, errortype='all'):
        """
        @brief  Compute the error between the true solution and the numerical solution.
        """
        hx, hy, hz = self.h[0], self.h[1], self.h[2]
        nx, ny, nz = self.nx, self.ny, self.nz
        mesh = self.mesh
        dimension = self.dimension
        node = mesh.node

        if callable(u):
            uI = u(mesh.node)

        if callable(uh):
            uh = uh(mesh.node)

        uI = uI.ravel()
        uh = uh.ravel()
        e = uI - uh

        nx, ny, nz = (2 if hx == 0 else nx, 2 if hy == 0 else ny, 2 if hz == 0 else nz)
        hx, hy, hz = (1 if h == 0 else h for h in [hx, hy, hz])

        if errortype == 'all':
            emax = bm.max(bm.abs(e))
            e0 = bm.sqrt(hx * hy * hz * bm.sum(e ** 2))
            el_2 = bm.sqrt(1 / ((nx - 1) * (ny - 1) * (nz - 1)) * bm.sum(e ** 2))

            return emax, e0, el_2
        elif errortype == 'max':
            emax = bm.max(bm.abs(e))
            return emax
        elif errortype == 'L2':
            e0 = bm.sqrt(hx * hy * hz * bm.sum(e ** 2))
            return e0
        elif errortype == 'l2':
            el_2 = bm.sqrt(1 / ((nx - 1) * (ny - 1)* (nz - 1))  * bm.sum(e ** 2))
            return el_2
        elif errortype == 'H1':
            e0 = hx * hy *hz* bm.sum(e ** 2)
            if dimension == 1:
                de = e[1:] - e[0:-1]
                e1 = bm.sqrt(bm.sum(de ** 2) / hx + e0 ** 2)
            elif dimension == 2:
                e_ori=e.reshape((nx+1, ny+1))
                diff_ey = (e_ori[1:,1:] - e_ori[1:,:-1])/hx
                diff_ex = (e_ori[1:,1:] - e_ori[:-1,1:])/hy
                h1 = bm.sum(diff_ey**2 + diff_ex**2) * hx * hy
                e1=bm.sqrt(h1+e0)
            elif dimension == 3:
                e_ori=e.reshape((nx+1, ny+1, nz+1))
                diff_ey = (e_ori[1:,1:,1:] - e_ori[1:,:,:-1])/hx
                diff_ex = (e_ori[1:,1:,1:] - e_ori[:-1,1:,1:])/hy
                diff_ez = (e_ori[1:,1:,1:] - e_ori[1:,:,:-1])/hz
                h1 = bm.sum(diff_ey**2 + diff_ex**2 + diff_ez**2) * hx * hy * hz
                e1 = bm.sqrt(h1+e0)
            return e1

    def solve(self, solver='spslove', reshape=False):
            pde = self.pde
            uh = self.lplace_uh()
            A = self.assembly()
            f = self.interpolate_node(pde.source)
            A, f = self.apply_dirichlet_bc(pde.dirichlet, A, f)
            if solver == 'spslove':
                uh[:] = spsolve(A, f, "scipy")
            elif solver == 'cg':
                uh[:]= cg(A, f)

            if reshape:
                return uh.reshape((self.nx+1, self.ny+1, self.nz+1))
            else:
                return uh.flatten()