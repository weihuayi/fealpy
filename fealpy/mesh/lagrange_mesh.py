import numpy as np

from .mesh_base import Mesh
from .mesh_data_structure import Mesh2dDataStructure

class LagrangeMesh(Mesh):
    def ref_cell_measure(self):
        raise NotImplementedError
    
    def shape_function(self, bc, p=None, index=np.s_[:]):
        raise NotImplementedError

    def grad_shape_function(self, bc, p=None, index=np.s_[:]):
        raise NotImplementedError
    
    def number_of_corner_nodes(self):
        """
        @berif 该函数返回角点节点的个数
        -----

        拉格朗日三角形网格中的节点分为单元角点节点，边内部节节点和单元内部节点。

        这些节点默认的编号顺序也是：角点节点，边内部节点，单元内部节点。
        """
        return self.ds.NCN

    def jacobi_matrix(self, bc, p=None, index=np.s_[:], return_grad=False):
        """
        @brief 计算参考单元 u 到实际实际单元 Jacobi 矩阵。

        x(xi, eta) = phi_0 x_0 + phi_1 x_1 + ... + phi_{ldof-1} x_{ldof-1}
        """
        p = self.p if p is None else p

        if isinstance(bc, tuple):
            TD = len(bc)
        elif isinstance(bc, np.ndarray):
            TD = bc.shape[-1] - 1
        else:
            raise ValueError(' `bc` should be a tuple or ndarray!')

        entity = self.entity(etype=TD)[index]
        gphi = self.grad_shape_function(bc, p=p)
        J = np.einsum(
                'cin, ...cim->...cnm',
                self.node[entity, :], gphi) #(NC,ldof,GD),(NQ,NC,ldof,TD)
        if return_grad is False:
            return J #(NQ,NC,GD,TD)
        else:
            return J, gphi

    def unit_normal(self, bc, index=np.s_[:]):
        """
        @brief 计算曲面情况下，积分点处的单位法线
        """
        J = self.jacobi_matrix(bc, index=index)

        n = np.cross(J[..., 0], J[..., 1], axis=-1)
        
        if self.GD == 3:
            l = np.sqrt(np.sum(n**2, axis=-1, keepdims=True))
            n /=l

        return n

    def first_fundamental_form(self, bc, index=np.s_[:], return_jacobi=False, return_grad=False):
        
        """
        @berif  计算网格曲面在积分点处的第一基本形式。
        """
        if isinstance(bc, tuple):
            TD = len(bc)
        elif isinstance(bc, np.ndarray):
            TD = bc.shape[-1] - 1
        else:
            raise ValueError(' `bc` should be a tuple or ndarray!')

        J = self.jacobi_matrix(bc, index=index,
                return_grad=return_grad)
        
        if return_grad:
            J, gphi = J

        shape = J.shape[0:-2] + (TD, TD)
        G = np.zeros(shape, dtype=self.ftype)
        for i in range(TD):
            G[..., i, i] = np.sum(J[..., i]**2, axis=-1)
            for j in range(i+1, TD):
                G[..., i, j] = np.sum(J[..., i]*J[..., j], axis=-1)
                G[..., j, i] = G[..., i, j]
        if (return_jacobi is False) & (return_grad is False):
            return G
        elif (return_jacobi is True) & (return_grad is False): 
            return G, J
        elif (return_jacobi is False) & (return_grad is True): 
            return G, gphi 
        else:
            return G, J, gphi

    def second_fundamental_form(self, bc, index=np.s_[:], return_jacobi=False, return_grad=False):
        """
        @berif 计算网格曲面在积分点处的第二基本形式。
        """
        if isinstance(bc, tuple):
            TD = len(bc)
        elif isinstance(bc, np.ndarray):
            TD = bc.shape[-1] - 1
        else:
            raise ValueError(' `bc` should be a tuple or ndarray!')

    def edge_length(self, q=None, index=np.s_[:]):
        """
        @berif 计算边的长度
        """
        p = self.p
        q = p if q is None else q

        qf = self.integrator(q, etype='edge')
        bcs, ws = qf.get_quadrature_points_and_weights() 

        J = self.jacobi_matrix(bcs, index=index)
        l = np.sqrt(np.sum(J**2, axis=(-1, -2)))
        a = np.einsum('i, ij->j', ws, l)
        return a

    def vtk_cell_type(self, etype='cell'):
        """
        @berif 返回网格单元对应的 vtk 类型。
        """
        raise NotImplementedError

    def to_vtk(self, etype='cell', index=np.s_[:], fname=None):
        """
        Parameters
        ----------

        @berif 把网格转化为 VTK 的格式
        """
        from .vtk_extent import vtk_cell_index, write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()
        if GD == 2:
            node = np.concatenate((node, np.zeros((node.shape[0], 1), dtype=self.ftype)), axis=1)

        cell = self.entity(etype)[index]
        cellType = self.vtk_cell_type(etype)
        idx = vtk_cell_index(self.p, cellType)
        NV = cell.shape[-1]

        cell = np.r_['1', np.zeros((len(cell), 1), dtype=cell.dtype), cell[:, idx]]
        cell[:, 0] = NV

        NC = len(cell)
        if fname is None:
            return node, cell.flatten(), cellType, NC 
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                    nodedata=self.nodedata,
                    celldata=self.celldata)


