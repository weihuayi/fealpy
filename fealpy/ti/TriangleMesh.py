import taichi as ti
import numpy as np
from scipy.sparse import csr_matrix
from fealpy.symcom  import LagrangeFEMSpace
from fealpy.ti.TriangleMeshData import phiphi,gphigphi,gphiphi
def construct_edge(cell):
    """ 
    """
    NC =  cell.shape[0] 
    NEC = 3 
    NVE = 2 

    localEdge = np.array([(1, 2), (2, 0), (0, 1)])
    totalEdge = cell[:, localEdge].reshape(-1, 2)
    _, i0, j = np.unique(np.sort(totalEdge, axis=-1),
            return_index=True,
            return_inverse=True,
            axis=0)
    NE = i0.shape[0]
    edge2cell = np.zeros((NE, 4), dtype=np.int_)

    i1 = np.zeros(NE, dtype=np.int_)
    i1[j] = np.arange(NEC*NC, dtype=np.int_)

    edge2cell[:, 0] = i0//3
    edge2cell[:, 1] = i1//3
    edge2cell[:, 2] = i0%3
    edge2cell[:, 3] = i1%3

    edge = totalEdge[i0, :]
    cell2edge = np.reshape(j, (NC, 3))
    return edge, edge2cell, cell2edge


@ti.data_oriented
class TriangleMesh():
    def __init__(self, node, cell, itype=ti.u32, ftype=ti.f64):
        assert cell.shape[-1] == 3

        self.itype = itype
        self.ftype = ftype

        NN = node.shape[0]
        GD = node.shape[1]
        self.node = ti.field(self.ftype, (NN, GD))
        self.node.from_numpy(node)

        self.cellspace = LagrangeFEMSpace(GD)
        
        NC = cell.shape[0]
        self.cell = ti.field(self.itype, shape=(NC, 3))
        self.cell.from_numpy(cell)

        self.glambda = ti.field(self.ftype, shape=(NC, 3, GD)) 
        self.cellmeasure = ti.field(self.ftype, shape=(NC, ))
        self.init_grad_lambdas()

        self.construct_data_structure(cell)

    def construct_data_structure(self, cell):

        edge, edge2cell, cell2edge = construct_edge(cell)
        NE = edge.shape[0]

        self.edge = ti.field(self.itype, shape=(NE, 2))
        self.edge.from_numpy(edge)

        self.edge2cell = ti.field(self.itype, shape=(NE, 4))
        self.edge2cell.from_numpy(edge2cell)

        NC = self.number_of_cells()
        self.cell2edge = ti.field(self.itype, shape=(NC, 3))
        self.cell2edge.from_numpy(cell2edge)

    def multi_index_matrix(self, p):
        ldof = (p+1)*(p+2)//2
        idx = np.arange(0, ldof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        multiIndex = np.zeros((ldof, 3), dtype=np.int_)
        multiIndex[:, 2] = idx - idx0*(idx0 + 1)/2
        multiIndex[:, 1] = idx0 - multiIndex[:,2]
        multiIndex[:, 0] = p - multiIndex[:, 1] - multiIndex[:, 2]
        return multiIndex

    def number_of_local_interpolation_points(self, p):
        return (p+1)*(p+2)//2

    def number_of_global_interpolation_points(self, p):
        NP = self.number_of_nodes()
        if p > 1:
            NE = self.number_of_edges()
            NP += (p-1)*NE
        if p > 2:
            NC = self.number_of_cells()
            NP += (p-2)*(p-1)*NC//2
        return NP

    @ti.kernel 
    def interpolation_points(self, p: ti.u32, ipoints: ti.template()):
        """
        @brief 生成三角形网格上 p 次的插值点
        """
        NN = self.node.shape[0]
        NE = self.edge.shape[0]
        NC = self.cell.shape[0]
        GD = self.node.shape[1]
        for n in range(NN):
            for d in range(GD):
                ipoints[n, d] = self.node[n, d]
        if p > 1:
            for e in range(NE):
                s1 = NN + e*(p-1)
                for i1 in range(1, p):
                    i0 = p - i1 # (i0, i1)
                    I = s1 + i1 - 1
                    for d in range(GD):
                        ipoints[I, d] = (
                                i0*self.node[self.edge[e, 0], d] + 
                                i1*self.node[self.edge[e, 1], d])/p
        if p > 2:
            cdof = (p-2)*(p-1)//2
            s0 = NN + (p-1)*NE
            for c in range(NC):
                i0 = p-2
                s1 = s0 + c*cdof
                for level in range(0, p-2):
                    i0 = p - 2 - level
                    for i2 in range(1, level+2):
                        i1 = p - i0 - i2 #(i0, i1, i2)
                        j = i1 + i2 - 2
                        I = s1 + j*(j+1)//2 + i2 - 1  
                        for d in range(GD):
                            ipoints[I, d] = (
                                    i0*self.node[self.cell[c, 0], d] + 
                                    i1*self.node[self.cell[c, 1], d] + 
                                    i2*self.node[self.cell[c, 2], d])/p
                        

    @ti.kernel
    def edge_to_ipoint(self, p: ti.u32, edge2ipoint: ti.template()):
        """
        @brief 返回每个边上对应 p 次插值点的全局编号
        """
        for i in range(self.edge.shape[0]):
            edge2dof[i, 0] = self.edge[i, 0]
            edge2dof[i, p] = self.edge[i, 1]
            for j in ti.static(range(1, p)):
                edge2dof[i, j] = self.node.shape[0] + i*(p-1) + j - 1

    @ti.kernel
    def cell_to_ipoint(self, p: ti.i32, cell2ipoint: ti.template()):
        """
        @brief 返回每个单元上对应 p 次插值点的全局编号
        """
        cdof = (p+1)*(p+2)//2 
        NN = self.node.shape[0]
        NE = self.edge.shape[0]
        for c in range(self.cell.shape[0]): 
            # 三个顶点 
            cell2ipoint[c, 0] = self.cell[c, 0]
            cell2ipoint[c, cdof - p - 1] = self.cell[c, 1] # 不支持负数索引
            cell2ipoint[c, cdof - 1] = self.cell[c, 2]

            # 第 0 条边
            e = self.cell2edge[c, 0]
            v0 = self.edge[e, 0]
            s0 = NN + e*(p-1)
            s1 = cdof - p
            if v0 == self.cell[c, 1]:
                for i in range(0, p-1):
                    cell2ipoint[c, s1] = s0 + i
                    s1 += 1
            else:
                for i in range(0, p-1):
                    cell2ipoint[c, s1] = s0 + p - 2 - i
                    s1 += 1

            # 第 1 条边
            e = self.cell2edge[c, 1]
            v0 = self.edge[e, 0]
            s0 = NN + e*(p-1)
            s1 = 2
            if v0 == self.cell[c, 0]:
                for i in range(0, p-1):
                    cell2ipoint[c, s1] = s0 + i 
                    s1 += i + 3
            else:
                for i in range(0, p-1):
                    cell2ipoint[c, s1] = s0 + p - 2 - i 
                    s1 += i + 3 

            # 第 2 条边
            e = self.cell2edge[c, 2]
            v0 = self.edge[e, 0]
            s0 = NN + e*(p-1)
            s1 = 1
            if v0 == self.cell[c, 0]:
                for i in range(0, p-1):
                    cell2ipoint[c, s1] = s0 + i 
                    s1 += i + 2
            else:
                for i in range(0, p-1):
                    cell2ipoint[c, s1] = s0 + p - 2 - i 
                    s1 += i + 2 

            # 内部点
            if p >= 3:
                level = p - 2 
                s0 = NN + (p-1)*NE + c*(p-2)*(p-1)//2
                s1 = 4
                s2 = 0
                for l in range(0, level):
                    for i in range(0, l+1):
                        cell2ipoint[c, s1] = s0 + s2 
                        s1 += 1
                        s2 += 1 
                    s1 += 2

    def number_of_nodes(self):
        return self.node.shape[0]

    def number_of_edges(self):
        return self.edge.shape[0]

    def number_of_cells(self):
        return self.cell.shape[0]

    def geo_dimension(self):
        return self.node.shape[1]

    def top_dimension(self):
        return 2 

    def entity(self, etype=2):
        if etype in {'cell', 2}:
            return self.cell
        elif etype in {'edge', 'face', 1}:
            return self.edge
        elif etype in {'node', 0}:
            return self.node
        else:
            raise ValueError("`etype` is wrong!")

    def vtk_cell_type(self, etype='cell'):
        if etype in {'cell', 2}:
            VTK_TRIANGLE = 5
            return VTK_TRIANGLE
        elif etype in {'face', 'edge', 1}:
            VTK_LINE = 3
            return VTK_LINE

    def to_vtk(self, fname, nodedata=None, celldata=None):
        """
        Parameters
        ----------

        Notes
        -----
        把网格转化为 VTK 的格式
        """
        from fealpy.mesh.vtk_extent import vtk_cell_index, write_to_vtu

        node = self.node.to_numpy()
        GD = self.geo_dimension()
        if GD == 2:
            node = np.concatenate((node, np.zeros((node.shape[0], 1))), axis=1)

        cell = self.cell.to_numpy(dtype=np.int_)
        cellType = self.vtk_cell_type()
        NV = cell.shape[-1]

        cell = np.r_['1', np.zeros((len(cell), 1), dtype=cell.dtype), cell]
        cell[:, 0] = NV

        NC = len(cell)
        print("Writting to vtk...")
        write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                nodedata=nodedata,
                celldata=celldata)

    @ti.kernel
    def init_grad_lambdas(self):
        """
        @brief 初始化网格中每个单元上重心坐标函数的梯度，以及单元的面积
        """
        assert self.node.shape[1] == 2

        for i in range(self.cell.shape[0]):
            x0 = self.node[self.cell[i, 0], 0]
            y0 = self.node[self.cell[i, 0], 1]

            x1 = self.node[self.cell[i, 1], 0]
            y1 = self.node[self.cell[i, 1], 1]

            x2 = self.node[self.cell[i, 2], 0]
            y2 = self.node[self.cell[i, 2], 1]

            l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) 

            self.cellmeasure[i] = 0.5*l
            self.glambda[i, 0, 0] = (y1 - y2)/l
            self.glambda[i, 0, 1] = (x2 - x1)/l 
            self.glambda[i, 1, 0] = (y2 - y0)/l 
            self.glambda[i, 1, 1] = (x0 - x2)/l
            self.glambda[i, 2, 0] = (y0 - y1)/l
            self.glambda[i, 2, 1] = (x1 - x0)/l
        

    @ti.func
    def cell_measure(self, i: ti.u32) -> ti.f64:
        x0 = self.node[self.cell[i, 0], 0]
        y0 = self.node[self.cell[i, 0], 1]

        x1 = self.node[self.cell[i, 1], 0]
        y1 = self.node[self.cell[i, 1], 1]

        x2 = self.node[self.cell[i, 2], 0]
        y2 = self.node[self.cell[i, 2], 1]

        l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) 
        l *= 0.5
        return l

    
    @ti.func
    def grad_lambda(self, i: ti.u32) -> (ti.types.matrix(3, 2, ti.f64), ti.f64):
        """
        计算第 i 个单元上重心坐标函数的梯度，以及单元的面积
        """

        assert self.node.shape[1] == 2

        x0 = self.node[self.cell[i, 0], 0]
        y0 = self.node[self.cell[i, 0], 1]

        x1 = self.node[self.cell[i, 1], 0]
        y1 = self.node[self.cell[i, 1], 1]

        x2 = self.node[self.cell[i, 2], 0]
        y2 = self.node[self.cell[i, 2], 1]

        l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) 

        gphi = ti.Matrix.zero(ti.f64, 3, 2)
        gphi[0, 0] = (y1 - y2)/l
        gphi[0, 1] = (x2 - x1)/l 
        gphi[1, 0] = (y2 - y0)/l 
        gphi[1, 1] = (x0 - x2)/l
        gphi[2, 0] = (y0 - y1)/l
        gphi[2, 1] = (x1 - x0)/l

        l *= 0.5
        return gphi, l

    @ti.func
    def surface_grad_lambda(self, i: ti.u32) -> (ti.types.matrix(3, 2, ti.f64), ti.f64):
        """
        计算第 i 个单元上重心坐标函数的梯度，以及单元的面积
        """
        assert self.node.shape[1] == 3

        x0 = self.node[self.cell[i, 0], 0]
        y0 = self.node[self.cell[i, 0], 1]
        z0 = self.node[self.cell[i, 0], 2]

        x1 = self.node[self.cell[i, 1], 0]
        y1 = self.node[self.cell[i, 1], 1]
        z1 = self.node[self.cell[i, 0], 2]

        x2 = self.node[self.cell[i, 2], 0]
        y2 = self.node[self.cell[i, 2], 1]
        z2 = self.node[self.cell[i, 0], 2]

        gphi = ti.Matrix.zero(ti.f64, 3, 3)
        return grad, l

    @ti.kernel
    def cell_stiff_matrices_1(self, S: ti.template()):
        """
        @brief 计算网格上的所有单元上的线性元刚度矩阵
        """
        for c in range(self.cell.shape[0]):
            gphi, cm = self.grad_lambda(c) 

            S[c, 0, 0] = cm*(gphi[0, 0]*gphi[0, 0] + gphi[0, 1]*gphi[0, 1])
            S[c, 0, 1] = cm*(gphi[0, 0]*gphi[1, 0] + gphi[0, 1]*gphi[1, 1])
            S[c, 0, 2] = cm*(gphi[0, 0]*gphi[2, 0] + gphi[0, 1]*gphi[2, 1])

            S[c, 1, 0] = S[c, 0, 1]
            S[c, 1, 1] = cm*(gphi[1, 0]*gphi[1, 0] + gphi[1, 1]*gphi[1, 1])
            S[c, 1, 2] = cm*(gphi[1, 0]*gphi[2, 0] + gphi[1, 1]*gphi[2, 1])

            S[c, 2, 0] = S[c, 0, 2]
            S[c, 2, 1] = S[c, 1, 2]
            S[c, 2, 2] = cm*(gphi[2, 0]*gphi[2, 0] + gphi[2, 1]*gphi[2, 1])



    @ti.kernel
    def cell_mass_matrices_11(self, S: ti.template()):
        """
        @brief 计算网格上的所有单元上的线性元的质量矩阵
        """        
        for c in range(self.cell.shape[0]):
            cm = self.cell_measure(c)
            c0 = cm/6.0
            c1 = cm/12.0

            S[c, 0, 0] = c0 
            S[c, 0, 1] = c1
            S[c, 0, 2] = c1

            S[c, 1, 0] = c1 
            S[c, 1, 1] = c0  
            S[c, 1, 2] = c1

            S[c, 2, 0] = c1 
            S[c, 2, 1] = c1 
            S[c, 2, 2] = c0

    def cell_to_dof(self, p):
        NC = self.number_of_cells()
        ldof = self.number_of_local_interpolation_points(p)
        cell2ipoint = ti.field(self.itype, shape=(NC, ldof))
        self.cell_to_ipoint(p, cell2ipoint)
        cell2dof = cell2ipoint.to_numpy()
        return cell2dof
    '''
    def cell_mass_matrix(self, p1, p2, c=None):
        index = str(p1)+str(p2)
        cellmeasure = self.cellmeasure.to_numpy()
        c2f = self.cell_to_dof(p1)
        NC = self.number_of_cells()
        if c == None:
            val = phiphi[index]
            val = np.einsum('cij,c->cij', val, cellmeasure)
        else:
            pass
            #val = phiphiphi[index]
            #coff = c[c2f]
            #val = np.einsum('cijk,c,ci->cjk', val, cellmeasure, coff)
        return val
    '''

    def cell_mass_matrix(self, p1, p2, c=None):
        index = str(p1)+str(p2)
        return phiphi[index]


    def cell_stiff_matrix(self, p1, p2):
        index = str(p1)+str(p2)
        A = gphigphi[index]
        B = self.glambda.to_numpy()
        val = np.einsum('ijkl, ckm, clm->cij', A, B ,B)
        return val 
    
    def cell_gphix_phi_matrix(self, p1, p2):
        index = str(p1)+str(p2)
        A = gphiphi[index]
        B = self.glambda.to_numpy()
        gx = np.einsum('ijk, ck ->cij', A, B[...,0])
        return gy
    
    def cell_gphiy_phi_matrix(self, p1, p2):
        index = str(p1)+str(p2)
        A = gphiphi[index]
        B = self.glambda.to_numpy()
        gy = np.einsum('ijk, ck ->cij', A, B[...,1])
        return gy
    
    
    def cell_gphix_gphix_matrix(self, p1, p2):
        index = str(p1)+str(p2)
        A = gphigphi[index]
        B = self.glambda.to_numpy()
        val = np.einsum('ijkl, ck, cl->cij', A, B[...,0] ,B[...,0])
        return val 
    
    def cell_gphix_gphiy_matrix(self, p1, p2):
        index = str(p1)+str(p2)
        A = gphigphi[index]
        B = self.glambda.to_numpy()
        val = np.einsum('ijkl, ck, cl->cij', A, B[...,0] ,B[...,1])
        return val 
    
    def cell_gphiy_gphix_matrix(self, p1, p2):
        index = str(p1)+str(p2)
        A = gphigphi[index]
        B = self.glambda.to_numpy()
        val = np.einsum('ijkl, ck, cl->cij', A, B[...,1] ,B[...,0])
        return val 
    
    def cell_gphiy_gphiy_matrix(self, p1, p2):
        index = str(p1)+str(p2)
        A = gphigphi[index]
        B = self.glambda.to_numpy()
        val = np.einsum('ijkl, ck, cl->cij', A, B[...,1], B[...,1])
        return val 

    '''
    def cell_mass_matrix(self, p1, p2):
        return self.cellspace.mass_matrix(p1, p2)

    def cell_stiff_matrices(self, p1, p2):
        A = np.array(self.cellspace.stiff_matrix(p1,p2).tolist(),dtype=np.float64)
        B = self.glambda.to_numpy()
        val = np.einsum('ijkl, ckm, clm->cij', A, B ,B)
        return val
    
    def cell_gphi_phi_matrices(self, p1, p2):
        A = np.array(self.cellspace.gphi_phi_matrix(p1,p2).tolist(),dtype=np.float64)
        B = self.glambda.to_numpy()
        gx = np.einsum('ijk, ck ->cij', A, B[...,0])
        gy = np.einsum('ijk, ck ->cij', A, B[...,1])
        return gx,gy
    '''
    
    def source_mass_vector(self, p1, p2 ,c):
        index = str(p1)+str(p2)
        val = phiphi[index]
        c2f1 = self.cell_to_dof(p1)
        c2f2 = self.cell_to_dof(p2)
        gdof = self.number_of_global_interpolation_points(p2)
        NC = self.number_of_cells()
        cellmeasure = self.cellmeasure.to_numpy()
        val = np.einsum('cij, ci, c -> cj', val, c[c2f1], cellmeasure)
        result = np.zeros((gdof))
        np.add.at(result, c2f2, val)
        return result
    
    def source_gphix_phi_vector(self, p1, p2 ,c):
        index = str(p1)+str(p2)
        val = gphiphi[index]
        c2f1 = self.cell_to_dof(p1)
        c2f2 = self.cell_to_dof(p2)
        gdof = self.number_of_global_interpolation_points(p1)
        NC = self.number_of_cells()
        cellmeasure = self.cellmeasure.to_numpy()
        B = self.glambda.to_numpy()
        val = np.einsum('ijk,ck,cj,c -> ci', val,B[...,0], c[c2f2], cellmeasure)
        result = np.zeros((gdof))
        np.add.at(result, c2f1, val)
        return result
    
    def source_gphixx_phi_vector(self, p1, p2 ,c):
        index = str(p1)+str(p2)
        val = gphiphi[index]
        c2f1 = self.cell_to_dof(p1)
        c2f2 = self.cell_to_dof(p2)
        gdof = self.number_of_global_interpolation_points(p2)
        NC = self.number_of_cells()
        cellmeasure = self.cellmeasure.to_numpy()
        B = self.glambda.to_numpy()
        val = np.einsum('ijk,ck,ci,c -> cj', val,B[...,0], c[c2f1], cellmeasure)
        result = np.zeros((gdof))
        np.add.at(result, c2f2, val)
        return result
    
    def source_gphiyy_phi_vector(self, p1, p2 ,c):
        index = str(p1)+str(p2)
        val = gphiphi[index]
        c2f1 = self.cell_to_dof(p1)
        c2f2 = self.cell_to_dof(p2)
        gdof = self.number_of_global_interpolation_points(p2)
        NC = self.number_of_cells()
        cellmeasure = self.cellmeasure.to_numpy()
        B = self.glambda.to_numpy()
        val = np.einsum('ijk,ck,ci,c -> cj', val,B[...,1], c[c2f1], cellmeasure)
        result = np.zeros((gdof))
        np.add.at(result, c2f2, val)
        return result
    
    def source_gphiy_phi_vector(self, p1, p2 ,c):
        index = str(p1)+str(p2)
        val = gphiphi[index]
        c2f1 = self.cell_to_dof(p1)
        c2f2 = self.cell_to_dof(p2)
        gdof = self.number_of_global_interpolation_points(p1)
        NC = self.number_of_cells()
        cellmeasure = self.cellmeasure.to_numpy()
        B = self.glambda.to_numpy()
        val = np.einsum('ijk,ck,cj,c -> ci', val,B[...,1], c[c2f2], cellmeasure)
        result = np.zeros((gdof))
        np.add.at(result, c2f1, val)
        return result

    def construct_matrix(self, p1, p2 ,matrixtype='mass'):
        '''
        '''
        if matrixtype == 'mass' :
            m = self.cell_mass_matrix(p1,p2)
        elif matrixtype == 'stiff':
            m = self.cell_stiff_matrix(p1,p2)
        elif matrixtype == 'gpx_gpx':
            m = self.cell_gphix_gphix_matrix(p1,p2)
        elif matrixtype == 'gpx_gpy':
            m = self.cell_gphix_gphiy_matrix(p1,p2)
        elif matrixtype == 'gpy_gpx':
            m = self.cell_gphiy_gphix_matrix(p1,p2)
        elif matrixtype == 'gpy_gpy':
            m = self.cell_gphiy_gphiy_matrix(p1,p2)
        elif matrixtype == 'gpx_p':
            m = self.cell_gphi_phi_matrix(p1,p2)
        elif matrixtype == 'gpy_p':
            m = self.cell_gphi_phi_matrix(p1,p2)
        NC = self.number_of_cells()
        cellmeasure = self.cellmeasure.to_numpy()
        m = np.einsum('cij,c->cij', m, cellmeasure)
        ldof1 = m.shape[-2]
        ldof2 = m.shape[-1]
        cell2dof1 = self.cell_to_dof(p1)
        cell2dof2 = self.cell_to_dof(p2)
        I = np.broadcast_to(cell2dof1[:, :, None], shape = m.shape)
        J = np.broadcast_to(cell2dof2[:, None, :], shape = m.shape)
        gdof1 = self.number_of_global_interpolation_points(p1)
        gdof2 = self.number_of_global_interpolation_points(p2)
        val = csr_matrix((m.flat, (I.flat, J.flat)), shape=(gdof1, gdof2))
        return val
    #'''  
    @ti.kernel
    def cell_convection_matrices_1(self, u: ti.template(), S: ti.template()):
        """
        @brief 计算网格上所有单元上的线性元对流矩阵

        (\\boldsymbol u \\cdot \\nabla \\phi, w)， 水平集函数中会用到
        """
        for c in range(self.cell.shape[0]):
            gphi, cm = self.grad_lambda(c) 

            c0 = cm/6.0
            c1 = cm/12.0

            U = ti.Matrix.zero(ti.f64, 3, 2)
            for i in ti.static(range(2)):
                U[0, i] += u[self.cell[c, 0], i]*c0 
                U[0, i] += u[self.cell[c, 1], i]*c1 
                U[0, i] += u[self.cell[c, 2], i]*c1 

            for i in ti.static(range(2)):
                U[1, i] += u[self.cell[c, 0], i]*c1 
                U[1, i] += u[self.cell[c, 1], i]*c0 
                U[1, i] += u[self.cell[c, 2], i]*c1 

            for i in ti.static(range(2)):
                U[2, i] += u[self.cell[c, 0], i]*c1 
                U[2, i] += u[self.cell[c, 1], i]*c1 
                U[2, i] += u[self.cell[c, 2], i]*c0 

            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    S[c, i, j] = U[i, 0]*gphi[j, 0] + U[i, 1]*gphi[j, 1]

    @ti.kernel
    def cell_convection_matrices_2(self, u: ti.template(), S: ti.template()):
        """
        @brief 计算网格上所有单元上的二次元对流矩阵

        (\\boldsymbol u \\cdot \\nabla \\phi, w)， 水平集函数中会用到
        """
        for c in range(self.cell.shape[0]):
            gphi, cm = self.grad_lambda(c) 

            c0 = cm/6.0
            c1 = cm/12.0

            U = ti.Matrix.zero(ti.f64, 3, 2)
            for i in ti.static(range(2)):
                U[0, i] += u[self.cell[c, 0], i]*c0 
                U[0, i] += u[self.cell[c, 1], i]*c1 
                U[0, i] += u[self.cell[c, 2], i]*c1 

            for i in ti.static(range(2)):
                U[1, i] += u[self.cell[c, 0], i]*c1 
                U[1, i] += u[self.cell[c, 1], i]*c0 
                U[1, i] += u[self.cell[c, 2], i]*c1 

            for i in ti.static(range(2)):
                U[2, i] += u[self.cell[c, 0], i]*c1 
                U[2, i] += u[self.cell[c, 1], i]*c1 
                U[2, i] += u[self.cell[c, 2], i]*c0 

            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    S[c, i, j] = U[i, 0]*gphi[j, 0] + U[i, 1]*gphi[j, 1]


   
    @ti.kernel
    def cell_source_vectors_1(self, f:ti.template(), bc:ti.template(), ws:ti.template(), F:ti.template()):
        """
        @brief 计算所有单元上的线性元载荷
        """
        for c in range(self.cell.shape[0]):
            x0 = self.node[self.cell[c, 0], 0]
            y0 = self.node[self.cell[c, 0], 1]

            x1 = self.node[self.cell[c, 1], 0]
            y1 = self.node[self.cell[c, 1], 1]

            x2 = self.node[self.cell[c, 2], 0]
            y2 = self.node[self.cell[c, 2], 1]
            l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) 
            l *= 0.5
            for q in ti.static(range(bc.n)):
                x = x0*bc[q, 0] + x1*bc[q, 1] + x2*bc[q, 1]
                y = y0*bc[q, 0] + y1*bc[q, 1] + y2*bc[q, 1]
                z = f(x, y)
                for i in ti.static(range(3)):
                    F[c, i] += ws[q]*bc[q, i]*z

            for i in range(3):
                F[c, i] *= l

    @ti.kernel
    def ti_cell_stiff_matrices(self, K: ti.types.sparse_matrix_builder()):
        """
        
        """
        for c in range(self.cell.shape[0]):
            gphi, cm = self.grad_lambda(c) 
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    I = self.cell[c, i]
                    J = self.cell[c, j]
                    K[I, J] += cm*(gphi[i, 0]*gphi[j, 0] + gphi[i, 1]*gphi[j, 1]) 

    def ti_stiff_matrix(self, c=None):
        """
        基于 Taichi 组装刚度矩阵
        """
        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        K = ti.linalg.SparseMatrixBuilder(NN, NN, max_num_triplets=9*NC)
        self.ti_cell_stiff_matrices(K)
        A = K.build()
        return A

    def stiff_matrix(self, p=1, c=None):
        """
        组装总体刚度矩阵
        """
        NC = self.number_of_cells()
        if p == 1:
            K = ti.field(ti.f64, (NC, 3, 3))
            self.cell_stiff_matrices_1(K)
        elif p == 2:
            K = ti.field(ti.f64, (NC, 6, 6))
            self.cell_stiff_matrices_2(K)

        M = K.to_numpy()
        if c is not None:
            M *= c # 目前假设 c 为一常数

        if p == 1:
            cell2dof = self.cell.to_numpy()
            gdof = NN 
        elif p == 2:
            cell2ipoint = ti.field(self.itype, shape=(NC, 6))
            self.cell_to_ipoint(p, cell2ipoint)
            cell2dof = cell2ipoint.to_numpy()

        I = np.broadcast_to(cell2dof[:, :, None], shape=M.shape)
        J = np.broadcast_to(cell2dof[:, None, :], shape=M.shape)

        gdof = self.number_of_global_interpolation_points()
        M = csr_matrix((K.to_numpy().flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return M
    
    def source_vector(self, f, p=1):
        """
        组装总体载荷向量
        """
        NN = self.node.shape[0]
        NC = self.cell.shape[0]
        bc = ti.Matrix([
            [0.6666666666666670,	0.1666666666666670,     0.1666666666666670],
            [0.1666666666666670,	0.6666666666666670,     0.1666666666666670],
            [0.1666666666666670,	0.1666666666666670,     0.6666666666666670]], dt=ti.f64)
        ws = ti.Vector([0.3333333333333330, 0.3333333333333330, 0.3333333333333330], dt=ti.f64)

        F = ti.field(ti.f64, (NC, 3))
        self.cell_source_vectors_1(f, bc, ws, F)

        bb = F.to_numpy()
        F = np.zeros(NN, dtype=np.float64)
        cell = self.cell.to_numpy()
        np.add.at(F, cell, bb)
        return F

    @ti.kernel
    def scalar_linear_interpolation(self, f: ti.template(), R: ti.template()):
        """! 定义在二维平面上的标量函数的线性插值
        """
        for i in range(self.node.shape[0]):
            R[i] = f(self.node[i, 0], self.node[i, 1])

    @ti.kernel
    def vector_linear_interpolation(self, f: ti.template(), R: ti.template()):
        """! 定义在二维平面上的向量函数的线性插值
        """
        for i in range(self.node.shape[0]):
            R[i, 0], R[i, 1] = f(self.node[i, 0], self.node[i, 1])

    @ti.kernel
    def surface_linear_scalar_interpolation(self, f: ti.template(), R: ti.template()):
        """! 定义在二维曲面上的标量函数的线性插值
        """
        for i in range(self.node.shape[0]):
            R[i] = f(self.node[i, 0], self.node[i, 1], self.node[i, 2])

    @ti.kernel
    def surface_linear_vector_interpolation(self, f: ti.template(), R: ti.template()):
        """! 定义在二维曲面上的向量函数的线性插值
        """
        for i in range(self.node.shape[0]):
            R[i, 0], R[i, 1] = f(self.node[i, 0], self.node[i, 1], self.node[i, 2])


    @ti.kernel
    def surface_cell_mass_matrix(self, S: ti.template()):
        """
        组装曲面三角形网格上的最低次单元刚度矩阵， 
        这里的曲面是指三角形网格的节点几何维数为 3
        """
        pass

    def add_plot(self, window):
        NN = self.number_of_nodes()
        vertices = ti.Vector.field(3, dtype=ti.f32, shape=NN)
        self.to_vertices_3d(vertices)
        canvas = window.get_canvas()
        canvas.set_background_color((1.0, 1.0, 1.0))
        canvas.triangles(vertices,indices=self.cell)
        canvas.circles(vertices,radius=0.01,color=(1.0,0.0,0.0))

    @ti.kernel
    def to_vertices_3d(self, vertices: ti.template()):
        if self.node.shape[1] == 2:
            for i in range(self.node.shape[0]):
                vertices[i][0] = self.node[i, 0]
                vertices[i][1] = self.node[i, 1]
                vertices[i][2] = 0.0
        elif self.node.shape[1] == 3:
            for i in range(self.node.shape[0]):
                vertices[i][0] = self.node[i, 0]
                vertices[i][1] = self.node[i, 1]
                vertices[i][2] = self.node[i, 2] 
    #'''


    
