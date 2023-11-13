import numpy as np
from numpy.linalg import inv
from fealpy.quadrature import GaussLobattoQuadrature


def coefficient_of_div_VESpace_represented_by_SMSpace(space, M):
    """
    @ brief div v
    由缩放单项式空间表示的系数,M_{sldof,sldof}k_{sldof,ldof}=b_{sldof,ldof}求系数k
    @ param M 取 p-1 (NC,sldof,sldof),(n_{k-1}, n_{k-1}) 质量矩阵
    @ return K 是一个列表，(n_{k-1}, N_k)
    """
    p = space.p
    mesh = space.mesh
    ldof = space.number_of_local_dofs()
    smldof = space.smspace.number_of_local_dofs(p-1)
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    NC = mesh.number_of_cells()
    NV = mesh.number_of_vertices_of_cells()
    hk = np.sqrt(mesh.cell_area())
    K = []
    for i in range(NC):
        cedge = np.zeros((NV[i], 2), dtype=np.int_)
        cedge[:, 0] = cell[i]
        cedge[:-1, 1] = cell[i][1:]
        cedge[-1, -1] = cell[i][0] 

        qf = GaussLobattoQuadrature(p + 1) # NQ
        bcs, ws = qf.quadpts, qf.weights

        v = node[cedge[:, 1]] - node[cedge[:, 0]]
        w = np.array([(0, -1), (1, 0)])
        nm = v@w 
        val = np.einsum('i,jk->jik', ws, nm)

        idx = np.zeros((val.shape), dtype=np.int_)
        idx1 = np.arange(0, NV[i]*2*p, 2*p).reshape(-1, 1) + np.arange(0, 2*p+1, 2)
        idx1[-1, -1] = 0
        idx[:, :, 0] = idx1
        idx[:, :, 1] = idx1+1

        b = np.zeros((smldof, ldof[i]),dtype=np.float64)
        b[1:, ldof[i]-(p*(p+1)//2-1):] = hk[i] * np.eye(p*(p+1)//2-1)
        np.add.at(b, (0, idx), val)
        k = inv(M[i])@b
        K.append(k)
    return K

def vector_decomposition(space, p):
    """
    @ brief [P_{k}(K)]^2 = \nabla P_{k+1}(K) \oplus x^{\perp} P_{k-1}(K)
    @ return A (NC, 2n_k, n_{k+1})  \nabla P_{k+1}(K) 的系数
    @ return B (2n_k, n_{k-1})  x^{\perp} P_{k-1}(K) 的系数
    """
    mesh =  space.mesh
    NC = space.mesh.number_of_cells()
    hk = np.sqrt(mesh.cell_area())
    

    ldof = space.smspace.number_of_local_dofs(p)*2
    ldof1 = space.smspace.number_of_local_dofs(p+1)
    ldof2 = space.smspace.number_of_local_dofs(p-1)
    row = np.arange(ldof)
    data = space.smspace.diff_index_1(p+1)
    x = data['x']
    y = data['y']
    cols = np.hstack((x[0], y[0]))
    value = np.repeat(np.arange(1, p+2), np.arange(1, p+2))
    value = hk[:, None, None]/np.hstack((value, value))
    A = np.zeros((NC, ldof, ldof1),dtype=np.float64)   #(NC, 2n_k,n_{k+1})
    np.add.at(A, (np.s_[:], row[None,:], cols[None,:]), value)

    data = space.smspace.diff_index_1(p)
    x = data['x']
    y = data['y']
    row = np.hstack((y[0], ldof//2+x[0]))
    cols = np.hstack((np.arange(ldof2), np.arange(ldof2)))
    value = np.repeat(np.arange(2, p+2), np.arange(1, p+1))
    value = np.hstack((y[1], -x[1]))/np.hstack((value, value))
    B = np.zeros((ldof, ldof2),dtype=np.float64) # (2n_k,n_{k-1})
    B[row, cols] = value
    return A, B 


def laplace_coefficient(space, p):
    mesh = space.mesh
    data = space.vmspace.diff_index_2() 
    cellarea = mesh.cell_area()
    smldof = space.vmspace.number_of_local_dofs()
    xx = data['xx']
    yy = data['yy']
    E = np.zeros(((p+1)*(p+2),p*(p-1)), dtype=np.float64)
    E[xx[0], np.arange(p*(p-1)//2)] = xx[1]#/cellarea
    E[xx[0]+(p+1)*(p+2)//2, p*(p-1)//2+np.arange(p*(p-1)//2)] += xx[1]
    E[yy[0], np.arange(p*(p-1)//2)] += yy[1]#/cellarea
    E[yy[0]+(p+1)*(p+2)//2, p*(p-1)//2+np.arange(p*(p-1)//2)] += yy[1]
    E = E[None, :, :]/cellarea[:, None, None]
    return E
