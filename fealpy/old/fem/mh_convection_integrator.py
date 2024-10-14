import numpy as np
class ScalarConvectionIntegrator:
    def __init__(self, c, C, q=3):
        self.coef = c
        self.C = C
        self.q = q    
    def assembly_cell_matrix(self, space, index=np.s_[:], cellmeasure=None,
            out=None):

        q = self.q
        coef = self.coef
        C = self.C
        mesh = space.mesh
        
        GD = mesh.geo_dimension()
        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        NC = len(cellmeasure)
        ldof = space.number_of_local_dofs() 
        
        if out is None:
            S = np.zeros((NC, ldof, ldof), dtype=space.ftype)
        else:
            S= out
  
        
        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = len(ws)
        
        gphi = space.grad_basis(bcs, index=index) 
        phi = space.basis(bcs, index=index) 
        
        S += np.einsum('q, qck, n, qcmn, c->ckm', ws, phi, coef, gphi, cellmeasure) 
        S += np.einsum('q, ck, n, qcmn, c->ckm', ws, C, coef, gphi, cellmeasure) 
        if out is None:
            return S
def Cchoice(mesh, b, grad_uh): 
    eps = 1e-10
    edge = mesh.entity('edge')
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    NC = mesh.number_of_cells()

    v = node[cell, :][:, [1,2,0]] - node[cell, :][:, [0,1,2]] #(NC,3,2)
    length = np.sqrt(np.square(v).sum(axis=2))
    v = v/length[:, :, None] #单位化  #(NC,3,2)

    sixarea = np.zeros((NC, 6, 2))
    sixarea[:, [3,5,1]] = v 
    sixarea[:, [0,2,4]] = -v  #(NC,6,2)

    index = np.zeros(NC, dtype=np.int_)
    for i in range(6):
        a = np.cross(sixarea[:, i], b[None, :])
        c = np.cross(b[None, :], sixarea[:, (i+1)%6])

        #判断在那个角区域
        flag = (a*c > -eps)& (a>-eps) & (c>-eps) 
        index[flag] = i

    for i in range(6):
        # 判断在那个边上
        flag = (np.sum(b*sixarea[:, i], axis=1)>-eps) & (np.cross(b, 
            sixarea[:, i])<eps) &(np.cross(b, sixarea[:, i])>-eps)
        index[flag] = i if i%2==0 else i-1

    s = b/np.sqrt(np.sum(np.square(b)))

    rm = np.array([[0,-1],[1,0]])
    v2 = sixarea[np.arange(NC), index, :]
    vv2 = (rm@v2.T).T # v2 垂直

    v3 = sixarea[np.arange(NC), (index+1)%6, :]
    vv3 = (rm@v3.T).T # v3 垂直

    v = v2 + v3
    v = v/np.sqrt(np.sum(np.square(v),axis=1))[:,None] #(NC, 2)
    vv = (rm@v.T).T #(NC,2)
    flag = np.sum(v3*vv, axis=1)>=0
    vv[~flag] = -vv[~flag]



    w = (rm@grad_uh.T).T 
    flag = np.sum(w*v, axis=1)>=0
    w[~flag] = -w[~flag]


    h = np.min(mesh.entity_measure(etype='edge'))
    bflag = (np.all(node[cell][:,:,1]<h+eps, axis=1))|\
    (np.all(node[cell][:,:,0]<h+eps, axis=1))|\
    (np.all(node[cell][:,:,1]>1-h-eps, axis=1))|\
    (np.all(node[cell][:,:,0]>1-h-eps, axis=1)) #

    #判断是否在V2
    a = np.cross(b, w)
    c = np.cross(w, v3)
    a1 = np.cross(-b, w)
    c1 = np.cross(w, -v3)
    V21 = (a*c>-eps) & (a>-eps) & (c>-eps)
    V22 = (a1*c1>-eps) & (a1>-eps) & (c1>-eps)
    V2flag = V21|V22

    # 判断是否在V3
    a = np.cross(v2, w)
    c = np.cross(w, b)
    a1 = np.cross(-v2, w)
    c1 = np.cross(w, -b)
    V31 = (a*c>-eps) & (a>-eps) & (c>-eps)
    V32 = (a1*c1>-eps) & (a1>-eps) & (c1>-eps)
    V3flag = V31|V32

    C = np.zeros((NC, 3))
    for i in range(NC):
        if np.all(b==0): 
            C[i] = np.zeros(3)
        elif index[i]%2 ==0:
            C[i] = np.array([-1/3, -1/3, -1/3])
            C[i, index[i]//2] = 2/3
        elif bflag[i]:
            C[i] = np.array([-1/3, -1/3, -1/3])
        elif (np.sum(b*grad_uh[i])<eps) &(np.sum(b*grad_uh[i])>-eps):
            C[i] = np.array([1/6, 1/6, 1/6])
            C[i, (index[i]+3)%6//2] = -1/3 #获得对角区域编号
        elif V2flag[i]:
            C[i] = np.array([-1/3, -1/3, -1/3])
            C[i, (index[i]-1)//2] = 2/3
        elif V3flag[i]:
            C[i] = np.array([-1/3, -1/3, -1/3])
            C[i, ((index[i]+1)%6)//2] = 2/3
        elif np.dot(w[i], vv[i])<eps:
            r2 = min(1, np.abs(np.dot(s,vv2[i])/np.dot(v[i],vv2[i]))+1-np.sign(np.dot(b,vv2[i])))
            Phi = min(1, 2*np.abs(np.dot(w[i],vv2[i]))/(r2*np.dot(v[i],v2[i])))
            C[i, (index[i]-1)//2] = -1/3+1/2*Phi*(1+np.dot((v2[i]-v3[i]),s)/(1-np.dot(v2[i],v3[i])))
            C[i, (index[i]+1)%6//2] = 1/3-C[i, (index[i]-1)//2]
            C[i, (index[i]+3)%6//2] = -1/3
        else:
            r3 = min(1, np.abs(np.dot(s,vv3[i])/np.dot(v[i],vv3[i]))+1-np.sign(np.dot(b,vv3[i])))
            Phi = min(1, 2*np.abs(np.dot(w[i],vv3[i]))/(r3*np.dot(v[i],v3[i])))
            C[i, (index[i]+1)%6//2] = -1/3+1/2*Phi*(1+np.dot((v3[i]-v2[i]),s)/(1-np.dot(v2[i],v3[i])))
            C[i, (index[i]-1)//2] = 1/3-C[i, (index[i]+1)%6//2]
            C[i, (index[i]+3)%6//2] = -1/3
    return C




