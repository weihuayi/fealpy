import pytest
import numpy as np
from fealpy.decorator import barycentric,cartesian
from fealpy.backend import backend_manager as bm
from fealpy.mesh.tetrahedron_mesh import TetrahedronMesh
from fealpy.quadrature.stroud_quadrature import StroudQuadrature 
from fealpy.functionspace.cm_conforming_fe_space3d import CmConformingFESpace3d
from cm_fe_space_3d_data import *
from fealpy.pde.biharmonic_triharmonic_3d import get_flist
import sympy as sp
import ipdb

class TestCmfespace3d:
    def test_coefficient_interpolation(self, m, p):
        #space = CmConformingFESpace3d(mesh, p, m, isCornerNode=node)

        #gdof = space.number_of_global_dofs()
        #print(gdof)
        #c2d = space.cell_to_dof()

        #for kk in range(gdof):
        #    uh = space.function()
        #    uh[:] = 0
        #    uh[kk] = 1
        #    #bcs = bm.array([[0, 0.5, 0.2, 0.3],[0, 0.3, 0.5, 0.2]], dtype=bm.float64)
        #    #bcs = bm.array([[0, 0.3, 0.8, 0.0],[0,0,0.3,0.8]])
        #    bcs = bm.array([[0,0,0,1],[0,1,0,0]])
        #    #bcs[1, 1:] = bcs[0, 1:][cell[1, 1:]]
        #    #print(bcs)
        #    #val = space.grad_m_value(uh, bcs, 1)
        #    val = space.value(uh, bcs)
        #    #print(val.shape)
        #    import numpy as np
        #    np.testing.assert_allclose(val[0,0], val[1,1], atol=1e-10)
        #    #print(np.max(np.abs(val[0,0]-val[1,1])))
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],2,2,2)
        NC = mesh.number_of_cells()
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        face = mesh.entity('face')
        f2e = mesh.face_to_edge()
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)

        from fealpy.functionspace.cm_conforming_fe_space3d_old import CmConformingFESpace3d as CmConformingFESpace3d_old
        space = CmConformingFESpace3d(mesh, 10, 1, isCornerNode)
        #space1 = CmConformingFESpace3d_old(mesh, 10, 1, isCornerNode)
        #np.testing.assert_allclose(space.coeff, space1.coeff, atol=1e-10)
        multiidx = space.multiIndex
        c2f = space.cell_to_dof()

        x = sp.symbols('x')
        y = sp.symbols('y')
        z = sp.symbols('z')
        u_sp = sp.sin(2*x)*sp.sin(2*y)*sp.sin(z)
        np.set_printoptions(threshold=np.inf)
        flist = get_flist(u_sp)
        #torch.set_printoptions(precision=16)
        #torch.set_printoptions(threshold=torch.inf)
        bc = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        nnode = 16
        cc, j = np.where(cell==nnode)
        #bcs = np.zeros((len(cc), 4)) #(12,4)
        #for i in range(len(cc)):
        #    bcs[i] = bc[j[i]]
        c2d = space.cell_to_dof()
        n2d = space.node_to_dof()[nnode][1:]
        uh = space.function()
        uh[:] = 0
        uh[n2d] = 3
        gval = space.grad_m_value(uh, bc, 2)
        print(gval.shape)
        node_frame, edge_frame, face_frame = space.get_frame() 
        print(node_frame.shape)
        from fealpy.functionspace.functional import symmetry_span_array, symmetry_index, span_array
        #sdelta = 2 
        #idx, num = symmetry_index(3, sdelta)
        #multiindex = mesh.multi_index_matrix(p=sdelta, etype=2)
        #nnn = np.zeros((multiindex.shape[0], idx.shape[0]))
        #for i in range(multiindex.shape[0]):
        #    __import__('ipdb').set_trace()
        #    nn = symmetry_span_array(node_frame[nnode][None,:], multiindex[i]).reshape(-1)[idx]
        #    nnn[i] = nn
        #print('1',nnn.shape)
        #print(nnn)
        #for i in range(sdelta):
        #    gval = space.grad_m_value(uh, bc, i+1)
        #    gvall = bm.zeros((len(cc), gval.shape[2]))
        #    for i in range(len(cc)):
        #        gvall[i] = gval[cc[i],j[i]]
        #    idx = np.max(np.abs(gvall-gvall[0]),axis=1)<1e-10
        #a = np.einsum('cl,l,nl->cn',gvall,num,nnn)


        sdelta = 2 
        idx, num = symmetry_index(3, sdelta) #(6,)
        multiindex = mesh.multi_index_matrix(p=sdelta, etype=2)
        mul = np.repeat(multiindex, num.astype(int), axis=0)
        nnn = np.zeros((mul.shape[0], idx.shape[0])) #(9,6)
        for i in range(mul.shape[0]):
            nn = symmetry_span_array(node_frame[nnode][None,:], mul[i]).reshape(-1)[idx]
            #nn = symmetry_span_array(node_frame[nnode][None,:], multiindex[i])
            nnn[i] = nn
        #print('1',nnn.shape)
        print(nnn)
        for i in range(sdelta):
            gval = space.grad_m_value(uh, bc, i+1)
            gvall = bm.zeros((len(cc), gval.shape[2])) #(12,6)
            for i in range(len(cc)):
                gvall[i] = gval[cc[i],j[i]]
            idx = np.max(np.abs(gvall-gvall[0]),axis=1)<1e-10
        a = np.einsum('cl,l,nl->cn',gvall,num,nnn)
        print('aaaaaa',a)


        #gval = space.grad_m_value(uh, bc, 1)
        #gvall = bm.zeros((len(cc), gval.shape[2])) #(12,3)
        #for i in range(len(cc)):
        #    gvall[i] = gval[cc[i],j[i]]
        #nnnn = node_frame[16]
        #aa = np.einsum('cl,nl->cn',gvall,nnnn)
        #print(aa)

        gval = space.grad_m_value(uh, bc, 2)
        gvall = bm.zeros((len(cc), gval.shape[2])) #(12,6)
        for i in range(len(cc)):
            gvall[i] = gval[cc[i],j[i]]
        nnnn = node_frame[16]
        gvalll = np.zeros((12,3,3))
        gvalll[:,0,0] = gvall[:,0]
        gvalll[:,0,1] = gvall[:,1]
        gvalll[:,0,2] = gvall[:,2]
        gvalll[:,1,0] = gvall[:,1]
        gvalll[:,1,1] = gvall[:,3]
        gvalll[:,1,2] = gvall[:,4]
        gvalll[:,2,0] = gvall[:,2]
        gvalll[:,2,1] = gvall[:,4]
        gvalll[:,2,2] = gvall[:,5]

        for i in range(3):
            for j in range(3):
                nnn = nnnn[i].reshape(3,1)@nnnn[j].reshape(1,3)
                print(nnn)
                aa = np.einsum('clk,lk->c',gvalll,nnn)
                print(aa)



        #vall = np.zeros((len(cc)), dtype=np.float64)
        #val = space.value(uh, bc)
        #for i in range(len(cc)):
        #    vall[i] = val[cc[i],j[i]]
        
        #nn = node[nnode]
        #f = flist[0](nn)
        #print('f',f)
        #g1f = flist[1](nn)
        #g2f = flist[2](nn)
        #g3f = flist[3](nn)
        #g4f = flist[4](nn)
        #print('g1f',g1f)
        #print('g2f',g2f)
        #print('g3f',g3f)
        #print('g4f',g4f)
        #import ipdb
        #ipdb.set_trace()
        #fI = space.interpolation(flist)
        #fvalue = space.value(fI, bc) #(48,4)
        #grad1fI =space.grad_m_value(fI, bc, 1)
        #print(grad1fI.shape)
        #grad2fI =space.grad_m_value(fI, bc, 2)
        ##print(grad2fI.shape)
        #grad3fI =space.grad_m_value(fI, bc, 3)
        #grad4fI =space.grad_m_value(fI, bc, 4)
        #print(cell[cc])
        #print(cc)
        #print(j)
        #print('fI',fvalue[cc,j])
        #print('grad1fI',grad1fI[cc,j])
        #print('grad2fI',grad2fI[cc,j])
        #print('grad3fI',grad3fI[cc,j])
        #print('grad4fI',grad4fI[cc,j])
        #np.testing.assert_allclose(fvalue[cc,j], f, atol=1e-12)
        #g1f = np.tile(g1f, (len(cc), 1))
        #g2f = np.tile(g2f, (len(cc), 1))
        #g3f = np.tile(g3f, (len(cc), 1))
        #g4f = np.tile(g4f, (len(cc), 1))
        #np.testing.assert_allclose(grad1fI[cc,j], g1f, atol=1e-10)
        #np.testing.assert_allclose(grad2fI[cc,j], g2f, atol=1e-10)
        #np.testing.assert_allclose(grad3fI[cc,j], g3f, atol=1e-10)
        #np.testing.assert_allclose(grad4fI[cc,j], g4f, atol=1e-10)



    #@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    def test_coefficient_matrix(self, m, p):
        node = bm.array([[-2, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0.0, 0.5], [0, 0, -1]], dtype=bm.float64)
        cell = bm.array([[3,0,1,2],[4,2,0,1]], dtype=bm.int32) 
        mesh = TetrahedronMesh(node, cell)
        node = mesh.entity('node')
        space = CmConformingFESpace3d(mesh, p, m, isCornerNode=node)

        gdof = space.number_of_global_dofs()
        print(gdof)
        c2d = space.cell_to_dof()

        for kk in range(gdof):
            uh = space.function()
            uh[:] = 0
            uh[kk] = 1
            #bcs = bm.array([[0, 0.5, 0.2, 0.3],[0, 0.3, 0.5, 0.2]], dtype=bm.float64)
            #bcs = bm.array([[0, 0.3, 0.8, 0.0],[0,0,0.3,0.8]])
            bcs = bm.array([[0,0,0,1],[0,1,0,0]])
            #bcs[1, 1:] = bcs[0, 1:][cell[1, 1:]]
            #print(bcs)
            #val = space.grad_m_value(uh, bcs, 1)
            val = space.value(uh, bcs)
            #print(val.shape)
            import numpy as np
            np.testing.assert_allclose(val[0,0], val[1,1], atol=1e-10)
            #print(np.max(np.abs(val[0,0]-val[1,1])))








    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", get_dof_index )
    def test_get_dof_index(self, data, backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],2,2,2)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)
        get_dof_index = space.dof_index["all"]        
        np.testing.assert_array_equal(get_dof_index, data["dof_index"])

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", number_of_internal_global_dofs)
    def test_number_of_internal_dofs(self, data, backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],2,2,2)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)
        gdof = space.number_of_global_dofs()
        idof = space.number_of_internal_dofs(data["etype"])
        #np.testing.assert_array_equal(ldof, data["ldof"])
        assert idof == data["ldof"]
        assert gdof == data["gdof"]


    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", nefc_to_internal_dof)
    def test_nefc_to_internal_dofs(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],1,1,1)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)
        n2d = space.node_to_dof()
        e2id = space.edge_to_internal_dof()
        f2id = space.face_to_internal_dof()
        c2id = space.cell_to_internal_dof()
        np.testing.assert_array_equal(n2d, data["node"])
        np.testing.assert_array_equal(e2id, data["edge"])
        np.testing.assert_array_equal(f2id, data["face"])
        np.testing.assert_array_equal(c2id, data["cell"])


    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", cell2dof)
    def test_cell_to_dofs(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],3,3,3)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)

        c2d = space.cell_to_dof()[data["cellnum"]]
        np.testing.assert_allclose(c2d, data["cell2dof"])

#    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
#    @pytest.mark.parametrize("data", coefficient_matrix)
#    def test_coefficient_matrix(self, data, backend):
#        bm.set_backend(backend)
#        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],1,1,1)
#        node = mesh.entity('node')
#        isCornerNode = np.zeros(len(node), dtype=np.bool)
#        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
#            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
#        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)
#        coefficient_matrix = space.coefficient_matrix() #(6, 364, 364)
#
#        np.testing.assert_allclose(coefficient_matrix[0, 180], data["cell0"], atol=1e-14)
#        np.testing.assert_allclose(coefficient_matrix[1, 256], data["cell1"], atol=1e-14)
#        np.testing.assert_allclose(coefficient_matrix[2, 200], data["cell2"], atol=1e-14)
#        np.testing.assert_allclose(coefficient_matrix[3, 20], data["cell3"], atol=1e-14)
#        np.testing.assert_allclose(coefficient_matrix[4, 105], data["cell4"], atol=1e-14)
#        np.testing.assert_allclose(coefficient_matrix[5, 78], data["cell5"], atol=1e-14)

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", basis)
    def test_basis(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],1,1,1)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)
        bcs = mesh.multi_index_matrix(p=4, etype=3)/4
        #qf = StroudQuadrature(3, 7)
        #bcs, ws = qf.get_points_and_weights()
        basis = space.basis(bcs)
 
        np.testing.assert_allclose(basis[0, 27], data["cell0"], atol=1e-14)
        np.testing.assert_allclose(basis[1, 20], data["cell1"], atol=1e-14)
        np.testing.assert_allclose(basis[2, 5],  data["cell2"], atol=1e-14)
        np.testing.assert_allclose(basis[3, 32], data["cell3"], atol=1e-14)
        np.testing.assert_allclose(basis[4, 11], data["cell4"], atol=1e-14)
        np.testing.assert_allclose(basis[5, 16], data["cell5"], atol=1e-14)


    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", grad_m_basis)
    def test_grad_m_basis(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],1,1,1)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)
        bcs = mesh.multi_index_matrix(p=4, etype=3)/4
        gmbasis = space.grad_m_basis(bcs, 1)

        np.testing.assert_allclose(gmbasis[0,3,:,0], data["cell0"], atol=1e-14)
        np.testing.assert_allclose(gmbasis[1,22,:,1], data["cell1"], atol=1e-14)
        np.testing.assert_allclose(gmbasis[2,13,:,2],  data["cell2"], atol=1e-14)
        np.testing.assert_allclose(gmbasis[3,33,:,0], data["cell3"], atol=1e-14)
        np.testing.assert_allclose(gmbasis[4,28,:,1], data["cell4"], atol=1e-14)
        np.testing.assert_allclose(gmbasis[5,7,:,2],  data["cell5"], atol=1e-12)

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    def test_interpolation1(self,  backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],2,2,2)
        NC = mesh.number_of_cells()
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        face = mesh.entity('face')
        f2e = mesh.face_to_edge()
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)

        from fealpy.functionspace.cm_conforming_fe_space3d_old import CmConformingFESpace3d as CmConformingFESpace3d_old
        space = CmConformingFESpace3d(mesh, 10, 1, isCornerNode)
        #space1 = CmConformingFESpace3d_old(mesh, 10, 1, isCornerNode)
        #np.testing.assert_allclose(space.coeff, space1.coeff, atol=1e-10)
        multiidx = space.multiIndex
        c2f = space.cell_to_dof()
        node_frame, edge_frame, face_frame = space.get_frame() 
        #print(edge_frame)

        x = sp.symbols('x')
        y = sp.symbols('y')
        z = sp.symbols('z')
        u_sp = sp.sin(2*x)*sp.sin(2*y)*sp.sin(z)
        #u_sp = x**2
        np.set_printoptions(threshold=np.inf)
        flist = get_flist(u_sp)
        #torch.set_printoptions(precision=16)
        #torch.set_printoptions(threshold=torch.inf)
        bc = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        nnode = 16
        nn = node[nnode]
        print(nn)
        f = flist[0](nn)
        print('f',f)
        g1f = flist[1](nn)
        g2f = flist[2](nn)
        g3f = flist[3](nn)
        g4f = flist[4](nn)
        print('g1f',g1f)
        print('g2f',g2f)
        print('g3f',g3f)
        print('g4f',g4f)
        import ipdb
        ipdb.set_trace()
        fI = space.interpolation(flist)
        fvalue = space.value(fI, bc) #(48,4)
        grad1fI =space.grad_m_value(fI, bc, 1)
        print(grad1fI.shape)
        grad2fI =space.grad_m_value(fI, bc, 2)
        #print(grad2fI.shape)
        grad3fI =space.grad_m_value(fI, bc, 3)
        grad4fI =space.grad_m_value(fI, bc, 4)
        cc, j = np.where(cell==nnode)
        print(cell[cc])
        print(cc)
        print(j)
        print('fI',fvalue[cc,j])
        print('grad1fI',grad1fI[cc,j])
        print('grad2fI',grad2fI[cc,j])
        print('grad3fI',grad3fI[cc,j])
        print('grad4fI',grad4fI[cc,j])
        np.testing.assert_allclose(fvalue[cc,j], f, atol=1e-12)
        g1f = np.tile(g1f, (len(cc), 1))
        g2f = np.tile(g2f, (len(cc), 1))
        g3f = np.tile(g3f, (len(cc), 1))
        g4f = np.tile(g4f, (len(cc), 1))
        np.testing.assert_allclose(grad1fI[cc,j], g1f, atol=1e-10)
        np.testing.assert_allclose(grad2fI[cc,j], g2f, atol=1e-10)
        np.testing.assert_allclose(grad3fI[cc,j], g3f, atol=1e-10)
        np.testing.assert_allclose(grad4fI[cc,j], g4f, atol=1e-10)

 
    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    # @pytest.mark.parametrize("data", interpolation)
    def test_interpolation(self, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],2,2,2)
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        face = mesh.entity('face')
        f2e = mesh.face_to_edge()
        
        #print(cell[np.array([12,13,14,15,16,17,18,23,39,40,46,47])])
        #print(bm.where(cell == 16)[0])
        #print(cell[np.array([36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 53, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 106, 107, 147, 148, 153, 154, 155, 160, 161])])
        NC = mesh.number_of_cells()
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        from fealpy.functionspace.cm_conforming_fe_space3d_old import CmConformingFESpace3d as CmConformingFESpace3d_old
        space = CmConformingFESpace3d(mesh, 10, 1, isCornerNode)
        multiidx = space.multiIndex
        node_frame, edge_frame, face_frame = space.get_frame() 
        #print(edge_frame)

        x = sp.symbols('x')
        y = sp.symbols('y')
        z = sp.symbols('z')
        u_sp = sp.sin(2*x)*sp.sin(2*y)*sp.sin(z)
        np.set_printoptions(threshold=np.inf)
        flist = get_flist(u_sp)
        #torch.set_printoptions(precision=16)
        #torch.set_printoptions(threshold=torch.inf)
        fI = space.interpolation(flist)
        c2f = space.cell_to_dof()
        coeff = space.coeff
        for j in range(NC):
            coeff[j] = np.linalg.inv(coeff[j])
        from fealpy.functionspace.bernstein_fe_space import BernsteinFESpace
        bspace = space.bspace
        bfI = bspace.interpolate(flist[0])
        bc2f = bspace.cell_to_dof()
        print(space.dof_index['edge'])
        a = []
        for j in range(NC):
            err = np.max(np.abs(fI[c2f[j]] - bfI[bc2f[j]] @ coeff[j]))
            #print(np.abs(fI[c2f[j]] - bfI[bc2f[j]] @ coeff[j]))
            print(j, err)
            #idx = np.where(np.abs(fI[c2f[j]] - bfI[bc2f[j]] @ coeff[j]) > 10)
            #print(multiidx[idx])
            #print(idx)
            #if idx[0].shape[0] != 0:
            #    a.append(j)
        #print(a)
        #print(cell[np.array(a)])
        #print(node_frame)
        #print(bfI.shape) #12167
        #print(space.coeff.shape) # (48, 364, 364)
        #print(fI.shape) #(6385)
        #bcoeff = binterpolation @ np.linalg.inv(space.coeff) 
        #np.testing.assert_allclose(interpolation, bcoeff, atol=1e-12)
 
#    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
#    @pytest.mark.parametrize("data", interpolation)
#    def test_interpolation(self, data, backend):
#        bm.set_backend(backend)
#        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],2,2,2)
#        node = mesh.entity('node')
#        isCornerNode = np.zeros(len(node), dtype=np.bool)
#        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
#            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
#        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)
#
#        x = sp.symbols('x')
#        y = sp.symbols('y')
#        z = sp.symbols('z')
#        u_sp = sp.sin(2*x)*sp.sin(2*y)*sp.sin(z)
#        np.set_printoptions(threshold=np.inf)
#        flist = get_flist(u_sp)
#        #torch.set_printoptions(precision=16)
#        #torch.set_printoptions(threshold=torch.inf)
#        interpolation = space.interpolation(flist)
#        np.testing.assert_allclose(interpolation, data["mesh2"], atol=1e-12)
#        #a = np.where(np.abs(interpolation-data[1]["mesh2"])>140)
#        #print(a)
        
    
    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    def test_source_vector(self, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],1,1,1)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 9, 1, isCornerNode)
        from fealpy.pde.biharmonic_triharmonic_3d import get_flist
        x = sp.symbols('x')
        y = sp.symbols('y')
        z = sp.symbols('z')
        f = x**5*y**4+z**9
        get_flist = get_flist(f)

        @cartesian
        def f(p):
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
            return x**5*y**4+z**9
        from fealpy.fem import BilinearForm
        from fealpy.fem import LinearForm, ScalarSourceIntegrator, ScalarMassIntegrator
        lform = LinearForm(space)
        lform.add_integrator(ScalarSourceIntegrator(f, q=14))
        F = lform.assembly()

        bform = BilinearForm(space)                                                 
        bform.add_integrator(ScalarMassIntegrator(coef=1,q=14))                     
        M = bform.assembly() #(582, 582) 

        fi = space.interpolation(get_flist)
        x = fi[:]
        aa = M@x
        print(bm.abs(aa-F).max())
        np.testing.assert_allclose(aa, F, atol=1e-14,rtol=0)

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    def test_matrix(self, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],1,1,1)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)
        from fealpy.pde.biharmonic_triharmonic_3d import get_flist
        x = sp.symbols('x')
        y = sp.symbols('y')
        z = sp.symbols('z')
        f = x**7*y**2+z**2
        get_flist = get_flist(f)

        from fealpy.fem import BilinearForm
        from fealpy.fem import LinearForm
        from fealpy.fem.scalar_mlaplace_source_integrator import ScalarMLaplaceSourceIntegrator
        from fealpy.fem.mthlaplace_integrator import MthLaplaceIntegrator
        lform = LinearForm(space)
        lform.add_integrator(ScalarMLaplaceSourceIntegrator(2, get_flist[2], q=14))
        gF = lform.assembly()

        bform = BilinearForm(space)                                                 
        bform.add_integrator(MthLaplaceIntegrator(m=2, coef=1,q=14))                     
        A = bform.assembly()  

        fi = space.interpolation(get_flist)
        x = fi[:]
        aa = A@x
        print(bm.abs(aa-gF).max())
        #np.testing.assert_allclose(aa, gF, atol=1e-14,rtol=0)


#    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
#    #@pytest.mark.parametrize("data", interpolation)
#    def test_boundary_interpolation(self,  backend):
#        bm.set_backend(backend)
#        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],1,1,1)
#        node = mesh.entity('node')
#        isCornerNode = np.zeros(len(node), dtype=np.bool)
#        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
#            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
#        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)
#        uh = space.function()
#        x = sp.symbols('x')
#        y = sp.symbols('y')
#        z = sp.symbols('z')
#        f = x**7*y**2+z**2
#        gD = get_flist(f)
#
#        boundary_interpolation = space.boundary_interpolate(gD, uh)


    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    #@pytest.mark.parametrize("data", interpolation)
    def test_boundary_dof(self,  backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],1,1,1)
        node = mesh.entity('node')
        import numpy as np
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)

        space = CmConformingFESpace3d(mesh,11, 1, isCornerNode)
        #uh = space.function()
        #from fealpy.pde.biharmonic_triharmonic_3d import get_flist
        #x = sp.symbols('x')
        #y = sp.symbols('y')
        #z = sp.symbols('z')
        #f = x**7*y**2+z**2
        #gD = get_flist(f)
        np.set_printoptions(threshold=np.inf)
        idxx = space.dof_index['edge']
        #print(idxx)

        boundary_dof = space.is_boundary_dof()
        #boundary_dof = space.is_boundary_dof1()
        mul = space.multiIndex
        all_edge_dof_index = bm.concatenate([bm.concatenate(item) if isinstance(item, list) else item for sublist in idxx for item in sublist])

        #print(mul[all_edge_dof_index])

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    #@pytest.mark.parametrize("data", interpolation)
    def test_get_corner(self,  backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],2,2,2)
        node = mesh.entity('node')
        import numpy as np
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)

        space = CmConformingFESpace3d(mesh,9, 1, isCornerNode)
        isCornerNode, isBdEdgeNode, isBdFaceNode,isCornerEdge,isBdFaceEdge = space.get_corner()
        #from fealpy.functionspace.cm_conforming_fe_space3d_old import CmConformingFESpace3d as CmConformingFESpace3d_old
        #space = CmConformingFESpace3d_old(mesh,9, 1, isCornerNode)
        #is_corner_node , is_corner_edge = space.get_corner()
        #print('old',is_corner_node)
        #print('old',is_corner_edge)
        print('new',sum(isCornerNode)) # 8
        print('new',sum(isBdEdgeNode)) # 12
        print('new',sum(isBdFaceNode)) # 6
        print('new',sum(isCornerEdge)) # 24
        print('new',sum(isBdFaceEdge)) # 48
        
        
    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    #@pytest.mark.parametrize("data", interpolation)
    def test_get_frame(self,  backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],2,2,2)
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        face = mesh.entity('face')
        #print('face',face)
        #print('edge',edge)
        import numpy as np
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)

        space = CmConformingFESpace3d(mesh,9, 1, isCornerNode)
        node_frame, edge_frame, face_frame = space.get_frame()
        f2e = mesh.face_to_edge()
        isCornerNode, isBdEdgeNode, isBdFaceNode,isCornerEdge,isBdFaceEdge = space.get_corner()
        #print('edge',list(zip(np.arange(len(edge)),edge)))
        #print('f2e',f2e)
        #print(face_frame.shape)
        #print(face.shape)
        #print(list(zip(np.arange(len(node)), node_frame)))
        #print(mesh.entity('face'))
        #print(face_frame)
        #isCornerNode, isBdEdgeNode, isBdFaceNode,isCornerEdge,isBdFaceNode = space.get_corner()
        #print('node_frame',node_frame[isBdEdgeNode])
        #print('edge_frame',(edge_frame,))
        np.set_printoptions(threshold=np.inf)
        #print('face_frame',(face_frame,))
        from fealpy.functionspace.cm_conforming_fe_space3d_old import CmConformingFESpace3d as CmConformingFESpace3d_old
        space = CmConformingFESpace3d_old(mesh,9, 1, isCornerNode)
        node_frame, edge_frame, face_frame = space.get_frame()
        #print('node_frame_old',node_frame[isBdEdgeNode])
        #print('edge_frame_old',(edge_frame,))
        #print('face_frame_old',(face_frame,))

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    #@pytest.mark.parametrize("data", interpolation)
    def test_is_boundary_dof(self,  backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],2,2,2)
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        face = mesh.entity('face')
        #print('face',face)
        #print('edge',edge)
        import numpy as np
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)

        import numpy as np
        np.set_printoptions(threshold=np.inf)
        space = CmConformingFESpace3d(mesh,10, 1, isCornerNode)
        isbddof = space.is_boundary_dof()


    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    #@pytest.mark.parametrize("data", interpolation)
    def test_boundary_interpolation(self,  backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],2,2,2)
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        face = mesh.entity('face')
        #print('face',face)
        #print('edge',edge)
        import numpy as np
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)

        import numpy as np
        np.set_printoptions(threshold=np.inf)
        space = CmConformingFESpace3d(mesh,9, 1, isCornerNode)
        isbddof = space.is_boundary_dof()
        from fealpy.fem import DirichletBC
        from fealpy.pde.biharmonic_triharmonic_3d import get_flist
        from fealpy.fem import BilinearForm
        from fealpy.fem.mthlaplace_integrator import MthLaplaceIntegrator
        from fealpy.fem import LinearForm, ScalarSourceIntegrator
        from fealpy.pde.biharmonic_triharmonic_3d import get_flist, DoubleLaplacePDE
        x = sp.symbols('x')
        y = sp.symbols('y')
        z = sp.symbols('z')
        u = (sp.sin(2*x)*sp.sin(2*y)*sp.sin(z))**2
        ulist = get_flist(u)
        pde = DoubleLaplacePDE(u)
        lform = LinearForm(space)
        bform = BilinearForm(space)
        integrator = MthLaplaceIntegrator(m=2, coef=1, q=14)
        lform.add_integrator(ScalarSourceIntegrator(pde.source, q=14))
        A = bform.assembly()
        F = lform.assembly()
        bc1 = DirichletBC(space, gd=ulist)
        A, F = bc1.apply(A, F)
        print(11)
 
if __name__=="__main__":

    t = TestCmfespace3d()
    # print(1)
    #t.test_coefficient_interpolation(1,11)
    #t.test_get_dof_index(get_dof_index[0], "numpy")
    # t.test_get_dof_index(get_dof_index[0], "pytorch")
    # t.test_number_of_internal_dofs(number_of_global_dofs, "pytorch")
    #t.test_cell_to_dofs(nefc_to_internal_dof[0], "numpy")
    #t.test_coefficient_matrix(cell2dof, "pytorch")
    #import ipdb
    #ipdb.set_trace()
    #t.test_coefficient_matrix(1,11)
    #t.test_basis(coefficient_matrix, "numpy")
    #t.test_interpolation( "numpy")
    #t.test_interpolation1("numpy")
    #t.test_source_vector("pytorch")
    #t.test_matrix("pytorch")
    #t.test_boundary_dof("numpy")
    #t.test_get_corner("numpy")
    #t.test_get_frame("numpy")
    #t.test_is_boundary_dof("numpy")
    #t.test_is_boundary_dof("pytorch")
    #t.test_boundary_interpolation("numpy")
