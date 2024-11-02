import taichi as ti
import numpy as np
from fealpy.ti.TetrahedronMesh import TetrahedronMesh
from fealpy.mesh import TetrahedronMesh as TMesh
from fealpy.ti.TriangleMesh import TriangleMesh
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.decorator import cartesian

ti.init()

@cartesian
def f(p):
    x = p[..., 0]
    y = p[..., 1]
    z = p[..., 2]
    r = x*x + y*y + z*z                                                        
    return  r 

@ti.func                                                                        
def tf(x: ti.f64, y: ti.f64, z: ti.f64) -> ti.f64:                                          
    r = x*x + y*y + z*z                                                        
    return  r 

@ti.data_oriented
class TetrahedronMeshTest():
    def __init__(self):
        node = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0]], dtype=np.float) # 节点坐标，形状为 (NN, 5)
        cell = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int) # 构成每个单元的四个点的编号，形状为 (NC, 3)

        self.tmesh = TMesh(node, cell)
        self.mesh = TetrahedronMesh(node, cell)
        self.sp = LagrangeFiniteElementSpace(self.tmesh, 1, q=2)

    def test_mesh_top(self):
        tedge = self.tmesh.entity("edge")
        edge = self.mesh.entity("edge").to_numpy()
        if(np.sum(np.abs(edge-tedge))==0):
            print("edge: True")
        else:
            print("edge: False")

        tface = self.tmesh.entity("face")
        face = self.mesh.entity("face").to_numpy()
        if(np.sum(np.abs(face-tface))==0):
            print("face: True")
        else:
            print("face: False")

        tc2e = self.tmesh.ds.cell2edge
        c2e = self.mesh.cell2edge.to_numpy()
        if(np.sum(np.abs(tc2e-c2e))==0):
            print("cell2edge: True")
        else:
            print("cell2edge: False")

        tc2f = self.tmesh.ds.cell_to_face()
        c2f = self.mesh.cell2face.to_numpy()
        if(np.sum(np.abs(tc2f-c2f))==0):
            print("cell2face: True")
        else:
            print("cell2face: False")

        tf2c = self.tmesh.ds.face2cell
        f2c = self.mesh.face2cell.to_numpy()
        if(np.sum(np.abs(tf2c-f2c))==0):
            print("face2cell: True")
        else:
            print("face2cell: False")

    @ti.kernel
    def test_grad_lambda(self):
        glambda, l = self.mesh.grad_lambda(0)
        print(glambda[0, :])

    def test_grad_lambda0(self):
        print(self.tmesh.grad_lambda()[0])

    def test_stiff_matrix(self):
        M = self.mesh.stiff_matrix()
        print(M.toarray())
        print(self.sp.stiff_matrix().toarray())

    def test_mass_matrix(self):
        M = self.mesh.mass_matrix()
        print(M.toarray())
        print(self.sp.mass_matrix().toarray())

    def test_source(self):
        v = self.mesh.source_vector(tf)
        v1 = self.sp.source_vector(f)
        print(v)
        print(v1)

test = TetrahedronMeshTest()
test.test_mesh_top()
test.test_grad_lambda()
test.test_grad_lambda0()

test.test_stiff_matrix()
test.test_mass_matrix()

test.test_source()


