from fealpy.backend import backend_manager as bm
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh
from scipy.sparse.linalg import spsolve
from fealpy.backend import backend_manager as bm
import matplotlib.pyplot as plt
class PDE:
    def __init__(self):
        pass
    def solution(self, p):
        x, y = p[..., 0], p[..., 1]
        return bm.exp(x**2 + y**2)-bm.exp(1)
    def source(self, p):
        x, y = p[..., 0], p[..., 1]
        return 4*(x**2 + y**2 + 1) * bm.exp(x**2 + y**2)


class PoissonFVM2D:
    def __init__(self,pde):
        self.pde = pde

    def generate_mesh(self, nx):
        self.mesh = TriangleMesh.from_unit_circle_gmsh(nx)

    def compute_circumcenter(self):
        node = self.mesh.entity("node")
        cell = self.mesh.entity("cell")
        p0, p1, p2 = node[cell[:, 0]], node[cell[:, 1]], node[cell[:, 2]]
        a = p1 - p0
        b = p2 - p0
        adot = (a * a).sum(axis=1)
        bdot = (b * b).sum(axis=1)
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        D = 2 * cross
        ux = (adot * b[:, 1] - bdot * a[:, 1]) / D
        uy = (bdot * a[:, 0] - adot * b[:, 0]) / D
        circumcenter = p0 + bm.stack([ux, uy], axis=1)
        return circumcenter

    def assemble_matrix(self,circumcenter):
        node = self.mesh.entity("node")
        edge = self.mesh.entity("edge")
        edge2cell = self.mesh.edge2cell
        edge_length = self.mesh.entity_measure("edge")
        NE = self.mesh.number_of_edges()
        flux_coef = bm.zeros(NE)
        c0 = edge2cell[:, 0]
        c1 = edge2cell[:, 1]
        edge2cell = self.mesh.edge2cell
        boundary_index = self.mesh.boundary_face_index()
        noboundary_index = bm.setdiff1d(bm.arange(NE), boundary_index)
        # 非边界边：通量系数
        cc0 = circumcenter[c0[noboundary_index]]
        cc1 = circumcenter[c1[noboundary_index]]
        dist = bm.linalg.norm(cc0 - cc1, axis=1)
        flux_coef[noboundary_index] = edge_length[noboundary_index] / dist
        # 边界边：通量系数
        e = edge[boundary_index]
        midpoint = (node[e[:, 0]] + node[e[:, 1]]) / 2
        c = c0[boundary_index]
        cc = circumcenter[c]
        dist_b = bm.linalg.norm(cc - midpoint, axis=1)
        flux_coef[boundary_index] = edge_length[boundary_index] / dist_b
        # 组装矩阵
        NC = self.mesh.number_of_cells()
        A = bm.zeros((NC, NC))
        for i in noboundary_index:
            ci0 = c0[i]
            ci1 = c1[i]
            A[ci0, ci0] -= flux_coef[i]
            A[ci1, ci1] -= flux_coef[i]
            A[ci0, ci1] += flux_coef[i]
            A[ci1, ci0] += flux_coef[i]
        for i in boundary_index:
            ci = c0[i]
            A[ci, ci] -= flux_coef[i]
        # 右端项
        rhs = self.mesh.integral(self.pde.source, q=3, celltype=True)
        return A ,rhs

    def run(self, nx=0.5, maxit=4):
        NC = bm.zeros(maxit, dtype=bm.int32)
        e = bm.zeros(maxit, dtype=bm.float64)
        for i in range(maxit):  
            self.generate_mesh(nx=nx)
            circumcenter = self.compute_circumcenter()
            A, rhs = self.assemble_matrix(circumcenter)
            uh = spsolve(A, rhs)
            u_exact = self.pde.solution(circumcenter)
            e[i] = bm.max(bm.abs(uh - u_exact))
            NC[i] = self.mesh.number_of_cells()
            if i < maxit - 1:
                nx = nx/2
        print('number of cells:', NC)
        print('error:', e)
        print('error rate:', e[:-1] / e[1:])
        fig = plt.figure()
        axes = fig.add_subplot()
        self.mesh.add_plot(axes)
        plt.show()

pde = PDE()
model = PoissonFVM2D(pde)
model.run(maxit=5)