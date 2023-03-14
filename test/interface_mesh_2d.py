import numpy as np
import matplotlib.pyplot as plt

from fealpy.geometry import CircleCurve, FoldCurve
from fealpy.mesh import MeshFactory as mf
from fealpy.mesh import PolygonMesh, HalfEdgeMesh2d 
from fealpy.geometry import CircleCurve, FoldCurve
from fealpy.mesh.interface_mesh_generator import find_cut_point 

from scipy.spatial import Delaunay

class HalfEdgeMesh2dWithInterface(HalfEdgeMesh2d):
    def __init__(self, box, interface, nx=10, ny=10):
        mesh = mf.boxmesh2d(box, nx = nx, ny = ny, meshtype='quad')
        mesh = HalfEdgeMesh2d.from_mesh(mesh)

        super(HalfEdgeMesh2dWithInterface, self).__init__(mesh.entity('node')[:], 
                mesh.ds.halfedge[:], mesh.ds.subdomain[:])
        
        self.interface = interface
        self.generate_mesh()

    def generate_mesh(self):
        NN = self.number_of_nodes()
        NC = self.number_of_all_cells()
        NE = self.number_of_edges()

        node = self.node
        cell = self.entity('cell')
        halfedge = self.entity('halfedge')

        isMainHEdge = self.ds.main_halfedge_flag()

        phiValue = self.interface(node[:])
        #phiValue[np.abs(phiValue) < 0.1*h**2] = 0.0
        phiSign = np.sign(phiValue)

        # Step 1: find the nodes near interface
        edge = self.entity('edge')
        isCutHEdge = phiValue[halfedge[:, 0]]*phiValue[halfedge[halfedge[:, 4], 0]] < 0 

        cutHEdge, = np.where(isCutHEdge&isMainHEdge)
        cutEdge = self.ds.halfedge_to_edge(cutHEdge)

        e0 = node[edge[cutEdge, 0]]
        e1 = node[edge[cutEdge, 1]]
        cutNode = find_cut_point(self.interface, e0, e1)

        self.refine_halfedge(isCutHEdge, newnode = cutNode)

        newHE = np.where(halfedge[:, 0] >= NN)[0]
        idx = np.argsort(halfedge[newHE, 1]).reshape(-1, 2)
        newHE = newHE[idx]
        ne = len(newHE)
        NE = len(halfedge)//2

        halfedgeNew = halfedge.increase_size(ne*2)
        halfedgeNew[:ne, 0] = halfedge[newHE[:, 1], 0]
        halfedgeNew[:ne, 1] = halfedge[newHE[:, 1], 1]
        halfedgeNew[:ne, 2] = halfedge[newHE[:, 1], 2]
        halfedgeNew[:ne, 3] = newHE[:, 0] 
        halfedgeNew[:ne, 4] = np.arange(NE*2+ne, NE*2+ne*2)

        halfedgeNew[ne:, 0] = halfedge[newHE[:, 0], 0]
        halfedgeNew[ne:, 1] = np.arange(NC, NC+ne)
        halfedgeNew[ne:, 2] = halfedge[newHE[:, 0], 2]
        halfedgeNew[ne:, 3] = newHE[:, 1] 
        halfedgeNew[ne:, 4] = np.arange(NE*2, NE*2+ne) 

        halfedge[halfedge[newHE[:, 0], 2], 3] = np.arange(NE*2+ne, NE*2+ne*2)
        halfedge[halfedge[newHE[:, 1], 2], 3] = np.arange(NE*2, NE*2+ne)
        halfedge[newHE[:, 0], 2] = np.arange(NE*2, NE*2+ne)
        halfedge[newHE[:, 1], 2] = np.arange(NE*2+ne, NE*2+ne*2)

        isNotOK = np.ones(ne, dtype=np.bool_)
        current = np.arange(NE*2+ne, NE*2+ne*2)
        while np.any(isNotOK):
            halfedge[current[isNotOK], 1] = np.arange(NC, NC+ne)[isNotOK]
            current[isNotOK] = halfedge[current[isNotOK], 2]
            isNotOK = current != np.arange(NE*2+ne, NE*2+ne*2)

        #增加主半边
        self.ds.hedge.extend(np.arange(NE*2, NE*2+ne))

        #更新subdomain
        subdomainNew = self.ds.subdomain.increase_size(ne)
        subdomainNew[:] = self.ds.subdomain[halfedge[newHE[:, 0], 1]]

        #更新起始边
        self.ds.hcell.increase_size(ne)
        self.ds.hcell[halfedge[:, 1]] = np.arange(len(halfedge)) # 的编号

        self.ds.NN = self.node.size
        self.ds.NC += ne 
        self.ds.NE = halfedge.size//2

interface = CircleCurve([0.5, 0.5], 0.31)
#interface = FoldCurve(a = 6)
#mesh = interfacemesh2d([0, 1, 0, 1], interface, n=10)

mesh = HalfEdgeMesh2dWithInterface([0, 1, 0, 1], interface, 16, 16)
fig = plt.figure()
axes = fig.gca() 
mesh.add_plot(axes)
plt.show()

