from fealpy.mesh import TetrahedronMesh
from fealpy.backend import backend_manager as bm
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mesh = TetrahedronMesh.from_one_tetrahedron()
node = mesh.entity('node')
mesh.uniform_refine()
NN = mesh.number_of_nodes()
NE = mesh.number_of_edges()
NF = mesh.number_of_faces()
NC = mesh.number_of_cells()

fig = plt.figure(1)
axes = fig.add_subplot(111, projection='3d')
mesh.add_plot(axes)
plt.title("Grid Image")
mesh.find_node(axes, showindex=True)
mesh.find_edge(axes, showindex=True)
plt.show()