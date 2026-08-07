
from fealpy.mesh import PrismMesh



mesh = PrismMesh.from_box([0, 1, 0, 1, 0, 1], nx=2, ny=2, nz=2)

print("geo dim =", mesh.geo_dimension())
print("top dim =", mesh.top_dimension())
print("prism indices:\n", mesh.Cells[0].indices)  # Prism
print("tri indices:\n", mesh.Tri.indices)         # Face[0]
print("quad indices:\n", mesh.Quad.indices)       # Face[1]
print("edge indices:\n", mesh.Edges[0].indices)   # Segment
print("prism -> tri relation:\n", mesh.block.relations[("prism", "tri")].tgt_indices)
print("prism -> quad relation:\n", mesh.block.relations[("prism", "quad")].tgt_indices)

from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

mesh.add_plot(ax, alpha=0.5)

plt.show()
