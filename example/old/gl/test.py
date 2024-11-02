import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh
from PIL import Image

mesh = TriangleMesh.from_box([0, 1920, 0, 1080])
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes, alpha=0.5)
img = Image.open("/home/cbtxs/data/src_1.jpg")
plt.imshow(img, extent=[0, 1920, 0, 1080])
plt.show()

