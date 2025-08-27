import argparse 
import matplotlib.pyplot as plt


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        The example using lagrange finite element method to solve the dld
        microfluidic chip problem.
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help="the backend of fealpy, can be 'numpy', 'torch', 'tensorflow' or 'jax'.")

parser.add_argument('--pde',
    default = 1, type = str,
    help = "Name of the PDE model, default is exp0001")

parser.add_argument('--space_degree',
        default=2, type=int,
        help='Degree of Lagrange finite element space, default is 2.')

parser.add_argument('--pbar_log',
                    default=True, type=bool,
                    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

options = vars(parser.parse_args())

from fealpy.backend import bm

bm.set_backend(options['backend'])

from fealpy.mesh import LagrangeTriangleMesh, TriangleMesh
from fealpy.fem import DLDMicrofluidicChipLFEMModel
from fealpy.mmesh.tool import high_order_meshploter


box = [-1.0, 1.0, -1.0, 1.0]
holes = [[-0.5, 0.5, 0.2], [0.5, 0.5, 0.2], [-0.5, -0.5, 0.2], [0.5, -0.5, 0.2]]

# mesh = LagrangeTriangleMesh.from_box_with_circular_holes(box=box, holes=holes, p=1)
mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=5, ny=5)

model = DLDMicrofluidicChipLFEMModel(options)
model.set_space_degree(options['space_degree'])
model.set_pde(1)
model.set_init_mesh(mesh)
model.setup()
model.run['uniform_refine'](5)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# high_order_meshploter(ax, mesh)
# plt.show()
