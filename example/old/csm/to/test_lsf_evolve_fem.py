import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.decorator import cartesian

# Define the domain and generate the triangular mesh
domain = [0, 1, 0, 1]
ns = 100
mesh = TriangleMesh.from_box(domain, nx=ns, ny=ns)

# Define the finite element space
degree = 1
space = LagrangeFESpace(mesh, p=degree)

# Define the velocity field $u$ for the evolution
@cartesian
def velocity_field(p):
    x = p[..., 0]
    y = p[..., 1]
    u = np.zeros(p.shape)
    u[..., 0] = np.sin(np.pi*x)**2 * np.sin(2*np.pi*y)
    u[..., 1] = -np.sin(np.pi*y)**2 * np.sin(2*np.pi*x)
    return u

# Initial level set function $\phi0$ representing the circle
@cartesian
def circle(p):
    x = p[..., 0]
    y = p[..., 1]
    return np.sqrt((x-0.5)**2 + (y-0.75)**2) - 0.15

# Initialize the level set function
lsf = space.function(array=space.interpolate(circle))
lsf = np.array( lsf.reshape(ns+1, ns+1) )

h2 = 1.0 / ns  # Assuming a uniform grid for simplicity
dpx = ( np.roll(lsf, shift=(0, -1), axis=(0, 1)) - lsf ) / h2
dmx = ( lsf - np.roll(lsf, shift=(0, 1), axis=(0, 1)) ) / h2
dpy = ( np.roll(lsf, shift=(-1, 0), axis=(0, 1)) - lsf ) / h2
dmy = ( lsf - np.roll(lsf, shift=(1, 0), axis=(0, 1)) ) / h2

def visualize_differences(lsf, dpx, dmx, dpy, dmy):
    fig, axs = plt.subplots(3, 2, figsize=(12, 18))

    im_lsf = axs[0, 0].imshow(lsf, origin='lower', cmap='jet')
    axs[0, 0].set_title('Level Set Function - lsf')
    fig.colorbar(im_lsf, ax=axs[0, 0])

    # Leaving the top-right subplot empty for a balanced layout
    axs[0, 1].axis('off')

    im0 = axs[1, 0].imshow(dpx, origin='lower', cmap='jet')
    axs[1, 0].set_title('Forward Difference in X - dpx')
    fig.colorbar(im0, ax=axs[1, 0])

    im1 = axs[1, 1].imshow(dmx, origin='lower', cmap='jet')
    axs[1, 1].set_title('Backward Difference in X - dmx')
    fig.colorbar(im1, ax=axs[1, 1])
    
    im2 = axs[2, 0].imshow(dpy, origin='lower', cmap='jet')
    axs[2, 0].set_title('Forward Difference in Y - dpy')
    fig.colorbar(im2, ax=axs[2, 0])
    
    im3 = axs[2, 1].imshow(dmy, origin='lower', cmap='jet')
    axs[2, 1].set_title('Backward Difference in Y - dmy')
    fig.colorbar(im3, ax=axs[2, 1])

    plt.tight_layout()
    plt.show()

# Assuming lsf and dpx are already computed
visualize_differences(lsf, dpx, dmx, dpy, dmy)


# Compute velocity magnitude
v = np.linalg.norm(space.interpolate(velocity_field, dim=2), axis=1)
v = v.reshape(ns+1, ns+1)

# Define the evolve function
def evolve(v, lsf, h2, dt = 0.001, num = 100):
    # vFull = np.pad(v, ((1,1), (1,1)), mode='constant', constant_values=0)
    vFull = v

    for i in range(num):
        dpx = ( np.roll(lsf, shift=(0, -1), axis=(0, 1)) - lsf ) / h2 # 向前差分
        dmx = ( lsf - np.roll(lsf, shift=(0, 1), axis=(0, 1)) ) / h2  # 向后差分
        dpy = ( np.roll(lsf, shift=(-1, 0), axis=(0, 1)) - lsf ) / h2
        dmy = ( lsf - np.roll(lsf, shift=(1, 0), axis=(0, 1)) ) / h2
        
        lsf = lsf - dt * np.minimum(vFull, 0) * np.sqrt( np.minimum(dmx, 0)**2 + np.maximum(dpx, 0)**2 + np.minimum(dmy, 0)**2 + np.maximum(dpy, 0)**2 ) \
                  - dt * np.maximum(vFull, 0) * np.sqrt( np.maximum(dmx, 0)**2 + np.minimum(dpx, 0)**2 + np.maximum(dmy, 0)**2 + np.minimum(dpy, 0)**2 )
        print("lsf:", lsf)

        output_dir = './results/'
        mesh.nodedata['phi'] = lsf
        mesh.nodedata['velocity'] = v
        fname = output_dir + 'levelset_' + str(i).zfill(10) + '.vtu'
        mesh.to_vtk(fname=fname)

# Parameters
h2 = 1.0 / ns  # Assuming a uniform grid for simplicity
print("h2:", h2)

# Evolution
evolve(v, lsf, h2)
