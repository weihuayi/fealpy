import jax.numpy as jnp
from jax import grad
from fealpy.jax.sph.kernel_function import QuinticKernel
from scipy.spatial import cKDTree 
import matplotlib.pyplot as plt

domain = [1,1]
nx = 4
ny = 4
quintic_kernel = QuinticKernel(h=domain[0]/nx,dim=2)

#生成测试点
def test_nodes(domain,nx,ny):
    x = jnp.linspace(0,domain[0],nx+1)
    y = jnp.linspace(0,domain[1],ny+1)
    X, Y = jnp.meshgrid(x,y)
    test_nodes = jnp.stack([X.ravel(),Y.ravel()],axis=-1)
    return test_nodes
test_nodes = test_nodes(domain,nx,ny) 

#生成正确值的点
def ture_nodes(domain,ny,nx):
    x = jnp.linspace(domain[0]/(2*nx),domain[0]-(domain[0]/(2*nx)),nx)
    y = jnp.linspace(domain[1]/(2*ny),domain[1]-(domain[1]/(2*ny)),ny)
    X, Y = jnp.meshgrid(x,y)
    ture_nodes = jnp.stack([X.ravel(),Y.ravel()],axis=-1)
    return ture_nodes
ture_nodes = ture_nodes(domain,nx,ny) 

#寻找附近的测试点
def find_neighbors(ture_nodes, test_nodes, h):
    tree = cKDTree(test_nodes)
    neighbor = [tree.query_ball_point(point, h) for point in ture_nodes]
    return neighbor
neighbor = find_neighbors(ture_nodes, test_nodes, domain[0]/nx)

#计算核函数
def kernel_function(kernel,ture_nodes,test_nodes,neighbor):
    kernel_function = []
    for i, indices in enumerate(neighbor):
        neighbor_nodes = test_nodes[jnp.array(indices)]
        ture_node = ture_nodes[i,:]
        a = ture_node - neighbor_nodes
        dist = jnp.sqrt(a[:,0]**2+a[:,1]**2)
        for i in range(dist.shape[0]):
            r = dist[i]
            k = quintic_kernel.value(r)
            kernel_function.append(k)
    return kernel_function
kernel = jnp.reshape(jnp.array(kernel_function(quintic_kernel,ture_nodes,test_nodes,neighbor)),(-1,4))

#测试函数
def test_function(nodes):
    x = nodes[:,0]
    y = nodes[:,1]
    return x * y

true_function = test_function(ture_nodes) 
test_function = test_function(test_nodes)
m = domain[0] * domain[1]
rho = test_nodes.shape[0]

#计算误差
errors = []
for i in range(true_function.shape[0]):
    true = true_function[i]
    neighbor_idx = jnp.array(neighbor[i])
    test = jnp.dot((m/rho)*jnp.take(test_function,neighbor_idx),kernel[i])
    errors.append(true - test)
average_error = jnp.sum(jnp.array(errors))/true_function.shape[0]
print(average_error)
