import jax.numpy as jnp
from jax import grad, jit, vmap
import jax.random
from fealpy.jax.sph.kernel_function import QuinticKernel, WendlandC2Kernel
from fealpy.jax.sph.node_set import NodeSet
from scipy.spatial import cKDTree 
import matplotlib.pyplot as plt

domain = [1,1]
nx = 20
ny = 20
h = 1.5*domain[0]/nx
num_node = 1000
quintic_kernel = WendlandC2Kernel(h=h,dim=2)

#生成测试点
def ture_node(domain, ny, nx):
    x = jnp.linspace(domain[0]/(2*nx),domain[0]-(domain[0]/(2*nx)),nx)
    y = jnp.linspace(domain[1]/(2*ny),domain[1]-(domain[1]/(2*ny)),ny)
    X, Y = jnp.meshgrid(x,y)
    ture_node = jnp.stack([X.ravel(),Y.ravel()],axis=-1)
    return ture_node
ture_node = jit(ture_node, static_argnums=(1, 2))(domain, nx, ny) 

#在区域内随机撒粒子
key = jax.random.PRNGKey(6)
test_node = jax.random.uniform(key,shape=(num_node,2),minval=0.0,maxval=1.0) * jnp.array(domain)

#测试函数
def function(node):
    x = node[:,0]
    y = node[:,1]
    return x*y
def function_grad(node):
    x = node[:,0]
    y = node[:,1]
    return y, x

#寻找测试点附近的粒子
def find_neighbors(ture_node, test_node, h):
    tree = cKDTree(test_node)
    neighbor = [tree.query_ball_point(point, h) for point in ture_node]
    neighbor_num = jnp.array([len(neigh) for neigh in neighbor])
    return neighbor, neighbor_num
neighbor,neighbor_num = find_neighbors(ture_node,test_node,h)

true_function = function(ture_node) 
test_function = function(test_node)
dx, dy = function_grad(ture_node)
true_function_grad = jnp.column_stack((dx,dy))
m = jnp.pi * h**2

#插值
def interpolate(kernel, ture_node, test_node, neighbor):
    kernel_function = []
    kernel_grad = []
    for i, indices in enumerate(neighbor):
        rho = len(indices) 
        neighbor_nodes = test_node[jnp.array(indices)]
        ture_nodes = ture_node[i,:]
        reduce = ture_nodes - neighbor_nodes
        dist = jnp.sqrt(reduce[:,0]**2+reduce[:,1]**2)
        w = quintic_kernel.value(dist)
        test = (m/rho)*test_function[jnp.array(indices)]*w
        result = jnp.sum(test)
        kernel_function.append(result)

        dw = vmap(quintic_kernel.grad_value)(dist)[:,jnp.newaxis]*reduce
        test_grad = ((m/rho)*test_function[jnp.array(indices)])[:,jnp.newaxis]*dw
        grad = jnp.array([jnp.sum(test_grad[:,0]),jnp.sum(test_grad[:,1])])
        kernel_grad.append(grad)
        
    return kernel_function, kernel_grad
kernel = jnp.array(interpolate(quintic_kernel,ture_node,test_node,neighbor)[0])
error = jnp.sum(jnp.abs(true_function-kernel))/(nx*ny)
#print(jnp.argmax(jnp.abs(true_function-kernel)))
#print(error)

grad_kernel = jnp.array(interpolate(quintic_kernel,ture_node,test_node,neighbor)[1])
grad_error = jnp.abs(true_function_grad-grad_kernel)
dx_error = jnp.sum(grad_error[:,0])/(nx*ny)
dy_error = jnp.sum(grad_error[:,1])/(nx*ny)
#print(jnp.max(jnp.abs(rue_function_grad-grad_kernel)))
#print(dx_error)
#print(dy_error)

'''
error_array = jnp.abs(true_function-kernel)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(ture_node[:, 0], ture_node[:, 1], c='blue', label='True Nodes')
plt.scatter(test_node[:, 0], test_node[:, 1], c='red', label='Test Nodes')
plt.legend()
plt.subplot(1, 2, 2)
x = jnp.linspace(domain[0]/(2*nx),domain[0]-(domain[0]/(2*nx)),nx)
y = jnp.linspace(domain[1]/(2*ny),domain[1]-(domain[1]/(2*ny)),ny)
X, Y = jnp.meshgrid(x, y)
plt.scatter(X.flatten(), Y.flatten(), c=error_array, cmap='viridis')
plt.colorbar(label='Value')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
'''