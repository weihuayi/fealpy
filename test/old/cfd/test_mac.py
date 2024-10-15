import numpy as np
import matplotlib.pyplot as plt  
from fealpy.mesh.uniform_mesh_2d import UniformMesh2d
from fealpy.cfd import NSMacSolver
from fealpy.pde.taylor_green_pde import taylor_greenData 
from scipy.sparse import diags, lil_matrix
from scipy.sparse import vstack
import pytest

Re = 1
nu = 1/Re
pde = taylor_greenData(Re)
domain = pde.domain()


nx = 4
ny = 4
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
mesh = UniformMesh2d([0,nx,0,ny],h=(hx,hy),origin=(domain[0],domain[2]))
umesh = UniformMesh2d([0, nx, 0, ny-1], h=(hx, hy), origin=(domain[0], domain[2]+hy/2))
vmesh = UniformMesh2d([0, nx-1, 0, ny], h=(hx, hy), origin=(domain[0]+hx/2, domain[2]))
pmesh = UniformMesh2d([0, nx-1, 0, ny-1], h=(hx, hy), origin=(domain[0]+hx/2, domain[2]+hy/2))

def plot_mesh():
    solver = NSMacSolver(Re, mesh)
    umesh = solver.umesh
    vmesh = solver.vmesh
    pmesh = solver.pmesh
    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    mesh.find_node(axes, color='y')
    umesh.find_node(axes, color='r')
    vmesh.find_node(axes, color='b')
    pmesh.find_node(axes, color='g')
    plt.show()

def test_grad_ux():
    solver = NSMacSolver(Re,mesh)
    result = solver.grad_ux()
    result1 = diags([1, -1],[4, -4],(20,20), format='csr')
    result1 = result1/np.pi
    np.allclose(result.toarray(),result1.toarray())

def test_grad_vx():
    solver = NSMacSolver(Re,mesh)
    result = solver.grad_vx()
    result1 = diags([1, -1],[4, -4],(20,20), format='csr')
    index = np.arange(0,20)
    result1[index[:4],index[:4]] = 2
    result1[index[:4],index[:4]+4] = 2/3
    result1[index[-4:],index[-4:]] = -2
    result1[index[-4:],index[-4:]-4] = -2/3
    result1 = result1/np.pi
    np.allclose(result.toarray(),result1.toarray())

def test_grad_uy():
    solver = NSMacSolver(Re,mesh)
    result = solver.grad_uy()
    result1 = diags([1, -1],[1, -1],(20,20), format='lil')
    index = np.arange(0,20,4)
    result1[index, index] = 2
    result1[index, index+1] = 2/3
    result1[index[1:], index[1:]-1] = 0
    index = np.arange(4-1, 4, 4)
    result1[index, index] = -2
    result1[index, index-1] = -2/3
    result1[index[:-1], index[:-1]+1] = 0
    result1 = result1/np.pi
    np.allclose(result.toarray(),result1.toarray())

def test_grad_vy():
    solver = NSMacSolver(Re,mesh)
    result = solver.grad_vy()
    result1 = diags([1, -1],[1, -1],(20,20), format='lil')
    result1 = result1/np.pi
    np.allclose(result.toarray(),result1.toarray())

def test_Tuv():
    solver = NSMacSolver(Re,mesh)
    result = solver.Tuv()
    result1 = np.zeros((20,20))
    index = np.arange(16)
    result1[index,index+index//4] = 1
    result1[index,index+index//4+1] = 1
    result1[index,index+index//4-5] = 1
    result1[index,index+index//4-4] = 1
    result1 = result1/4
    np.allclose(result,result1)

def test_Tvu():
    solver = NSMacSolver(Re,mesh)
    result = solver.Tvu()
    result1 = np.zeros((20,20))
    arr = np.arange(0,20)
    split_array = np.array_split(arr,5)
    lists = [sub_array.tolist() for sub_array in split_array]
    num = len(lists)
    for i in range(num):
        i_array = np.ones_like(lists[i])
        result1[lists[i],lists[i]-i*i_array] = 1
        result1[lists[i],lists[i]-i*i_array-1] = 1
    for i in range(num-1):
        i_array = np.ones_like(lists[i])
        result1[lists[i],lists[i]-i*i_array+5*i_array] = 1
        result1[lists[i],lists[i]-i*i_array-1+5*i_array] = 1
    index = lists[-1][:3]
    N_array = np.ones_like(index)  
    result1[index,index] = 1
    result1[index,index+N_array] = 1 
    result1 = result1/4
    np.allclose(result,result1) 

def test_laplace_u():
    solver = NSMacSolver(Re,mesh)
    result = solver.laplace_u()
    result1 = diags([-4, 1, 1, 1, 1],[0, 1, -1, 4, -4],(20,20), format='lil')
    index = np.arange(0,20,4)
    result1[index, index] = -6
    result1[index[2:]-1, index[2:]-1] = -6
    result1[index[1:], index[1:]-1] = 0
    result1[index[2:]-1, index[2:]] = 0
    result1[index, index+1] = 4/3
    result1[index[2:]-1, index[2:]-2] = 4/3
    result1 = 4*result1/(np.pi**2)
    np.allclose(result.toarray(),result1.toarray()) 

def test_laplace_v():
    solver = NSMacSolver(Re,mesh)
    result = solver.laplace_v()
    result1 = diags([-1, 1, 1, 1, 1],[0, 1, -1, 5, -5],(20,20), format='lil')
    index = np.arange(0,20)
    result1[index[:5],index[:5]] = -6
    result1[index[-5:],index[-5:]] = -6
    result1[index[:5],index[:5]+5] = 4/3
    result1[index[-5:],index[-5:]-5] = 4/3
    result1 = 4*result1/(np.pi**2)
    np.allclose(result.toarray(),result1.toarray()) 

def test_grand_uxp():
    solver = NSMacSolver(Re,mesh)
    result = solver.grand_uxp()
    result1 = diags([1, -1],[0, -4],(16,16), format='lil')
    A = lil_matrix((4, 16))
    result1 = vstack([result1,A],format='lil')
    result1 = 2*result1/np.pi
    np.allclose(result.toarray(),result1.toarray()) 

def test_grand_vyp():
    solver = NSMacSolver(Re,mesh)
    result = solver.grand_vyp()
    result1 = diags([0],[0],(20,16), format='lil')
    arr = np.arange(0,20)
    split_array = np.array_split(arr,4)
    lists = [sub_array.tolist() for sub_array in split_array]
    num = len(lists)
    for i in range(num-1):
        i_array= np.ones_like(lists[i])
        result1[lists[i],lists[i]-i*i_array] = 1
        result1[lists[i],lists[i]-i*i_array-1] = -1
    index = lists[-1][:4]
    num_array = np.ones_like(index)
    result1[index,index-(num-1)*num_array] = 1
    result1[index,index-(num-1)*num_array-1] = -1
    result1 = 2*result1/np.pi
    np.allclose(result.toarray(),result1.toarray()) 
