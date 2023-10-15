import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import numpy as np
import time
import numba
from numba.typed import List 
dx = 1.25e-4
H = 1.5*dx
dt = 1e-6
uin = np.array([5.0, 0.0])
uwall = np.array([0.0, 0.0])
domain=[0,0.05,0,0.005]
init_domain=[0.0+0.02,0.005+0.02,0,0.005]

rho0 = 737.54
#Tem = 500
mu0 = 938.4118
tau = 16834.4
powern = 0.3083
B = 5.914e7

eta = 0.5 #动量方程中的参数
c1 = 0.0894

maxstep = 20000
dtype = [("position", "float64", (2, )), 
         ("velocity", "float64", (2, )),
         ("rho", "float64"),
         ("mass", "float64"),
         ("pressure", "float64"),
         ("sound", "float64"),
         ("isBd", "bool"),
         ("isGate", "bool"),
         ("isHd", "bool")]

#生成粒子位置
def initial_position(dx):
    #fluid particles
    fp = np.mgrid[init_domain[0]:init_domain[1]:dx, \
            init_domain[2]+dx:init_domain[3]:dx].reshape(2,-1).T
     
    #wall particles 
    x0 = np.arange(domain[0],domain[1],dx)

    bwp = np.column_stack((x0,np.full_like(x0,domain[2])))
    uwp = np.column_stack((x0,np.full_like(x0,domain[3])))
    wp = np.vstack((bwp,uwp))

    #dummy particles
    bdp = np.mgrid[domain[0]:domain[1]:dx, \
            domain[2]-dx:domain[2]-dx*4:-dx].reshape(2,-1).T
    udp = np.mgrid[domain[0]:domain[1]:dx, \
            domain[3]+dx:domain[3]+dx*3:dx].reshape(2,-1).T
    dp = np.vstack((bdp,udp))
    
    # gate particles
    gp = np.mgrid[-dx:-dx-4*H:-dx, \
            init_domain[2]+dx:init_domain[3]:dx].reshape(2,-1).T
    return fp,wp,dp,gp

#核函数
@numba.jit(nopython=True)
def kernel(r):
    d = np.linalg.norm(r)
    q = d/H
    val = 7 * (1-q/2)**4 * (2*q+1) / (4*np.pi*H**2)
    return val

@numba.jit(nopython=True)
def gradkernel(r):
    d = np.linalg.norm(r)
    q = d/H
    val = -(35 * q * (1-q/2)**3)/(4*np.pi*H**3)
    if d!=0:
        val /= d
    else:
        val = 0
    return val*r
#寻找周围有贡献的粒子
def find_neighbors_within_distance(points, h):
    tree = cKDTree(points)
    neighbors = tree.query_ball_tree(tree, h)
    return neighbors

#计算墙粒子外推速度
@numba.jit(nopython=True)
def wall_extrapolation(particles, idx):
    position = particles["position"]
    velocity = particles['velocity']
    mass = particles["mass"]  
    rho = particles['rho']
    Btag = particles['isBd']
    Htag = particles['isHd']
    Ftag = ~Btag & ~Htag
    result = np.zeros_like(velocity)
    for i in np.where(Btag)[0]:
        sum0 = np.array([0.0, 0.0])
        sum1 = 0
        for j in idx[i]:
            if Ftag[j]:
                xij = position[i] - position[j]
                uj = velocity[j]
                wij = kernel(xij)
                sum0 += mass[j] * uj * wij /rho[j] 
                sum1 += mass[j] * wij /rho[j]
            if sum1 != 0:
                result[i] = 2*uwall - sum0/sum1
            else:
                result[i] = 0
    return result[Btag]

#新增粒子
def gate_change(particles): 
    Gtag = particles['isGate']
    particles['position'][Gtag] += dt*particles['velocity'][Gtag]
    con = particles['position'][:,0] >= domain[0]
    tag = con & Gtag
    particles['isGate'][tag] = False
    if np.sum(tag) != 0:
        print("添加一排")
        y = np.arange(domain[2]+dx, domain[3], dx) 
        lp = np.column_stack((np.full_like(y, domain[0]-4*H), y))
        new_particles = np.zeros(lp.shape[0],dtype=dtype)
        new_particles['rho'] = rho0
        new_particles['position'] = lp
        new_particles['velocity'] = uin
        new_particles['isGate'] = True
        particles = np.concatenate((particles,new_particles), axis = 0)
    return particles

def draw(particles, i):
    plt.clf()
    Btag = particles['isBd']
    Htag = particles['isHd']
    Gtag = particles['isGate']
    color = np.where(Btag, 'red', np.where(Htag, 'green', np.where(Gtag, 'black', 'blue')))
    #c = particles['rho']
    #c = particles['pressure']
    #plt.scatter(particles['position'][:, 0], particles['position'][:, 1], c=c, cmap='jet', s=5)
    plt.scatter(particles['position'][:, 0], particles['position'][:, 1], c=color, s=5)
    #plt.colorbar(cmap='jet')
    plt.title(f"Time Step: {i}")
    #plt.show()
    plt.pause(0.1)

#初始化
fp,wp,dp,gp = initial_position(dx)
num_particles = gp.shape[0] + fp.shape[0] + wp.shape[0] + dp.shape[0]
particles = np.zeros(num_particles,dtype=dtype)
particles['rho'] = rho0
particles['position'] = np.vstack((fp,wp,dp,gp))
particles['mass'] = dx**2 * rho0
particles['isBd'][fp.shape[0]:fp.shape[0]+wp.shape[0]] = True
particles['isHd'][fp.shape[0]+wp.shape[0]:-gp.shape[0]] = True
particles['isGate'][-gp.shape[0]:] = True
Btag = particles['isBd']
Htag = particles['isHd']
Gtag = particles['isGate']
Ftag = ~Btag & ~Htag & ~Gtag


#生成墙粒子和虚粒子关系
tree = cKDTree(particles['position'][Btag])
dummy_idx = tree.query(particles['position'][Htag], k=1)[1]

#初始速度
particles['velocity'][Ftag] = uin
particles['velocity'][Btag] = uwall
particles['velocity'][Gtag] = uin

idx = find_neighbors_within_distance(particles["position"], 2*H)
idx = List([np.array(neighbors) for neighbors in idx])
particles['velocity'][Htag] = wall_extrapolation(particles,idx)[dummy_idx]


#可视化
color = np.where(Btag, 'red', np.where(Htag, 'green', np.where(Gtag, 'black', 'blue')))
#c = particles['pressure']
plt.scatter(particles['position'][:, 0], particles['position'][:, 1], c=color, s=5)
#plt.scatter(particles['position'][:, 0], particles['position'][:, 1] ,c=c,cmap='jet' ,s=5)
plt.colorbar(cmap='jet')
plt.grid(True)
plt.show()



for i in range(200):
    print("i:", i)
    particles = gate_change(particles) 
    
    #更新标签
    Btag = particles['isBd']
    Htag = particles['isHd']
    Gtag = particles['isGate']
    Ftag = ~Btag & ~Htag & ~Gtag
    
    #更新索引
    particles['position'][Ftag] += 2*dt*particles['velocity'][Ftag]
    draw(particles, i)
