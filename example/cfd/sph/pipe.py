import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import numpy as np
import time
import numba
from numba.typed import List 
dx = 0.05
dy = 0.05
rho0 = 1000
H = 0.92*np.sqrt(dx**2+dy**2)
dt = 0.001
#c0 = 10
gamma = 7
alpha = 0.1
eta = 0.001
maxstep = 20000
dtype = [("position", "float64", (2, )), 
         ("velocity", "float64", (2, )),
         ("rho", "float64"),
         ("mass", "float64"),
         ("pressure", "float64"),
         ("sound", "float64"),
         ("isBd", "bool"),
         ("isHd", "bool")]

#生成粒子位置
def initial_position(dx,dy):
    pp = np.mgrid[dx:1+dx:dx, dy:1:dy].reshape(2,-1).T

    x0 = np.arange(0,10,dx)
    x1 = np.arange(-dx/2,10+dx,dx)

    bp0 = np.column_stack((x0,np.zeros_like(x0)))
    bp1 = np.column_stack((x1,np.full_like(x1,-dy)))
    bp = np.vstack((bp0,bp1))
    dp0 = np.column_stack((x0,np.full_like(x0,1)))
    dp1 = np.column_stack((x1,np.full_like(x1,1+dy)))
    dp = np.vstack((dp0,dp1))
    bm = np.vstack((bp,dp))

    mp0 = np.column_stack((x0,np.full_like(x0,-2*dy)))
    mp1 = np.column_stack((x1,np.full_like(x1,-3*dy)))
    mp2 = np.column_stack((x0,np.full_like(x0,-4*dy)))
    mp = np.vstack((mp0,mp1,mp2))
    vp0 = np.column_stack((x0,np.full_like(x0,1+2*dy)))
    vp1 = np.column_stack((x1,np.full_like(x1,1+3*dy)))
    vp2 = np.column_stack((x0,np.full_like(x0,1+4*dy)))
    vp = np.vstack((vp0,vp1,vp2))
    hm = np.vstack((mp,vp))
    return pp,bm,hm

#初始化
pp,bm,hm = initial_position(dx,dy)
num_particles = bm.shape[0] + hm.shape[0]
particles = np.zeros(num_particles,dtype=dtype)
particles['rho'] = rho0
particles['position'] = np.vstack((bm,hm))
particles['isBd'][:bm.shape[0]] = True
particles['isBd'][bm.shape[0]:] = False
particles['isHd'][:bm.shape[0]] = False
particles['isHd'][bm.shape[0]:] = True

#可视化
color = np.where(particles['isBd'], 'red', np.where(particles['isHd'], 'green', 'blue'))
plt.scatter(particles['position'][:, 0], particles['position'][:, 1], c=color, s=5)
plt.grid(True)
plt.show()

#新增粒子
def add_particles_at_surface(particles,dy):
    y = np.arange(dy,1,dy)
    lp = np.column_stack((np.full_like(y, dx), y))
    new_particles = np.zeros(lp.shape[0],dtype=dtype)
    new_particles['rho'] = rho0
    new_particles['position'] = lp
    new_particles['velocity'] = np.array([2,0],dtype=np.float64)
    new_particles['isBd'] = False
    new_particles['isHd'] = False
    particles = np.concatenate((particles,new_particles), axis = 0)
    return particles



#删除粒子
def remove_particles(particles):
    condition = particles['position'][:, 0] <= 10.0
    particles = particles[condition]
    return particles


#寻找周围有贡献的粒子
def find_neighbors_within_distance(points, h):
    tree = cKDTree(points)
    neighbors = tree.query_ball_tree(tree, h)
    return neighbors

#核函数
@numba.jit(nopython=True)
def kernel(r, h):
    d = np.sqrt(np.sum(r**2, axis=-1))
    q = d/h
    if 0 <= q <= 1:
        val = 10*(1-(3*q**2)/2+(3*q**3)/4)/(7*np.pi*h**2)
    elif 1 <= q <= 2:
        val = 5*(2-q)**3/(14*np.pi*h**2)
    else:
        val = 0
    return val

@numba.jit(nopython=True)
def gradkernel(r, h):
    d = np.sqrt(np.sum(r**2))
    q = d/h
    if 0 < q <= 1:
        val = 10*(-3*q+(9*q**2)/4)/(7*np.pi*h**3)
        val /= d
    elif 1 <= q <= 2:
        val = -15*(2-q)**2/(14*np.pi*h**3)
        val /= d
    else:
        val = 0
    return r*val

#计算密度
@numba.jit(nopython=True)
def change_rho(particles, idx):
    num = particles['rho'].shape[0]
    position = particles["position"]
    velocity = particles["velocity"] 
    mass = dx*dy*particles['rho']
    result = np.zeros(num) 
    for i in range(num):
        for j in idx[i]:
            rij = position[i] - position[j]
            gk = gradkernel(rij, H) 
            vij = velocity[i] - velocity[j]
            result[i] += mass[j]*np.dot(gk,vij)
        particles['rho'][i] += dt * result[i]


@numba.jit(nopython=True)
def change_position(particles, idx):
    num = particles['rho'].shape[0]
    position = particles["position"]
    velocity = particles["velocity"] 
    rho = particles["rho"] 
    mass = dx*dy*rho
    result = np.zeros((num,2))
    for i in range(num):
        for j in idx[i]:
            rhoij = (rho[i] + rho[j])/2
            vij = velocity[j] - velocity[i]
            rij = position[i] - position[j]
            ke = kernel(rij, H)
            result[i] += 0.5*mass[j]*vij*ke/rhoij
        result[i] = velocity[i] + 0.5*result[i] 
    
    Btag = particles['isBd']==True
    Htag = particles['isHd']==True
    Ftag = ~Btag & ~Htag
    particles['position'][Ftag] += dt * result[Ftag] 

def grad_u(particles,idx):
    num = particles['rho'].shape[0]
    position = particles['position']
    rho = particles['rho']
    velocity = particles['velocity']
    mass = dx*dy*rho
    grad_u = np.zeros((num,2,2))
    for i in range(num):
        for j in idx[i]:
            rij = position[i] - position[j]
            gk = gradkernel(rij, H) 
            uj = velocity[j]
            graduj = np.outer(uj,gk)  #(2,2)
            grad_u[i] += mass[j]*graduj/rho[j]
    return grad_u

#计算压强
@numba.jit(nopython=True)
def change_p(particles,idx):
    v = particles['velocity']
    v2 = np.sum(v*v,axis=1)
    c2 = 100*np.max(v2)
    B = c2 * rho0 / gamma
    position = particles["position"]
    Btag = particles['isBd']==True
    Htag = particles['isHd']==True
    Ftag = ~Btag & ~Htag
    
    # 计算流体粒子的压强
    particles['pressure'][Ftag] = B * ((particles['rho'][Ftag]/rho0)**gamma - 1)
    pressure = particles['pressure']
    # 计算固壁粒子的压强
    for i in np.where(Btag)[0]:
        sum0 = 0
        sum1 = 0
        for j in idx[i]:
            if Ftag[j]:
                pj = pressure[j]
                rij = np.linalg.norm(position[i] - position[j])
                sum0 += pj*(2*H-rij)
                sum1 += 2*H-rij
            if sum0 != 0:
                pressure[i] = sum0/sum1
            else:
                pressure[i] = 0
    # 计算固壁h外虚粒子的压强
    for i in np.where(Htag)[0]:
        sumf = 0
        sumdf = 0
        tagf = 0
        sumd = 0
        tagd = 0
        di = np.abs(position[i,1])
        if di >= 1:
            di -= 1    
        for j in idx[i]:
            if Ftag[j]:
                sumf += pressure[j]
                tagf += 1
                djf = np.abs(position[j,1])
                if djf >= 0.5:
                    djf  = 1-djf
                sumdf += djf
                 
            elif  Btag[i]:
                sumd += pressure[j]
                tagd += 1 
        if tagd==0 or tagf==0:
            pressure[i] = 0
        else:
            pressure[i] = sumf/tagf + (1 + di*tagf/sumdf) * (sumd/tagd - sumf/tagf) 
    particles['pressure'][:] = pressure
    particles['sound'][:] = (B*gamma/rho0 * (particles['rho']/rho0)**(gamma-1))**0.5 

#更新速度
#@numba.jit(nopython=True)
def change_v(particles, idx, gradu):
    divu = gradu[:,0,0] + gradu[:,1,1]
    sigma = eta*(gradu+gradu.transpose(0,2,1))
    pressure = particles["pressure"]
    sigma[:,0,0] -= pressure  
    sigma[:,1,1] -= pressure 
    sigma[:,0,0] -= 2/3*eta*divu  
    sigma[:,1,1] -= 2/3*eta*divu  

    num = particles['rho'].shape[0]
    rho = particles['rho']
    mass = dx*dy*rho  ## 用哪一步密度计算质量
    position = particles["position"]
    velocity = particles["velocity"]
    sound = particles["sound"]
    
    Btag = particles['isBd']==True
    Htag = particles['isHd']==True
    Ftag = ~Btag & ~Htag
    #固壁外虚粒子
    for i in range(num):
        if Ftag[i]:
            result = np.array([0,0],dtype=np.float64)
            for j in idx[i]:
                val = sigma[i]/rho[i]**2 + sigma[j]/rho[j]**2
                rij = position[i] - position[j]
                gk = gradkernel(rij, H)
                vij = velocity[i] - velocity[j]
                pij = 0
                if np.dot(rij, vij) < 0:
                    pij = -alpha * (sound[i] + sound[j])/2
                    pij *= H*np.dot(rij,vij)/(np.dot(rij,rij)+0.01*H*H)
                    pij += 2*np.dot(vij,rij)
                    pij /= (rho[i] + rho[j])/2
                val[0,0] -= pij
                val[1,1] -= pij
                result += mass[j]*val@gk
            particles['velocity'][i] += dt*result
        elif Htag[i]:
            sumdf = 0
            sumvf = np.array([0,0],dtype=np.float64)
            tagf = 0
            dd = np.abs(position[i,1])
            if dd >= 1:
                dd -= 1    
            for j in idx[i]:
                if Ftag[j]:
                    sumvf += velocity[j]
                    tagf += 1
                    djf = np.abs(position[j,1])
                    if djf >= 0.5:
                        djf  = 1-djf
                    sumdf += djf
            if tagf==0:
                particles['velocity'][i] = 0
            else:
                particles['velocity'][i] = -(tagf*dd/sumdf)*(sumvf/tagf)
        elif Btag[i]:
                particles['velocity'][i] = 0

    
#密度重置
@numba.jit(nopython=True)
def rein_rho(particles, idx):
    num = particles['rho'].shape[0]
    rho = particles['rho']
    mass = dx*dy*rho  ## 用哪一步密度计算质量
    position = particles["position"]
    vol = mass/rho
    A = np.zeros((num,3,3))
    for i in range(num):
        for j in idx[i]:
            rij = position[i] - position[j]
            wij = kernel(rij, H)
            Abar = np.zeros((3,3)) 
            Abar[0,1] = rij[0] 
            Abar[0,2] = rij[1]
            Abar[1,2] = rij[0]*rij[1]
            Abar += Abar.T
            Abar[0,0] = 1
            Abar[1,1] = rij[0]**2
            Abar[2,2] = rij[1]**2
            A[i] += Abar*wij*vol[j]
    for i in range(num):
        particles['rho'][i] = 0
        invA = np.linalg.inv(A[i])
        if invA is None: #检查逆矩阵是否存在
            #当逆矩阵不存在时
            sum_numer = 0
            sum_denom = 0
            for j in idx[i]:
                rij = position[i] - position[j]
                wij = kernel(rij,H)
                mb = mass[j]
                rho_b = particles['rho'][j]
                sum_numer += mb*wij
                sum_denom += (mb*wij)/rho_b
            if sum_denom != 0:
                particles['rho'][i] = sum_numer/sum_denom
        else:
            for j in idx[i]:
                rij = position[i] - position[j]
                wij = kernel(rij, H)
                wmls = invA[0,0] + invA[1,0]*rij[0] + invA[2,0]*rij[1]
                particles['rho'][i] += wij*wmls*mass[j]

def draw(particles, i):
    plt.clf()
    tag = ~particles['isBd'] & ~particles['isHd']
    color = np.where(particles['isBd'], 'red', np.where(particles['isHd'], 'green', 'blue'))
    #plt.scatter(particles['position'][:, 0], particles['position'][:, 1], c=color, cmap='jet', s=5)
    plt.scatter(particles['position'][:, 0], particles['position'][:, 1], c=color, s=5)
    #plt.colorbar(cmap='jet')
    plt.clim(0, 1)  # 颜色范围
    plt.title(f"Time Step: {i}")
    #plt.show()
    plt.pause(0.01)

for i in range(200):
    print("i:", i)
    particles = add_particles_at_surface(particles,dy) 
    idx = find_neighbors_within_distance(particles["position"], 2*H)
    idx = List([np.array(neighbors) for neighbors in idx])
    change_rho(particles, idx)
    change_p(particles,idx)
    gradu = grad_u(particles, idx)
    change_v(particles,idx,gradu)
    change_position(particles,idx)
    '''
    if i%20==0 and i!=0:
       rein_rho(particles, idx)
    '''
    particles = remove_particles(particles)
    draw(particles, i)
