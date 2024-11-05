import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import numpy as np
import time
import numba
from numba.typed import List 
dx = 1.25e-4
H = 1.5*dx
dt = 1e-7
uin = np.array([5.0, 0.0])
uwall = np.array([0.0, 0.0])
domain=[0,0.05,0,0.005]
init_domain=[0.0,0.005,0,0.005]

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
         ("isGate", "bool"),
         ("isBd", "bool"),
         ("isHd", "bool")]


#新增粒子
def gate_change(particles): 
    Gtag = particles['isGate']
    particles['position'][Gtag] += dt*particles['velocity'][Gtag]
    con = particles['position'][:,0] >= domain[0]
    tag = con & Gtag
    particles['isGate'][tag] = False
    if np.sum(tag) != 0:
        y = np.arange(domain[2]+dx, domain[3], dx) 
        lp = np.column_stack((np.full_like(y, domain[0]-4*H), y))
        new_particles = np.zeros(lp.shape[0],dtype=dtype)
        new_particles['rho'] = rho0
        new_particles['position'] = lp
        new_particles['velocity'] = uin
        new_particles['mass'] = rho0 * dx* (domain[3]-domain[2])/lp.shape[0]
        new_particles['isGate'] = True
        particles = np.concatenate((particles,new_particles), axis = 0)
    return particles

'''
#删除粒子
def remove_particles(particles):
    condition = particles['position'][:, 0] <= doamin[1]
    particles = particles[condition]
    return particles
'''

#寻找周围有贡献的粒子
def find_neighbors_within_distance(points, h):
    tree = cKDTree(points)
    neighbors = tree.query_ball_tree(tree, h)
    return neighbors

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

#计算压强
@numba.jit(nopython=True)
def change_p(particles,idx):
    position = particles['position']
    rho = particles['rho']
    mass = particles['mass']
    Btag = particles['isBd']
    Htag = particles['isHd']
    Gtag = particles['isGate']
    Ftag = ~Btag & ~Htag & ~Gtag
    FGtag = ~Btag & ~Htag 
    # 计算流体粒子的压强
    particles['pressure'][FGtag] = B * (np.exp((1-rho0/rho[FGtag])/c1) -1)
    pressure = particles['pressure']
    particles['sound'][FGtag] = np.sqrt(B * rho0/(c1*rho[FGtag]**2) \
                                * np.exp((1-rho0/rho[FGtag])/c1))
    # 计算固壁粒子的压强
    for i in np.where(Btag)[0]:
        sum0 = 0
        sum1 = 0
        for j in idx[i]:
            if FGtag[j]:
                xij = position[i] - position[j]
                pj = pressure[j]
                wij = kernel(xij)
                sum0 += mass[j] * pj * wij /rho[j] 
                sum1 += mass[j] * wij /rho[j]
            if sum1 != 0:
                particles['pressure'][i] = sum0/sum1
            else:
                particles['pressure'][i] = 0
     
    # 计算固壁外虚粒子的压强
    particles['pressure'][Htag] = particles['pressure'][Btag][dummy_idx] 


#计算密度
@numba.jit(nopython=True)
def continue_equation(particles, idx, rho):
    num = particles.shape[0]
    position = particles["position"]
    velocity = particles["velocity"]
    mass = particles['mass']
    pressure = particles["pressure"]
    c = particles["sound"]
    Btag = particles['isBd']
    Htag = particles['isHd']
    Gtag = particles['isGate']
    Ftag = ~Btag & ~Htag & ~Gtag
    change_rho = np.zeros_like(rho)
    for i in np.where(Ftag)[0]:
        result = 0
        for j in idx[i]:
            if i!=j:
                xij = position[i] - position[j]
                gwij = gradkernel(xij)
                Aij = np.zeros((2,2))
                for k in idx[i]:
                    xik = position[i] - position[k]
                    gwik = gradkernel(xik) 
                    Aij += mass[k]/rho[k] * np.outer(-xik,gwik)
                condA = np.linalg.cond(Aij)
                if condA <1e15:
                    gwij = np.linalg.inv(Aij) @ gwij
                rij = np.linalg.norm(xij)
                temp = velocity[i] - velocity[j]
                rhoc_bar = 0.25*(rho[i]+rho[j])*(c[i]+c[j])
                temp += (pressure[i] - pressure[j]) * xij / (rhoc_bar*rij) 
                temp1 = np.dot(temp, gwij)
                result += mass[j]/rho[j] * temp1
        change_rho[i] = rho[i]*result
    return change_rho       


#计算墙粒子外推速度
@numba.jit(nopython=True)
def wall_extrapolation(particles, idx, velocity):
    position = particles["position"]
    mass = particles["mass"]  
    rho = particles['rho']
    Btag = particles['isBd']
    Htag = particles['isHd']
    Gtag = particles['isGate']
    Ftag = ~Btag & ~Htag & ~Gtag
    FGtag = ~Btag & ~Htag 
    result = np.zeros_like(velocity)
    for i in np.where(Btag)[0]:
        sum0 = np.array([0.0, 0.0])
        sum1 = 0
        for j in idx[i]:
            if FGtag[j]:
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


#计算粘度
@numba.jit(nopython=True)
def mu_wlf(particles,idx):
    num = particles.shape[0]
    position = particles['position']
    rho = particles['rho']
    velocity = particles['velocity']
    mass = particles['mass']
    mu = np.zeros_like(rho)
    for i in range(num):
        D = np.zeros((2,2))
        for j in idx[i]:
            xij = position[i] - position[j]
            gwij = gradkernel(xij) 
            uij = velocity[i] - velocity[j]
            graduij = np.outer(uij,gwij)  #(2,2)
            D += - 0.5*mass[j]/rho[j] * (graduij+graduij.T)
        #wlf
        gamma = np.sqrt(np.sum(2*D*D))
        mu[i] = mu0/(1+(mu0*gamma/tau)**(1-powern))
    return mu


#计算加速度
@numba.jit(nopython=True)
def momentum_equation(particles, idx, velocity):
    num = particles.shape[0]
    position = particles["position"]
    pressure = particles["pressure"]
    c = particles["sound"]
    rho = particles['rho']
    mass = particles["mass"]  
    mu = mu_wlf(particles, idx)

    Btag = particles['isBd']
    Htag = particles['isHd']
    Gtag = particles['isGate']
    Ftag = ~Btag & ~Htag & ~Gtag
    change_velocity = np.zeros_like(velocity) 
    for i in np.where(Ftag)[0]:
        sum0 = np.array([0.0, 0.0])
        sum1 = np.array([0.0, 0.0])
        for j in idx[i]:
            if i!=j :
                xij = position[i] - position[j]
                gwij = gradkernel(xij)
     
                Aij = np.zeros((2,2))
                for k in idx[i]:
                    xik = position[i] - position[k]
                    gwik = gradkernel(xik)
                    Aij += mass[k]/rho[k] * np.outer(-xik,gwik)
                condA = np.linalg.cond(Aij)
                if condA <1e15:
                    gwij = np.linalg.inv(Aij) @ gwij

                rij = np.linalg.norm(xij)
                uij = velocity[i] - velocity[j]
                rhoc_bar = 0.25*(rho[i]+rho[j])*(c[i]+c[j]) 
                beta = min(eta*((mu[i]+mu[j])/rij), rhoc_bar)
                
                temp_sum0 = pressure[i] + pressure[j]
                temp_sum0 -= beta*np.dot(xij,uij)/rij
                temp_sum0 *= mass[j]/(rho[i]*rho[j])
                sum0 += gwij*temp_sum0 

                temp_sum1 = np.dot(xij, gwij)/(rij**2 + (0.01*H)**2)
                temp_sum1 *= mass[j]*(mu[i] + mu[j])/(rho[i]*rho[j]) 
                sum1 += uij*temp_sum1

        change_velocity[i] = -sum0 + sum1
    return change_velocity        

@numba.jit(nopython=True)
def free_surface(particles, idx):
    num = particles['rho'].shape[0]
    velocity = particles["velocity"]
    position = particles['position']
    rho = particles["rho"]
    mass = particles['mass']
    Btag = particles['isBd']
    Htag = particles['isHd']
    Gtag = particles['isGate']
    Ftag = ~Btag & ~Htag & ~Gtag
    FreeTag = Ftag.copy()
    FreeTag[:] = False
    # 找到所有精确自由面粒子 
    for i in np.where(Ftag)[0]:
        # 算粒子浓度及其梯度
        C = 0
        gradC = np.array([0.0,0.0])
        for j in idx[i]:
            xij = position[i] - position[j]
            wij = kernel(xij)
            gwij = gradkernel(xij)
            C += mass[j] * wij /rho[j]
            gradC += mass[j] * gwij /rho[j]
        ''''
        if C < 0.85:
            print("asd")
            FreeTag[i] = True
            As = np.zeros((2,2))
            for k in idx[i]:
                xik = position[i] - position[k]
                gwik = gradkernel(xik)
                As += mass[k]/rho[k] * np.outer(xik,gwik) # 与论文不一样
            
            normal = (As @ gradC)/np.linalg.norm(As @ gradC)
            print(normal)
        '''
        #找界面粒子
        if C < 0.85:
            As = np.zeros((2,2))
            for k in idx[i]:
                xik = position[i] - position[k]
                gwik = gradkernel(xik)
                As += mass[k]/rho[k] * np.outer(xik,gwik)
            
            normal = (As @ gradC)/np.linalg.norm(As @ gradC)
            rotate = np.array([[0,-1],[1,0]],dtype=np.float64)
            perpen = rotate @ normal
            positionT = position[i] + normal*H
            
            for k in np.where(Ftag)[0]:
                cond1 = np.linalg.norm(position[k] - position[i]) >= np.sqrt(2)*H
                cond2 = np.linalg.norm(positionT-position[k]) < H
                
                cond3 = np.linalg.norm(position[i]-position[k]) < np.sqrt(2)*H
                cond4 = (np.abs(np.dot(positionT-position[k],normal)) + \
                       np.abs(np.dot(positionT-position[k],perpen))) < H
                
                if  (cond1 & cond2) or (cond3 & cond4):
                    FreeTag[i] = False
                    break
                FreeTag[i] = True
    for m in np.where(FreeTag)[0]:
        for n in idx[m]:
            if Ftag[n]:
                FreeTag[n] = True
    return FreeTag 

@numba.jit(nopython=True)
def shifting(particles, idx, position, FreeTag):
    num = particles['rho'].shape[0]
    velocity = particles["velocity"]
    rho = particles["rho"]
    mass = particles['mass']
    result = np.zeros_like(position)
    Btag = particles['isBd']
    Htag = particles['isHd']
    Gtag = particles['isGate']
    Ftag = ~Btag & ~Htag & ~Gtag
    InnerTag = ~FreeTag & Ftag
    #自由面粒子位移 
    for i in np.where(FreeTag)[0]:
        gradC = np.array([0.0,0.0])
        As = np.zeros((2,2))
        for j in idx[i]:
            xij = position[i] - position[j]
            wij = kernel(xij)
            gwij = gradkernel(xij)
            gradC += mass[j] * gwij /rho[j] 
            As += mass[j]/rho[j] * np.outer(-xij,gwij) 
        normal = (As @ gradC)/np.linalg.norm(As @ gradC)
        result[i] = -5*H*(np.eye(2) - np.outer(normal,normal))*gradC*dt
        result[i] *= np.linalg.norm(velocity[i]) 
    
    #内部粒子位移
    for i in np.where(InnerTag)[0]:
        gradC = np.array([0.0,0.0])
        for j in idx[i]:
            xij = position[i] - position[j]
            wij = kernel(xij)
            gwij = gradkernel(xij)
            gradC += mass[j] * gwij /rho[j]  
        result[i] = -5*H*gradC*dt
        result[i] *= np.linalg.norm(velocity[i]) 
    return result
    
@numba.jit(nopython=True)
def change_position(particles, idx, position):
    num = particles['rho'].shape[0]
    velocity = particles["velocity"] 
    rho = particles["rho"] 
    mass = particles['mass']
    result = np.zeros_like(position)
    Btag = particles['isBd']
    Htag = particles['isHd']
    Gtag = particles['isGate']
    Ftag = ~Btag & ~Htag & ~Gtag
    for i in np.where(Ftag)[0]:
        temp = np.array([0.0, 0.0])
        for j in idx[i]:
            rhobar = (rho[i] + rho[j])/2
            vji = velocity[j] - velocity[i]
            xij = position[i] - position[j]
            wij = kernel(xij)
            temp += mass[j]*vji*wij/rhobar
        result[i] = velocity[i] + 0.5*temp
    return result

@numba.jit(nopython=True)
def change_position(particles, idx, position):
    num = particles['rho'].shape[0]
    velocity = particles["velocity"] 
    rho = particles["rho"] 
    mass = particles['mass']
    result = np.zeros_like(position)
    Btag = particles['isBd']
    Htag = particles['isHd']
    Gtag = particles['isGate']
    Ftag = ~Btag & ~Htag & ~Gtag
    for i in np.where(Ftag)[0]:
        temp = np.array([0.0, 0.0])
        for j in idx[i]:
            rhobar = (rho[i] + rho[j])/2
            vji = velocity[j] - velocity[i]
            xij = position[i] - position[j]
            wij = kernel(xij)
            temp += mass[j]*vji*wij/rhobar
        result[i] = velocity[i] + 0.5*temp
    return result

def draw(particles, i):
    plt.clf()
    Btag = particles['isBd']
    Htag = particles['isHd']
    Gtag = particles['isGate']
    color = np.where(Btag, 'red', np.where(Htag, 'green', np.where(Gtag, 'black', 'blue')))
    c = particles['velocity'][:,0]
    plt.figure(figsize=(20,2))
    #c = particles['rho']
    #c = particles['pressure']
    
    plt.scatter(particles['position'][:, 0], particles['position'][:, 1], c=c, cmap='jet', s=5)
    #plt.scatter(particles['position'][:, 0], particles['position'][:, 1], c=color, s=5)
    plt.colorbar(cmap='jet')
    plt.clim(-7,7)
    
    plt.title(f"Time Step: {i}")
    fname = './' + 'test_'+ str(i+1).zfill(10) + '.png'
    plt.savefig(fname)
    #plt.show()
    #plt.pause(0.01)

#初始化
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
            domain[2]+dx:domain[3]:dx].reshape(2,-1).T
    return fp,wp,dp,gp

fp,wp,dp,gp = initial_position(dx)
num_particles = gp.shape[0] + fp.shape[0] + wp.shape[0] + dp.shape[0]
particles = np.zeros(num_particles,dtype=dtype)
particles['rho'] = rho0
particles['position'] = np.vstack((fp,wp,dp,gp))
#particles['mass'] = rho0*dx**2
#打标签
particles['isBd'][fp.shape[0]:fp.shape[0]+wp.shape[0]] = True
particles['isHd'][fp.shape[0]+wp.shape[0]:-gp.shape[0]] = True
particles['isGate'][-gp.shape[0]:] = True
Btag = particles['isBd']
Gtag = particles['isGate']
Htag = particles['isHd']
Ftag = ~Btag & ~Htag & ~Gtag
particles['mass'][~Gtag] = rho0 * (init_domain[1]- init_domain[0]) * (init_domain[3]-init_domain[2]) /fp.shape[0]
particles['mass'][Gtag] = rho0 * 6*dx * (domain[3]-domain[2]) / gp.shape[0]

#生成墙粒子和虚粒子关系
tree = cKDTree(particles['position'][Btag])
dummy_idx = tree.query(particles['position'][Htag], k=1)[1]

#初始速度
particles['velocity'][Ftag] = uin
particles['velocity'][Btag] = uwall
particles['velocity'][Gtag] = uin

idx = find_neighbors_within_distance(particles["position"], 2*H)
idx = List([np.array(neighbors) for neighbors in idx])
particles['velocity'][Htag] = wall_extrapolation(particles,idx,particles['velocity'])[dummy_idx]

isfree = free_surface(particles, idx)
#print(particles['position'][isfree])
'''
shifting(particles, idx, particles['position'])
测试外推
particles['pressure'][Btag] = np.arange(wp.shape[0])
particles['pressure'][Htag] = particles['pressure'][Btag][dummy_idx]
'''

#可视化
color = np.where(Btag, 'red', np.where(Htag, 'green', np.where(Gtag, 'black', \
        np.where(isfree, 'orange', 'blue'))))
#color = np.where(Btag, 'red', np.where(Htag, 'green', np.where(Gtag, 'black', 'blue')))
#color = particles['mass']
plt.scatter(particles['position'][:, 0], particles['position'][:, 1] ,c=color,s=5)
plt.colorbar(cmap='jet')
plt.grid(True)
ax = plt.gca()
ax.set_aspect('equal')
plt.show()
for i in range(100):
    print("i:", i)
    particles = gate_change(particles) 
    
    #更新标签
    Btag = particles['isBd']
    Htag = particles['isHd']
    Gtag = particles['isGate']
    Ftag = ~Btag & ~Htag & ~Gtag
    
    # 更新半步压力和声速
    change_p(particles,idx)
    
    #更新索引
    idx = find_neighbors_within_distance(particles["position"], 2*H)
    idx = List([np.array(neighbors) for neighbors in idx])

    # 更新半步密度和半步质量
    rho_0 = particles['rho'].copy()
    F_rho_0 = continue_equation(particles, idx, rho_0)
    rho_1 = rho_0 + 0.5*dt*F_rho_0

    #更新半步速度
    velocity_0 = particles['velocity'].copy()
    F_velocity_0 = momentum_equation(particles, idx, velocity_0)
    velocity_1 = velocity_0 + 0.5*dt*F_velocity_0
    velocity_1[Htag] = wall_extrapolation(particles,idx,velocity_1)[dummy_idx]
    
    
    '''
    #更新半步位置
    position_0 = particles['position']
    is_free_particles = free_surface(particles, idx)
    #F_position_0 = change_position(particles, idx, position_0)
    F_position_0 = shifting(particles, idx, position_0, is_free_particles)
    position_1 = position_0 + 0.5*dt*F_position_0
    '''
    particles['rho'] = rho_1
    particles['velocity'] = velocity_1
    particles['velocity'][Htag] = wall_extrapolation(particles,idx,velocity_1)[dummy_idx]
    #particles['position'] = position_1
    
    # 更新压力和声速
    change_p(particles,idx)

    # 更新密度和质量
    F_rho_1 = continue_equation(particles, idx, rho_1)
    
    #更新半步速度
    F_velocity_1 = momentum_equation(particles, idx, velocity_1)
    
    #更新半步位置
    #F_position_1 = change_position(particles, idx, position_1)
    F_position_1 = shifting(particles, idx, particles['position'], is_free_particles)

    particles['rho'] = rho_0 + 0.5*dt*(F_rho_1+F_rho_0)
    particles['velocity'] = velocity_0 + 0.5*dt*(F_velocity_0 + F_velocity_1)
    particles['velocity'][Htag] = wall_extrapolation(particles,idx,particles['velocity'])[dummy_idx]
    #particles['position'] = position_0 + 0.5*dt*(F_position_0 + F_position_1)
    particles['position'] = position_0 + F_position_1
    draw(particles, i)
