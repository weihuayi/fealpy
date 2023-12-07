#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: success.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: 2023年08月20日 星期日 19时23分27秒
	@bref 
	@ref 
'''  

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time
import numba
from numba.typed import List  

dx = 0.025
dy = 0.025
rho0 = 1000
H = 0.92*np.sqrt(dx**2+dy**2)
dt = 0.001
c0 = 10
g = np.array([0.0, -9.8])
gamma = 7
alpha = 0.3
maxstep = 10000

dtype = [("position", "float64", (2, )), 
         ("velocity", "float64", (2, )),
         ("rho", "float64"),
         ("mass", "float64"),
         ("pressure", "float64"),
         ("sound", "float64"),
         ("isBd", "bool"),
         ("isDd", "bool")]

def initial_position(dx, dy): 
    pp = np.mgrid[dx:1+dx:dx, dy:2+dy:dy].reshape(2, -1).T
    
    #wall
    wbp = np.mgrid[0:4+dx:dx, 0:dy:dy].reshape(2, -1).T
    wlp = np.mgrid[0:dx:dx, dy:4+dy:dy].reshape(2, -1).T
    wrp = np.mgrid[4:4+dx/2:dx, dy:4+dy:dy].reshape(2, -1).T
    wp = np.vstack((wbp, wlp, wrp))
     
    #dummy    
    dbp = np.mgrid[-3*dx:4+4*dx:dx, -3*dy:-dy:dy].reshape(2, -1).T
    wlp = np.mgrid[-3*dx:-dx:dx, 0:4+dy:dy].reshape(2, -1).T
    wrp = np.mgrid[4+dx:4+dx*4:dx, 0:4+dy:dy].reshape(2, -1).T
    dp = np.vstack((dbp,wlp, wrp))

    return pp, wp, dp

# 初始化
pp,wp,dp = initial_position(dx, dy)
num_particles = pp.shape[0] + wp.shape[0] + dp.shape[0]
particles = np.zeros(num_particles, dtype=dtype)
particles["rho"] = rho0
particles["mass"] = 2*rho0/pp.shape[0]
particles['position'] = np.vstack((pp,wp,dp))
particles['isBd'][pp.shape[0]:-dp.shape[0]] = True
particles['isDd'][-dp.shape[0]:] = True
Btag = particles['isBd']
Dtag = particles['isDd']
Ftag = ~Btag & ~Dtag

#生成墙粒子和虚粒子关系
tree = cKDTree(particles['position'][Btag])
dummy_idx = tree.query(particles['position'][Dtag], k=1)[1]

# Visualization
color = np.where(Btag, 'red', np.where(Dtag, 'green', 'blue'))
plt.scatter(particles['position'][:, 0], particles['position'][:, 1] ,c=color ,s=5)
plt.grid(True)
ax = plt.gca()
ax.set_aspect('equal')
plt.show()

def find_neighbors_within_distance(points, h):
    tree = cKDTree(points)
    neighbors = tree.query_ball_tree(tree, h)
    return neighbors


@numba.jit(nopython=True)
def kernel(r):
    d = np.sqrt(np.sum(r**2, axis=-1))
    q = d/H
    val = 7 * (1-q/2)**4 * (2*q+1) / (4*np.pi*H**2)
    return val

@numba.jit(nopython=True)
def gradkernel(r):
    d = np.sqrt(np.sum(r**2))
    q = d/H
    val = -35/(4*np.pi*H**3) * q * (1-q/2)**3 
    if d==0:
        val = 0
    else :
        val /= d
    return r*val
    

#计算墙粒子外推速度
@numba.jit(nopython=True)
def wall_extrapolation(particles, idx, velocity):
    position = particles["position"]
    mass = particles["mass"]  
    rho = particles['rho']
    Btag = particles['isBd']
    Dtag = particles['isDd']
    Ftag = ~Btag & ~Dtag
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
                result[i] =  - sum0/sum1
            else:
                result[i] = 0
    return result[Btag]

@numba.jit(nopython=True)
def continue_equation(particles, idx):
    num = particles.shape[0]
    position = particles["position"]
    velocity = particles["velocity"] 
    mass = particles["mass"] 
    result = np.zeros_like(particles['rho'])
    Btag = particles['isBd']
    Dtag = particles['isDd']
    Ftag = ~Btag & ~Dtag
    for i in np.where(Ftag)[0]:
        for j in idx[i]:
            rij = position[i] - position[j]
            gk = gradkernel(rij)
            vij = velocity[i] - velocity[j]
            result[i] += mass[j]*np.dot(gk,vij)
    return result

@numba.jit(nopython=True)
def change_position(particles, idx):
    num = particles.shape[0]
    position = particles["position"]
    velocity = particles["velocity"] 
    rho = particles["rho"] 
    mass = particles["mass"] 
    result = np.zeros_like(position)
    Btag = particles['isBd']
    Dtag = particles['isDd']
    Ftag = ~Btag & ~Dtag
    for i in np.where(Ftag)[0]:
        for j in idx[i]:
            rhoij = (rho[i] + rho[j])/2
            vji = velocity[j] - velocity[i]
            rij = position[i] - position[j]
            ke = kernel(rij)
            result[i] += 0.5*mass[j]*vji*ke/rhoij
        result[i] += velocity[i]
    return result    

def state_equation(particles,i):
    Btag = particles['isBd']
    Dtag = particles['isDd']
    Ftag = ~Btag & ~Dtag
    position = particles['position']
    pressure = particles['pressure']
    mass = particles['mass']
    rho = particles['rho']
    B = c0**2*rho0/gamma
    particles['pressure'][Ftag] = B * ((particles['rho'][Ftag]/rho0)**gamma - 1)
    particles['sound'][Ftag] = (B*gamma/rho0 * (particles['rho'][Ftag]/rho0)**(gamma-1))**0.5 
    # 计算固壁粒子的压强
    for i in np.where(Btag)[0]:
        sum0 = 0
        sum1 = 0
        for j in idx[i]:
            if Ftag[j]:
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
    particles['pressure'][Dtag] = particles['pressure'][Btag][dummy_idx] 
@numba.jit(nopython=True)
def momentum_equation(particles, idx):
    num = particles.shape[0]
    rho = particles['rho']
    mass = particles["mass"] 
    position = particles["position"]
    velocity = particles["velocity"]
    sound = particles["sound"]
    pressure = particles["pressure"]
    result = np.zeros_like(velocity)
    Btag = particles['isBd']
    Dtag = particles['isDd']
    Ftag = ~Btag & ~Dtag
    for i in np.where(Ftag)[0]:
        for j in idx[i]:
            val = pressure[i]/rho[i]**2 + pressure[j]/rho[j]**2
            rij = position[i] - position[j]
            gk = gradkernel(rij)
            vij = velocity[i] - velocity[j]
            piij = 0
            if np.dot(rij, vij) < 0:
                piij = -alpha * (sound[i] + sound[j])/2
                piij *= H*np.dot(rij,vij)/(np.dot(rij,rij)+0.01*H*H)
                piij /= (rho[i] + rho[j])/2
            result[i] += -mass[j]*(val+piij)*gk
        result[i] += g 
    return result


@numba.jit(nopython=True)
def rein_rho(particles, idx):
    num = particles.shape[0]
    rho = particles['rho']
    mass = particles["mass"] 
    position = particles["position"]
    vol = mass/rho
    A = np.zeros((num,3,3))
    result = np.zeros_like(rho)
    for i in range(num):
        for j in idx[i]:
            rij = position[i] - position[j]
            wij = kernel(rij)
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
        condA = np.linalg.cond(A[i])
        if condA <1e15:
            invA = np.linalg.inv(A[i])
            for j in idx[i]:
                rij = position[i] - position[j]
                wij = kernel(rij)
                wmls = invA[0,0] + invA[1,0]*rij[0] + invA[2,0]*rij[1]
                result[i] += wij*wmls*mass[j]
        else:
            #当逆矩阵不存在时
            sum_numer = 0
            sum_denom = 0
            for j in idx[i]:
                rij = position[i] - position[j]
                wij = kernel(rij)
                mb = mass[j]
                rho_b = particles['rho'][j]
                sum_numer += mb*wij
                sum_denom += (mb*wij)/rho_b
            if sum_denom != 0:
                result[i] = sum_numer/sum_denom
            else:
                result[i] = particles['rho'][i]
    return result

def draw(particles, i):
    plt.clf()
    tag = np.where(particles['isBd'])
    c = particles['velocity'][:,0]
    #c = particles['pressure']
    #c = particles['rho']
    #c[tag] = 0
    plt.scatter(particles['position'][:, 0], particles['position'][:, 1] ,c=c,cmap='jet' ,s=5)
    plt.colorbar(cmap='jet')
    plt.title(f"Time Step: {i}")
    plt.clim(0,7)
    ax = plt.gca()
    ax.set_aspect('equal')
    #plt.show()
    plt.pause(0.001)

for i in range(maxstep):
    print("i:", i)
    idx = find_neighbors_within_distance(particles["position"], 2*H)
    idx = List([np.array(neighbors) for neighbors in idx])
    
   
    # 更新半步密度和半步质量
    rho_0 = particles['rho'].copy()
    F_rho_0 = continue_equation(particles, idx)
    rho_half = rho_0 + 0.5*dt*F_rho_0

    #更新半步速度
    velocity_0 = particles['velocity'].copy()
    F_velocity_0 = momentum_equation(particles, idx)
    velocity_half = velocity_0 + 0.5*dt*F_velocity_0
    velocity_half[Dtag] = wall_extrapolation(particles,idx,velocity_half)[dummy_idx]
    

    #更新半步位置
    position_0 = particles['position'].copy()
    F_position_0 = change_position(particles, idx)
    position_half = position_0 + 0.5*dt*F_position_0

    particles['rho'] = rho_half
    particles['velocity'] = velocity_half
    particles['position'] = position_half
    # 更新半步压力和声速
    state_equation(particles,i)

     
    # 更新密度
    F_rho_1 = continue_equation(particles, idx)
    rho_1 = rho_0 + 0.5*dt*F_rho_1

    
    #更新速度
    F_velocity_1 = momentum_equation(particles, idx)
    velocity_1 = velocity_0 + 0.5*dt*F_velocity_1
    
    #更新半步位置
    F_position_1 = change_position(particles, idx)
    position_1 = position_0 + 0.5*dt*F_position_1

    particles['rho'] = 2*rho_1 - rho_0 
    particles['velocity'] = 2*velocity_1 - velocity_0
    particles['velocity'][Dtag] = wall_extrapolation(particles,idx,particles['velocity'])[dummy_idx]
    particles['position'] = 2*position_1 - position_0
    # 更新压力和声速
    state_equation(particles,i)
 
    if i%30==0 and i!=0:
        particles['rho'] = rein_rho(particles, idx)
    draw(particles, i)


#color = np.where(particles['isBd'], 'red', 'blue')
tag = np.where(particles['isBd'])
c = particles['velocity'][:,0]
#c = particles['pressure']
#c = particles['rho']
c[tag] = max(c)
plt.scatter(particles['position'][:, 0], particles['position'][:, 1] ,c=c,cmap='jet' ,s=5)
plt.colorbar(cmap='jet')
plt.clim(0, 7) 
plt.grid(True)
plt.show()
