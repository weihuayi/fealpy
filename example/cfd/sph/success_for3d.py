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

domain = [0,1.6,0,0.67,0,0.4]
dx = 0.0225
dy = 0.0225
dz = 0.0225
rho0 = 1000
H = 0.866025*np.sqrt(dx**2+dy**2+dz**2)
dt = 0.00005



c0 = 10
h_swl = 0.3
g = np.array([0.0, 0.0, -9.8])
gamma = 7
alpha = 0.1
maxstep = 2000
dtype = [("position", "float64", (3, )), 
         ("velocity", "float64", (3, )),
         ("rho", "float64"),
         ("mass", "float64"),
         ("pressure", "float64"),
         ("sound", "float64"),
         ("isBd", "bool")]

def initial_position(dx, dy, dz): 
    pp = np.mgrid[dx:0.4+dx:dx, dy:0.6475+dy:dy, dz:0.36+dz:dz].reshape(3, -1).T
    
    #前
    fp0 = np.mgrid[0:1.6+dx:dx, 0.67:0.67+dy:dy, 0:0.4+dz:dz].reshape(3, -1).T
    fp1 = np.mgrid[dx/2:1.6+dx/2:dx, 0.67+dy/2:0.67+dy/2+dy:dy, dz/2:0.4+dz/2:dz].reshape(3, -1).T
    fp = np.vstack((fp0,fp1))
    
    #后
    bp0 = np.mgrid[0:1.6+dx:dx, 0:0+dy:dy, 0:0.4+dz:dz].reshape(3, -1).T
    bp1 = np.mgrid[dx/2:1.6+dx/2:dx, 0-dy/2:0-dy/2+dy:dy, dz/2:0.4+dz/2:dz].reshape(3, -1).T
    bp = np.vstack((bp0,bp1))
    
    #左
    lp0 = np.mgrid[0:0+dx:dx, dy:0.67:dy, dz:0.4+dz:dz].reshape(3, -1).T
    lp1 = np.mgrid[-dx/2:dx/2:dx, dy+dy/2:0.67-dy/2:dy, dz/2:0.4-dz/2:dz].reshape(3, -1).T
    lp = np.vstack((lp0,lp1))
    #右
    rp0 = np.mgrid[1.6:1.6+dx:dx, dy:0.67:dy, dz:0.4+dz:dz].reshape(3, -1).T
    rp1 = np.mgrid[1.6+dx/2:1.6+3*dx/2:dx, dy+dy/2:0.67-dy/2:dy, dz/2:0.4+dz/2:dz].reshape(3, -1).T
    rp = np.vstack((rp0,rp1))
    
    #下
    bop0 = np.mgrid[dx:1.6:dx, dy:0.67:dy, 0:0+dz:dz].reshape(3, -1).T
    bop1 = np.mgrid[dx+dx/2:1.6-dx/2:dx, dy+dy/2:0.67-dy/2:dy, 0-dz/2:0+dz/2:dz].reshape(3, -1).T
    bop = np.vstack((bop0,bop1))
    
    boundaryp = np.vstack((fp,bp,lp,rp,bop))

    return pp, boundaryp

# 初始化
pp,bpp = initial_position(dx, dy, dz)
num_particles = pp.shape[0] + bpp.shape[0]
particles = np.zeros(num_particles, dtype=dtype)
particles["rho"] = rho0
particles["mass"] = 0.4*0.6475*0.36*rho0/pp.shape[0]
particles['position'] = np.vstack((pp,bpp))
particles['isBd'][pp.shape[0]:] = True
# Visualization
color = np.where(particles['isBd'], 'red', 'blue')
ax = plt.axes(projection='3d')
ax.scatter(particles['position'][:, 0], particles['position'][:, 1] ,particles['position'][:, 2],c=color ,s=5)
plt.grid(True)
ax.set_aspect('equal')
plt.show()

def find_neighbors_within_distance(points, h):
    tree = cKDTree(points)
    neighbors = tree.query_ball_tree(tree, h)
    return neighbors


@numba.jit(nopython=True)
def kernel(r):
    d = np.linalg.norm(r)
    q = d/H
    if 0 <= q < 1:
        return (1 - 1.5*q**2 + 0.75*q**3)/(np.pi*H**3)
    elif 1 <= q < 2:
        return 0.25*(2-q)**3/(np.pi*H**3)
    else:
        return 0

@numba.jit(nopython=True)
def gradkernel(r):
    d = np.linalg.norm(r)
    q = d/H
    result = np.zeros(3)
    if 0 < q < 1:
        result = r*(1 - 3*q + 2.25*q**2)/(d*np.pi*H**3)
    elif 1 <= q < 2:
        result = -r*0.75*(2-q)**2/(d*np.pi*H*3)
    return result
        
    

@numba.jit(nopython=True)
def continue_equation(particles, idx):
    num = particles.shape[0]
    position = particles["position"]
    velocity = particles["velocity"] 
    mass = particles["mass"] 
    result = np.zeros_like(particles['rho'])
    for i in range(num):
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
    Ftag = ~particles['isBd']
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
    B = c0**2*rho0/gamma
    particles['pressure'] = B * ((particles['rho']/rho0)**gamma - 1)
    particles['sound'] = (B*gamma/rho0 * (particles['rho']/rho0)**(gamma-1))**0.5 
    '''2010   
    if i%15==0 and i!=0:
        particles['pressure'] = B * ((particles['rho']/rho0)**gamma - 1)
        particles['sound'] = (B*gamma/rho0 * (particles['rho']/rho0)**(gamma-1))**0.5 
    else:
        tag = particles['isBd']
        particles['pressure'][~tag] = B * ((particles['rho'][~tag]/rho0)**gamma - 1)
        particles['sound'] = (B*gamma/rho0 * (particles['rho']/rho0)**(gamma-1))**0.5 
    '''

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
    Ftag = ~particles['isBd']
    for i in np.where(Ftag)[0]:
    #for i in range(num):
        for j in idx[i]:
            if i != j:  # 排除自己
                val = pressure[i]/rho[i]**2 + pressure[j]/rho[j]**2
                rij = position[i] - position[j]
                gk = gradkernel(rij)
                vij = velocity[i] - velocity[j]
                piij = 0
                if np.dot(rij, vij) < 0:
                    piij = -alpha * (sound[i] + sound[j])/2
                    piij *= H*np.dot(rij,vij)/(np.dot(rij,rij)+0.01*H*H)
                    piij /= (rho[i] + rho[j])/2
                result[i] += mass[j]*(val+piij)*gk
    result = -result + g 
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
            Abar = np.zeros((4,4)) 
            Abar[0,1] = rij[0] 
            Abar[0,2] = rij[1] 
            Abar[0,3] = rij[2]
            Abar[1,2] = rij[0]*rij[1]
            Abar[1,3] = rij[0]*rij[2]
            Abar[2,3] = rij[1]*rij[2]
            Abar += Abar.T
            Abar[0,0] = 1
            Abar[1,1] = rij[0]**2
            Abar[2,2] = rij[1]**2
            Abar[3,3] = rij[2]**2
            A[i] += Abar*wij*vol[j]
    for i in range(num):
        condA = np.linalg.cond(A[i])
        if condA <1e15:
            invA = np.linalg.inv(A[i])
            for j in idx[i]:
                rij = position[i] - position[j]
                wij = kernel(rij, H)
                wmls = invA[0,0] + invA[1,0]*rij[0] + invA[2,0]*rij[1] + invA[3,0]*rij[2]
                result[i] += wij*wmls*mass[j]
        else:
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
                result[i] = sum_numer/sum_denom
    return result

def draw(particles, i):
    plt.clf()
    c = particles['velocity'][:,2]
    #c = particles['rho']
    ax = plt.axes(projection='3d')
    im = ax.scatter(particles['position'][:, 0], particles['position'][:, 1] ,particles['position'][:, 2],c=c ,s=5)
    plt.colorbar(im, cmap='jet')
    plt.title(f"Time Step: {i}")
    ax.set_aspect('equal')
    plt.pause(0.005)
    #plt.show()

for i in range(10000):
    print("i:", i)
    #print(np.sum(np.abs(particles['position'])))
    idx = find_neighbors_within_distance(particles["position"], 2*H)
    idx = List([np.array(neighbors) for neighbors in idx])
    
    # 更新半步压力和声速
    state_equation(particles,i)

    # 更新半步密度和半步质量
    rho_0 = particles['rho']
    F_rho_0 = continue_equation(particles, idx)
    rho_1 = rho_0 + 0.5*dt*F_rho_0

    #更新半步速度
    velocity_0 = particles['velocity']
    F_velocity_0 = momentum_equation(particles, idx)
    velocity_1 = velocity_0 + 0.5*dt*F_velocity_0
    

    #更新半步位置
    position_0 = particles['position']
    F_position_0 = change_position(particles, idx)
    position_1 = position_0 + 0.5*dt*F_position_0

    particles['rho'] = rho_1
    particles['velocity'] = velocity_1
    particles['position'] = position_1
    
    
    # 更新压力和声速
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

    particles['rho'] = 2*rho_1 - rho0 
    particles['velocity'] = 2*velocity_1 - velocity_0
    particles['position'] = 2*position_1 - position_0
    
    print("velocity", np.sum(particles['velocity']))
    print("position", np.sum(particles['position']))

    draw(particles, i)

    



'''
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
'''
