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
dx = 0.05
dy = 0.05
rho0 = 1000
H = 0.92*np.sqrt(dx**2+dy**2)
dt = 0.001
c0 = 10
h_swl = 2
g = np.array([0.0, -9.8])
gamma = 7
alpha = 0.1
maxstep = 2000
dtype = [("position", "float64", (2, )), 
         ("velocity", "float64", (2, )),
         ("rho", "float64"),
         ("mass", "float64"),
         ("pressure", "float64"),
         ("sound", "float64"),
         ("isBd", "bool")]

def initial_position(dx, dy): 
    pp = np.mgrid[2*dx:1+dx:dx, 2*dy:2+dy:dy].reshape(2, -1).T
    
    x0 = np.arange(0, 4+dx, dx)
    x1 = np.arange(-dx/2, 4+dx+dx/2, dx) 
    y = np.arange(dy, 4+dy, dx)
    
    bp0 = np.column_stack((x0, np.zeros_like(x0)))
    bp1 = np.column_stack((x1, np.full_like(x1, dx)))
    bp = np.vstack((bp0,bp1))
    
    lp0 = np.column_stack((np.zeros_like(y), y))
    lp1 = np.column_stack((np.full_like(y, dy), y+dy/2))
    lp = np.vstack((lp0,lp1))

    rp0 = np.column_stack((np.full_like(y, 4), y)) 
    rp1 = np.column_stack((np.full_like(y, 4-dy), y+dy/2)) 
    rp = np.vstack((rp0,rp1))
    bpp = np.vstack((bp,lp,rp))
    return pp, bpp

# 初始化
pp,bpp = initial_position(dx, dy)
num_particles = pp.shape[0] + bpp.shape[0]
particles = np.zeros(num_particles, dtype=dtype)
particles["rho"] = rho0
particles['position'] = np.vstack((pp,bpp))
particles['isBd'][pp.shape[0]:] = True
particles['isBd'][:pp.shape[0]] = False
# Visualization
color = np.where(particles['isBd'], 'red', 'blue')
plt.scatter(particles['position'][:, 0], particles['position'][:, 1] ,c=color ,s=5)
plt.grid(True)
plt.show()

def find_neighbors_within_distance(points, h):
    tree = cKDTree(points)
    neighbors = tree.query_ball_tree(tree, h)
    return neighbors


@numba.jit(nopython=True)
def kernel(r, h):
    d = np.sqrt(np.sum(r**2, axis=-1))
    q = d/h
    val = 7 * (1-q/2)**4 * (2*q+1) / (4*np.pi*h**2)
    return val

@numba.jit(nopython=True)
def gradkernel(r, h):
    d = np.sqrt(np.sum(r**2))
    q = d/h
    val = -35/(4*np.pi*h**3) * q * (1-q/2)**3 
    val /= d
    return val
    

@numba.jit(nopython=True)
def change_rho(particles, idx):
    num = particles['rho'].shape[0]
    position = particles["position"]
    velocity = particles["velocity"] 
    mass = dx*dy*particles['rho']
    result = np.zeros(num)
    for i in range(num):
        for j in idx[i]:
            if i != j:  # 排除自己
                rij = position[i] - position[j]
                gk = gradkernel(rij, H) * rij
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
    
    tag = particles['isBd']
    particles['position'][~tag] += dt * result[~tag] 

def change_p(particles,i):
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
def change_v(particles, idx):
    num = particles['rho'].shape[0]
    rho = particles['rho']
    mass = dx*dy*rho  ## 用哪一步密度计算质量
    position = particles["position"]
    velocity = particles["velocity"]
    sound = particles["sound"]
    pressure = particles["pressure"]
    result = np.zeros((num,2)) 
    for i in range(num):
        for j in idx[i]:
            if i != j:  # 排除自己
                val = pressure[i]/rho[i]**2 + pressure[j]/rho[j]**2
                rij = position[i] - position[j]
                gk = gradkernel(rij, H) * rij
                vij = velocity[i] - velocity[j]
                pij = 0
                if np.dot(rij, vij) < 0:
                    pij = -alpha * (sound[i] + sound[j])/2
                    pij *= H*np.dot(rij,vij)/(np.dot(rij,rij)+0.01*H*H)
                    pij /= (rho[i] + rho[j])/2
                result[i] += mass[j]*(val+pij)*gk
    result = -result + g 
    tag = particles['isBd']
    particles['velocity'][~tag] += dt * result[~tag]
    #particles['position'][~tag] += dt * particles['velocity'][~tag] 


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
        for j in idx[i]:
            rij = position[i] - position[j]
            wij = kernel(rij, H)
            wmls = invA[0,0] + invA[1,0]*rij[0] + invA[2,0]*rij[1]
            particles['rho'][i] += wij*wmls*mass[j]

def draw(particles, i):
    plt.clf()
    tag = np.where(particles['isBd'])
    #c = particles['velocity'][:,0]
    c = particles['pressure']
    #c = particles['rho']
    c[tag] = 0
    plt.scatter(particles['position'][:, 0], particles['position'][:, 1] ,c=c,cmap='jet' ,s=5)
    plt.colorbar(cmap='jet')
    plt.title(f"Time Step: {i}")
    plt.pause(0.005)

for i in range(maxstep):
    print("i:", i)
    #print(np.sum(np.abs(particles['position'])))
    idx = find_neighbors_within_distance(particles["position"], 2*H)
    idx = List([np.array(neighbors) for neighbors in idx])
    change_rho(particles, idx)
    change_p(particles,i)
    change_v(particles,idx)
    change_position(particles,idx)
    if i%30==0 and i!=0:
        rein_rho(particles, idx)
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
