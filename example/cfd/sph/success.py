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

dx = 0.05
dy = 0.05
rho0 = 1000
H = 0.92*np.sqrt(dx**2+dy**2)
dt = 0.001
c0 = 10
h_swl = 2
g = np.array([0.0, -9.8])
gamma = 7
alpha = 0.3
maxstep = 21
dtype = [("position", "float64", (2, )), 
         ("velocity", "float64", (2, )),
         ("rho", "float64"),
         ("mass", "float64"),
         ("pressure", "float64"),
         ("sound", "float64"),
         ("isBd", "bool")]

def initial_position(dx, dy): 
    pp = np.mgrid[2*dx:1+dx:dx, 2*dy:2+dy:dy].reshape(2, -1).T
    
    bpp_x = np.arange(0, 4+dx, dx)
    bpp_y = np.array([0, dy, 4, 4-dy])
    bpp_x, bpp_y = np.meshgrid(bpp_x, bpp_y)
    bpp_ud = np.vstack((bpp_x.ravel(), bpp_y.ravel())).T
    
    bpp_x = np.array([0, dx, 4, 4-dx])
    bpp_y = np.arange(2*dy, 4-2*dy, dy)
    bpp_x, bpp_y = np.meshgrid(bpp_x, bpp_y)
    bpp_lr = np.vstack((bpp_x.ravel(), bpp_y.ravel())).T
    
    bpp = np.vstack((bpp_lr, bpp_ud))
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

def kernel(r, h):
    d = np.sqrt(np.sum(r**2, axis=-1))
    q = d/h
    val = np.zeros_like(q)
    tag = np.where((q >= 0) & (q <= 2))
    val[tag] = 7 * (1-q[tag]/2)**4 * (2*q[tag]+1) / (4*np.pi*h**2)
    return val

def gradkernel(r, h):
    d = np.sqrt(np.sum(r**2,axis=-1))
    q = d/h
    val = np.zeros_like(q)
    tag = np.where((q > 0) & (q <= 2))
    val[tag] = -35/(4*np.pi*h**3) * q[tag] * (1-q[tag]/2)**3 
    val[tag] /= d[tag]
    return val
    
def change_rho(particles):
    num = particles['rho'].shape[0]
    mass = dx*dy*particles['rho']
    position = particles["position"]
    velocity = particles["velocity"] 
     
    rij = position[:, np.newaxis, :] - position
    gk = gradkernel(rij, H)
    vij = velocity[:, np.newaxis, :] - velocity
    newrho = np.einsum('ijk, ijk, j, ij-> i', vij, rij, mass, gk) 
    particles["rho"] += dt * newrho

def change_p(particles):
    B = c0**2*rho0/gamma
    particles['pressure'] = B * ((particles['rho']/rho0)**gamma - 1)
    particles['sound'] = (B*gamma/rho0 * (particles['rho']/rho0)**(gamma-1))**0.5 


def change_v(particles):
    num = particles['rho'].shape[0]
    rho = particles['rho']
    mass = dx*dy*rho  ## 用哪一步密度计算质量
    position = particles["position"]
    velocity = particles["velocity"]
    sound = particles["sound"]
    pressure = particles["pressure"]
    
    rij = position[:, np.newaxis, :] - position
    gk = gradkernel(rij, H)
    vij = velocity[:, np.newaxis, :] - velocity
    
    PI_ij = np.zeros_like(gk)
    rv = np.einsum('ijk,ijk->ij', rij, vij)
    rij2 = np.einsum('ijk, ijk->ij', rij, rij)
    tag = np.where(rv<0)
    cbar = (sound[:,np.newaxis] + sound)/2
    rhobar = (rho[:,np.newaxis] + rho)/2
    mu = H*rv/(rij2+0.01*H*H)
    PI_ij[tag] = -alpha * cbar[tag] * mu[tag] / rhobar[tag]
    
    val = pressure/rho**2
    val = val[:, np.newaxis] + val + PI_ij 
    newvelocity = -np.einsum('j, ij, ij, ijk -> ik', mass, val, gk, rij) + g 
    tag = particles['isBd']
    particles['velocity'][~tag] += dt * newvelocity[~tag]
    particles['position'][~tag] += dt * particles['velocity'][~tag] 


def rein_rho(particles):
    num = particles['rho'].shape[0]
    rho = particles['rho']
    mass = dx*dy*rho  ## 用哪一步密度计算质量
    vol = mass/rho
    position = particles["position"] 
    rij = position[:, np.newaxis, :] - position
    ke = kernel(rij, H)
    A0 = np.zeros((num,num,3,3))
    A0[:,:,0,1] = rij[:,:,0]
    A0[:,:,0,2] = rij[:,:,1]
    A0[:,:,1,2] = rij[:,:,0] * rij[:,:,1]
    A0[:,:,1,0] = rij[:,:,0]
    A0[:,:,2,0] = rij[:,:,1]
    A0[:,:,2,1] = rij[:,:,0] * rij[:,:,1] 
    A0[:,:,0,0] = 1
    A0[:,:,1,1] = rij[:,:,0] * rij[:,:,0]
    A0[:,:,2,2] = rij[:,:,1] * rij[:,:,1]
    
    A = np.einsum('ij, ijkl, j->ikl', ke, A0, vol)
    invA = np.linalg.inv(A)
    beta =  invA[:,np.newaxis,0,0] + invA[:,np.newaxis,1,0]*rij[:,:,0] + invA[:,np.newaxis,2,0]*rij[:,:,1]
    particles['rho'] = np.einsum('ij,ij,j->i', beta, ke, mass)

for i in range(maxstep):
    print("i:", i)
    #print(np.sum(np.abs(particles['position'])))
    change_rho(particles)
    change_p(particles)
    change_v(particles)
    if i%30==0 and i!=0:
        rein_rho(particles)

tag = np.where(particles['isBd'])
#c = particles['velocity'][:,0]
c = particles['pressure']
#c = particles['rho']
c[tag] = 1
plt.scatter(particles['position'][:, 0], particles['position'][:, 1] ,c=c,cmap='jet' ,s=5)
plt.colorbar(cmap='jet')
plt.grid(True)
plt.show()
