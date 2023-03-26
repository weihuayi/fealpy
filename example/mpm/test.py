#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: test.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: 2023年03月03日 星期五 16时49分00秒
	@bref 
	@ref 
'''  
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation

N = 1000
n_grid = 100
dx = 1/n_grid
dt = 2e-4

p_rho = 1
p_vol = (dx*0.5)**2
p_mass = p_rho*p_vol
gravity = 9.8
bound = 5
E = 400
max_step = 9000

grid_v = np.zeros((n_grid,n_grid,2),np.float64)
grid_m = np.zeros((n_grid,n_grid),np.float64)

class Particle:
    def __init__(self, x, v, rho, mass, vol, C, J):
        self.x = x   # 位置
        self.v = v   # 速度
        self.rho = rho   # 密度
        self.mass = mass # 质量
        self.vol = vol
        self.C = C  #梯度
        self.J = J  #体积比

def init():
    particles = []
    for i in range(N):
        x = np.array([0.4 * random.random() + 0.2, 0.4 * random.random() + 0.2])
        v = np.array([0, -1])
        J = 1
        C = np.zeros((2,2))
        rho = p_rho
        mass = p_mass
        vol = p_vol
        particles.append(Particle(x, v, rho, mass, vol, C, J))
    return particles

def p2g(particles):
    global grid_m
    global grid_v
    for p in particles:
        Xp =  p.x / dx # 求商、求余
        base = (Xp - 0.5).astype(np.int32)
        fx = Xp - base
        w = np.array([0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]) 
        stress = -dt * 4 * E * p.vol * (p.J - 1) / dx**2
        affine = np.array([[stress, 0], [0, stress]]) + p.mass * p.C 
        for i in range(3):
            for j in range(3):
                offset = np.array([i, j])
                dpos = (offset - fx) * dx
                weight = w[i,0] * w[j,1]
                grid_v[tuple(base + offset)] += weight * (p.mass * p.v + affine @ dpos)
                grid_m[tuple(base + offset)] += weight * p.mass 

def set_bound():
    global grid_v
    tag = grid_m > 0
    grid_v[tag,:] /= grid_m[tag,np.newaxis]
    for i in range(n_grid):
        for j in range(n_grid):
            grid_v[i,j,1] -= dt * gravity 
            if i < bound and grid_v[i,j,0] < 0:
                grid_v[i,j,0] = 0
            if i > n_grid - bound and grid_v[i,j,0] > 0:
                grid_v[i,j,0] = 0
            if j < bound and grid_v[i,j,1] < 0:
                grid_v[i,j,1] = 0
            if j > n_grid - bound and grid_v[i,j,1] > 0:
                grid_v[i,j,1] = 0


def g2p(particles):
    global grid_m
    global grid_v
    for p in particles:
        Xp =  p.x / dx
        base = (Xp - 0.5).astype(np.int32)
        fx = Xp - base
        w = np.array([0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]) 
        new_v = np.zeros(2)
        new_C = np.zeros((2,2))
        for i in range(3):
            for j in range(3):
                offset = np.array([i, j])
                dpos = (offset - fx) * dx
                weight = w[i,0] * w[j,1]
                g_v = grid_v[tuple(base + offset)]
                new_v += weight * g_v
                new_C += 4 * weight * np.outer(g_v, dpos) / dx**2
        p.v = new_v
        p.x += dt * p.v
        p.J *= 1 + dt * new_C.trace()
        p.C = new_C

particles= init()

fig, ax = plt.subplots()
sc = ax.scatter([],[])
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('X')
plt.ylabel('Y')

xx = []
yy = []

for i in range(max_step):
    grid_v[:] = 0.0
    grid_m[:] = 0.0
    p2g(particles)
    set_bound()
    g2p(particles)
    xx.append([p.x[0] for p in particles])
    yy.append([p.x[1] for p in particles])

def update(i):
    x = xx[i]
    y = yy[i]
    sc.set_offsets(np.c_[x, y])  # 更新散点图的位置
    return sc,

anim = FuncAnimation(fig, update, frames=max_step, interval=0.1, blit=True)

FFwriter = animation.FFMpegWriter(fps=300, extra_args=['-vcodec', 'libx264'])
anim.save('basic_animation.mp4', writer = FFwriter)
plt.show()
