#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: miaoting.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: 2023年08月21日 星期一 09时25分56秒
	@bref 
	@ref 
'''  
import numpy as np
import matplotlib.pyplot as plt  
  
G = np.array([0, 12000 * -9.8])  
REST_DENS = 1000.  #流体的静态密度
GAS_CONST = 2000.  #K:为与流体相关的常数 
H = 16.  
HSQ = H * H  
MASS = 65.  
VISC = 250.  
DT = 0.0008  
  
M_PI = np.pi  
POLY6 = 315 / (65. * M_PI * pow(H, 9.))  
SPIKY_GRAD = -45 / (M_PI * pow(H, 6.))  
VISC_LAP = 45 / (M_PI * pow(H, 6.))  
  
EPS = H  
BOUND_DAMPING = -0.5  
VIEW_HEIGHT = 800  
VIEW_WIDTH = 1200  
  
  
class Particles:  
    def __init__(self, x: np.ndarray):  
        self.x = x  
        self.v = np.zeros_like(x)  
        self.f = np.zeros_like(x)  
        self.rho = np.zeros(len(x))  
        self.p = np.zeros(len(x))  
  
    def draw_particles(self):  
        particles = self  
        for i,(x, y) in enumerate(particles.x):  
            circ = plt.Circle((x, y), H / 2, color='r', alpha=0.3)  
            ax.add_artist(circ)  
            ax.set_aspect("equal")  
            ax.set_xlim([0, 1200])  
            ax.set_ylim([0, 800])  
  
    @classmethod  
    def initSPH(cls):   #初始化
        y = EPS  
        positions = []  
        while y < VIEW_HEIGHT - EPS * 2.:  
            x = EPS  
            while x <= VIEW_WIDTH:  
                if 2 * np.abs(x - 400) ** 2 - 2 * np.abs(x - 400) * (y - 200) + (y - 200) ** 2 <= 30000:  
                    jitter = np.random.randn()  
                    positions.append([x + jitter, y])  
                x += H  
            y += H  
        print(f"Initializing heartbreak with {len(positions)} particles")  
        return cls(x=np.array(positions))  
  
    def computeDensityPressure(self):   #计算压强
        particles = self  
        for i, particle_i_pos in enumerate(particles.x):  
            particles.rho[i] = 0
            for j, particle_j in enumerate(particles.x):  
                rij = particles.x[j, :] - particles.x[i, :]  
                r2 = np.sum(rij * rij)  
                if r2 < HSQ:  
                    particles.rho[i] += MASS * POLY6 * np.power(HSQ - r2, 3.)  
        particles.p[i] = GAS_CONST * (particles.rho[i] - REST_DENS)  
  
    def computeForces(self):  #计算每个粒子的受力
        particles = self  
        for i, pos_i in enumerate(particles.x):  
            fpress = np.array([0., 0.])  
            fvisc = np.array([0, 0.])  
            for j, pos_j in enumerate(particles.x):  
                if i == j:  
                    continue  
                rij = pos_j - pos_i  
                r = np.linalg.norm(rij)  
            if r < H:  
                fpress += -rij / r * MASS * (particles.p[i] + particles.p[j]) / (2. * particles.rho[j]) * SPIKY_GRAD * pow(H - r, 2.)  #由流体内部的压力差产生的作用力
                fvisc += VISC * MASS * (particles.v[j, :] -particles.v[i, :]) / particles.rho[j] * VISC_LAP * (H - r)  #由粒子之间的速度差引起的作用力
                fgrav = G * particles.rho[i]  #外力
        particles.f[i] = fpress + fvisc + fgrav  
  
    def integrate(self):  #时间步积分
        particles = self  
        for i, pos in enumerate(particles.x):  
            particles.v[i, :] += DT * particles.f[i] / particles.rho[i]  
            particles.x[i, :] += DT * particles.v[i, :]  
  
            if pos[0] - EPS < 0.0:  
                particles.v[i, 0] *= BOUND_DAMPING  
                particles.x[i, 0] = EPS  
            if pos[0] + EPS > VIEW_WIDTH:  
                particles.v[i, 0] *= BOUND_DAMPING  
                particles.x[i, 0] = VIEW_WIDTH - EPS  
            if pos[1] - EPS < 0.0:  
                particles.v[i, 1] *= BOUND_DAMPING  
                particles.v[i, 1] += EPS  
                particles.x[i, 1] = EPS  
            if pos[1] + EPS > VIEW_HEIGHT:  
                particles.v[i, 1] *= BOUND_DAMPING  
                particles.v[i, 1] += EPS  
                particles.x[i, 1] = VIEW_HEIGHT - EPS  
  
    def update(self):  
        self.computeDensityPressure()  
        self.computeForces()  
        self.integrate()  
  
  
particles = Particles.initSPH()  
for i in range(400):  
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))  
    fig.set_tight_layout(True)  
    particles.update()  
    particles.draw_particles()  
    plt.savefig(f'c_damped_{i}.png')  
    plt.show()  
    plt.close()
