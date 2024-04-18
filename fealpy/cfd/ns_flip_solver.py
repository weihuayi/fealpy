#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: ns_flip_solver.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Sat 13 Apr 2024 04:32:23 PM CST
	@bref 
	@ref 
'''  
import numpy as np

class NSFlipSolver:
    def __init__(self, particles, mesh):
        self.mesh = mesh
        self.particles = particles
    
    def e(self, position):
        i,j = self.mesh.cell_location(position)
        node = self.mesh.node
        cell = self.mesh.entity('cell')
        result = position - node[i,j,:]
        result += np.column_stack((i,j))
        return result


    def NGP(self,position,e):
        i0,j0 = self.mesh.cell_location(position)
        epsilon = e[:,0]
        eta = e[:,1]
        num = len(epsilon)
        distance0 = np.zeros((num,num))
        distance1 = np.zeros((num,num))
        for i in range(num):
            distance0[i] = abs(i0-epsilon[i])
            distance1[i] = abs(j0-eta[i])
        distance = np.stack((distance0, distance1), axis=-1)
        print(distance[0])
        result = np.where(np.all(distance < 0.5, axis=-1), 1, 0)
        print(result[0])
        return result

    def bilinear(self,position,e):
        i0,j0 = self.mesh.cell_location(position)
        epsilon = e[:,0]
        eta = e[:,1]
        num = len(epsilon)
        distance0 = np.zeros((num,num))
        distance1 = np.zeros((num,num))
        for i in range(num):
            distance0[i] = abs(i0-epsilon[i])
            distance1[i] = abs(j0-eta[i])
        distance = np.stack((distance0, distance1), axis=-1)
        print(distance[0])
        result = np.where(np.all(distance <= 1, axis=-1), (1-distance0)@(1-distance1), 0)
        print(result[0])
        return result
    
    def P2G_center(self,particles,cell_center,Vc):
        m_p = particles["mass"]
        e_p = particles["internal_energy"]
        position = self.particles["position"]
        num_p = len(position[:,0])
        num_c = len(cell_center[:,0])
        distance = np.zeros_like(position)
        S_pc = np.zeros((num_p,num_c)) #求插值函数S_pc
        for i in range(num_p):
            distance = abs(position[i] - cell_center)
            distance = distance[:,0] + distance[:,1]
            min = np.min(distance)
            S_pc[i] = np.where(distance == min,1,0)
        rho_c = m_p@S_pc/Vc
        I_c = (e_p@S_pc)/(rho_c*Vc)
        print(rho_c)
        print(I_c) #为什么有些会出现nan?
        return rho_c,I_c