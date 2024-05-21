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
from scipy.sparse import diags,csr_matrix,hstack

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

    def coordinate(self,position):
        i,j = self.mesh.cell_location(position)
        nx = self.mesh.ds.nx
        ny = self.mesh.ds.ny
        result =  i * nx + j 
        return result

<<<<<<< HEAD
    def bilinear(self,position):
        i0,j0 = self.mesh.cell_location(position)
        e = self.e(position)
||||||| 57d1018e
    def bilinear(self,position,e):
        i0,j0 = self.mesh.cell_location(position)
=======
    def NGP(self,position,vertex):
        i0,j0 = self.mesh.cell_location(vertex)
        e = self.e(position)
>>>>>>> upstream/master
        epsilon = e[:,0]
        eta = e[:,1]
        num_p = len(e)
        num_v = (self.mesh.ds.nx + 1)*(self.mesh.ds.ny + 1)
        distance0 = np.zeros((num_p,num_v))
        distance1 = np.zeros((num_p,num_v))
        for i in range(num_p):
            distance0[i] = abs(i0-epsilon[i])
            distance1[i] = abs(j0-eta[i])
        distance = np.stack((distance0, distance1), axis=-1)
        result = np.where(np.all(distance < 0.5, axis=-1), 1, 0)
        result = csr_matrix(result)
        return result

    def bilinear(self,position,vertex):
        i0,j0 = self.mesh.cell_location(vertex)
        e = self.e(position)
        epsilon = e[:,0]
        eta = e[:,1]
        num_p = len(e)
        num_v = (self.mesh.ds.nx + 1)*(self.mesh.ds.ny + 1)
        distance0 = np.zeros((num_p,num_v))
        distance1 = np.zeros((num_p,num_v))
        for i in range(num_p):
            distance0[i] = abs(i0-epsilon[i])
            distance1[i] = abs(j0-eta[i])
        distance = np.stack((distance0, distance1), axis=-1)
        result = np.where(np.all(distance <= 1, axis=-1), (1-distance0)*(1-distance1), 0)
        result = csr_matrix(result)
        return result
    
<<<<<<< HEAD
    def P2G_center(self, particles):
        m_p = particles["mass"]
        e_p = particles["internal_energy"]
||||||| 57d1018e
    def P2G_center(self,particles,cell_center,Vc):
        m_p = particles["mass"]
        e_p = particles["internal_energy"]
=======
    def P2G_cell(self, particles):
        m_p = particles["mass"] #粒子质量
        e_p = particles["internal_energy"] #粒子内能
        Vc = self.mesh.cell_area() #单元面积
>>>>>>> upstream/master
        position = self.particles["position"]
<<<<<<< HEAD
        cell_center = self.mesh.cell_center()
        Vc = self.mesh.cell_area()
        num_p = len(position[:,0])
        num_c = len(cell_center[:,0])
        distance = np.zeros_like(position)
        S_pc = np.zeros((num_p,num_c)) #求插值函数S_pc
||||||| 57d1018e
        num_p = len(position[:,0])
        num_c = len(cell_center[:,0])
        distance = np.zeros_like(position)
        S_pc = np.zeros((num_p,num_c)) #求插值函数S_pc
=======
        index = self.coordinate(particles["position"])
        num_p = len(position)
        num_c = self.mesh.ds.nx * self.mesh.ds.ny
        S_pc = diags([0], [0], shape=(num_p, num_c), format='csr')
>>>>>>> upstream/master
        for i in range(num_p):
            S_pc[i,index[i]] = 1
        rho_c = m_p@S_pc/Vc
<<<<<<< HEAD
        I_c = (e_p@S_pc)/(rho_c*Vc)
        print(rho_c)
        print(I_c) #为什么有些会出现nan?
        return rho_c,I_c
||||||| 57d1018e
        I_c = (e_p@S_pc)/(rho_c*Vc)
        print(rho_c)
        print(I_c) #为什么有些会出现nan?
        return rho_c,I_c
=======
        I_c = (e_p@S_pc)/(rho_c*Vc) #会除以很多0,这里改怎么避免？
        return rho_c, I_c

    def P2G_vertex(self,particles):
        m_p = particles["mass"] #粒子质量
        v_p = particles["velocity"] #粒子速度
        num_v = (self.mesh.ds.nx + 1)*(self.mesh.ds.ny + 1)
        vertex = self.mesh.node.reshape(num_v,2)
        S_pv = self.bilinear(particles["position"],vertex)
        M_v = m_p@S_pv
        #U_v = m_p*v_p/M_v ?
        return M_v
>>>>>>> upstream/master
