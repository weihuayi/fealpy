#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: cross_solver.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Fri 17 May 2024 04:05:10 PM CST
	@bref 
	@ref 
'''  
import numpy as np
from fealpy.cfd import NSFEMSolver
from fealpy.levelset.ls_fem_solver import LSFEMSolver

class CrossSolver():
    def __init__(self,pde):
        self.pde = pde
        

    def Heavside(self, phi, epsilon):
        pass
        
    
    '''
    计算参数函数rho,c,lambda,eta
    '''
    def function(self, fun):
        pass
    '''
    计算剪切速率
    '''
    def gamma(self,u):
        pass

    '''
    计算eta_l
    '''
    def eta_l(self,T,p,gamma):
        pass
    
    '''
    界面法向散度
    '''
    def kappa(self, phi):
        pass

    '''
    计算ns方程
    '''
    def NS():
        pass
    
    '''
    计算温度方程
    '''
    def Tempeture():
        pass

    
    '''
    计算LS方程
    '''
    def LS():
        pass
    

    


