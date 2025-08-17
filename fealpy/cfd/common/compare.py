#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: compare.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Tue 22 Oct 2024 06:34:41 PM CST
	@bref 
	@ref 
'''  
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
#n_b = np.load('b.npy')
#n_x = np.load('x.npy')
n_A = np.load('A.npy')
n_u0 = np.load('u0.npy')
#n_xx = np.load('xx.npy')
#n_isBdof = np.load('isBdDof.npy')



#o_b = np.load('ob.npy')
#o_x = np.load('ox.npy')
o_A = np.load('oA.npy')
o_u0 = np.load('ou0.npy').T.reshape(-1)
#o_xx = np.load('oxx.npy')
#o_isBdof = np.load('oisBdDof.npy')

#print(np.sum(np.abs(n_b-o_b)))
print(np.sum(np.abs(n_A-o_A)))
print(np.sum(np.abs(n_u0-o_u0)))
#print(np.sum(np.abs(n_x-o_x)))
#print(np.sum(~(n_isBdof==o_isBdof)))
#print(np.sum(np.abs(n_xx-o_xx)))
