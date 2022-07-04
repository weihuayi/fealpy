
#!/usr/bin/env python3
# 
import sys
import argparse
import numpy as np
from fealpy.pde.adi_2d import ADI_2d as PDE
from fealpy.mesh import TriangleMesh
from fealpy.mesh import MeshFactory as mf
from fealpy.decorator import cartesian, barycentric
from numpy.linalg import inv
import matplotlib.pyplot as plt
from fealpy.functionspace import FirstKindNedelecFiniteElementSpace2d
from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.quadrature import  GaussLegendreQuadrature
from scipy.sparse import csr_matrix, coo_matrix
from numpy.linalg import inv
from scipy.sparse import csr_matrix, spdiags, eye, bmat 		
from scipy.sparse.linalg import spsolve
from fealpy.tools.show import showmultirate, show_error_table


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        三角形网格上最低次混合RTN元
        """)

parser.add_argument('--nt',
        default=100, type=int,
        help='时间剖分段数，默认剖分 100 段.')
        
parser.add_argument('--ns',
        default=5, type=int,
        help='空间各个方向初始剖分段数， 默认剖分 10 段.')
        
parser.add_argument('--nmax',
        default=5, type=int,
        help='空间迭代次数， 默认迭代5次.')
					
##
args = parser.parse_args()
ns = args.ns
nt = args.nt
nmax = args.nmax
##初始网格
box = [0, 1, 0, 1]
mesh = mf.boxmesh2d(box, nx=ns, ny=ns, meshtype='tri')  

##
###真解
sigma = 3*np.pi
epsilon = 1.0
mu = 1.0
pde = PDE(sigma, epsilon, mu)


tau = 1.0e-5
"""
errorType = ['$|| E - E_h||_{\Omega,0}$',
		'$|| U - H_h||_{\Omega,0}$'
             ]

errorMatrix = np.zeros((len(errorType), nmax), dtype=np.float64)
"""
errorType = ['$|| E - E_h||_{\Omega,0}$'
             ]

errorMatrix = np.zeros((1, nmax), dtype=np.float64)
NDof = np.zeros(nmax, dtype=np.float64)

for n in range(nmax):
	# 电场初始值
	space = FirstKindNedelecFiniteElementSpace2d(mesh, p=0)
	def init_E_value(p):
		return pde.Efield(p, 0.5*tau)	
	Eh0 = space.interpolation(init_E_value)
	Eh1 = space.function()
	gdof = space.number_of_global_dofs()
	NDof[n] = gdof 

	smspace = ScaledMonomialSpace2d(mesh, p=0)  #分片常数
	# 磁场初始值
	def init_H_value(p):
		return pde.Hz(p, tau)    
	Hh0 = smspace.local_projection(init_H_value)
	Hh1 = smspace.function()
	
	def get_phi_curl_matrix():
		qf = mesh.integrator(q=9, etype='cell')
		bcs, ws = qf.get_quadrature_points_and_weights()
		cellmeasure = mesh.entity_measure('cell')
		ps= mesh.bc_to_point(bcs) #(NQ, NC, GD)
	
		curlpsi = space.curl_basis(bcs) #(NQ, NC, ldof) and ldof=3
		gdof = space.number_of_global_dofs()
		cell2dof = space.cell_to_dof() #(NC, ldof)
		
		
		phi = smspace.basis(ps) #(1,1,1)
		smsgdof = smspace.number_of_global_dofs() #(NC,)
		smscell2dof = smspace.cell_to_dof() #(NC, Lldof) and Lldof=1
		
		M = np.einsum('i, imd, ijk, j->jkd', ws, phi, curlpsi, cellmeasure, optimize=True)
		#print('M.shape=',M.shape)
		
		#(NC,ldof)-->(NC,ldof, 1)=M.shape
		I = cell2dof[:, :, None]
		#(NC,Lldof)-->(NC,Lldof,1)-->(NC, ldof, Lldof)=M.shape 
		J = np.broadcast_to(smscell2dof[:, :, None], shape=M.shape)
		M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, smsgdof))
		
		return M
	
	M_EMat = space.mass_matrix(epsilon)
	M_sigMat = space.mass_matrix(sigma)
	M_SMat = space.curl_matrix(1.0/mu)
	M_HMat = smspace.mass_matrix()
	
	M_CMat = get_phi_curl_matrix()
	TM_CMat = M_CMat.T #转置
	
	LMat = M_EMat + tau/2*M_sigMat + (tau**2/4)*M_SMat
	RMat = M_EMat - tau/2*M_sigMat + (tau**2/4)*M_SMat
	
	for i in range(nt):
		# t1 时间层的计算Eh的右端项
		x = Hh0.T.flat # 把 Hh 按列展平
		RtH1 = tau*M_CMat@x
		
		y = Eh0.T.flat # 把 Eh 按列展平
		RtE1 = RMat@y
		@cartesian
		def sol_g(p):
			return pde.gsource(p, (i+1)*tau) 
		Rt1 = space.source_vector(sol_g)
		Rt1 = tau*Rt1	
		F1 = RtH1 + RtE1 + Rt1
		
		# 下一个时间层的边界条件处理,得到处理完边界之后的总刚和右端项
		# 下一个时间层的电场的计算
		edge2dof = space.dof.edge_to_dof()
		gdof = space.number_of_global_dofs()
		isDDof = np.zeros(gdof, dtype=np.bool_)
		index = mesh.ds.boundary_edge_index()
		isDDof[edge2dof[index]] = True
	
		bdIdx = np.zeros(LMat.shape[0], dtype=np.int_)
		bdIdx[isDDof] = 1
		Tbd = spdiags(bdIdx, 0, LMat.shape[0], LMat.shape[0])
		T = spdiags(1-bdIdx, 0, LMat.shape[0], LMat.shape[0])	
		A1 = T@LMat@T + Tbd
		F1[isDDof] = y[isDDof]
		Eh1[:] = spsolve(A1, F1)
		
		
		# 下一个时间层磁场的计算
		A2 = mu*M_HMat
		@cartesian
		def sol_fz(p):
			return pde.fzsource(p, (i+1.5)*tau)
		
		Rt2 = smspace.source_vector(sol_fz)	
		Rt2 = tau*Rt2
		y1 = Eh1.T.flat #下一个时间层的Eh
		F2 = mu*M_HMat@x - tau*TM_CMat@y1 + Rt2
		Hh1[:] = spsolve(A2, F2)
	
		Eh0 = Eh1 
		Hh0 = Hh1
	
	# 最后一个时间层nmax的电场的真解E
	@cartesian
	def solutionE(p):
		return pde.Efield(p, (nt + 1.5)*tau)
			
	errorMatrix[0, n] = space.integralalg.error(solutionE, Eh0)	
	# 最后一个时间层itmax的磁场的真解Hz
	@cartesian
	def solutionH(p):
		return pde.Hz(p, (nt + 2)*tau)
		
	#errorMatrix[1, n] = smspace.integralalg.error(solutionH, Hh0)
	
	if n < nmax - 1:
		mesh.uniform_refine()
		
	

print("errorMatrix = ", errorMatrix)

showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)   
plt.show()	
	
	
	
	
	
	
	
	

	
	
	
	
	



	
	
	


