#!/usr/bin/env python3
# 
import numpy as np
import sys
import matplotlib.pyplot as plt

from fealpy.functionspace import FourierSpace 
from fealpy.mesh import StructureQuadMesh

class FourierSpaceTest():
	def __init__(self):
		pass

	def linear_equation_fft_solver_1d_test(self, N):
		"""
		-\Delta u + u = f

		u(x) = sin(x) [0, 2*\pi)
		"""
		def u(p):
			x = p[0]
			return np.sin(x) 

		def f(p):
			x = p[0]
			return 2*np.sin(x) 

		box = np.array([[2*np.pi]]) 
		space = FourierSpace(box, N)
		U = space.linear_equation_fft_solver(f)
		error = space.error(u, U)
		print(error)

	def linear_equation_fft_solver_2d_test(self, N):
		"""
		-\Delta u + u = f

		u(x, y) = sin(x)*sin(y) [0, 2*\pi)^2
		"""
		def u(p):
			x = p[0]
			y = p[1]
			return np.sin(x)*np.sin(y)

		def f(p):
			x = p[0]
			y = p[1]
			return 3*np.sin(x)*np.sin(y) 

		box = np.array([
		[2*np.pi, 0],
		[0, 2*np.pi]]) 
		space = FourierSpace(box, 6)
		U = space.linear_equation_fft_solver(f)
		error = space.error(u, U)
		print(error)

	def linear_equation_fft_solver_3d_test(self, N):
		"""
		-\Delta u + u = f

		u(x, y, z) = sin(x)*sin(y)*sin(z) [0, 2*\pi)^3
		"""
		def u(p):
			x = p[0]
			y = p[1]
			z = p[2]
			return np.sin(x)*np.sin(y)*np.sin(z)

		def f(p):
			x = p[0]
			y = p[1]
			z = p[2]
			return 4*np.sin(x)*np.sin(y)*np.sin(z)

		box = np.array([
		[2*np.pi, 0, 0],
		[0, 2*np.pi, 0],
		[0, 0, 2*np.pi]]) 
		space = FourierSpace(box, 6)
		U = space.linear_equation_fft_solver(f)
		error = space.error(u, U)
		print(error)

	def parabolic_equation_solver_wu_test(self, NS, NT):
		"""
		u_t = \Delta u + w*u

		w = 1
		u(x, 0) = 1

		周期边界条件
		Operator splitting method
		"""
		from fealpy.timeintegratoralg.timeline import UniformTimeLine

		box = np.array([
		[2*np.pi, 0],
		[0, 2*np.pi]])
		space = FourierSpace(box, NS)
		timeline = UniformTimeLine(0, 1, NT)
		NL = timeline.number_of_time_levels()
		q = space.function(dim=NL)
		w = space.function()
		w[:] = 1
		q[0] = 1
		k, k2 = space.reciprocal_lattice(return_square=True)
		dt = timeline.current_time_step_length()
		E0 = np.exp(-dt/2*w)
		E1 = np.exp(-dt*k2)
		for i in range(1, NL):
			q0 = q[i-1]
			q1 = np.fft.fftn(E0*q0)
			q1 *= E1
			q[i] = np.fft.ifftn(q1).real
			q[i] *= E0

		print(q)

	def parabolic_equation_solver_uu_test(self, NS, NT):
		"""
		u_t = \Delta u + u^2

		w = 1
		u(x, 0) = 1

		周期边界条件
		Semi-implicit method
		The discrete scheme: (time step size is dt)
		\hat{u}^{n+1} = ( \hat{u}^{n} + dt \hat{(u^{n})^{2}} ) / ( I + dt k^2 )
		"""
		from fealpy.timeintegratoralg.timeline import UniformTimeLine

		box = np.array([
		[2*np.pi, 0],
		[0, 2*np.pi]])
		space = FourierSpace(box, NS)
		timeline = UniformTimeLine(0, 1, NT)
		NL = timeline.number_of_time_levels()
		u = space.function(dim=NL)
		u[0] = 1
		k, k2 = space.reciprocal_lattice(return_square=True)
		dt = timeline.current_time_step_length()
		A = 1./(1 + dt*k2)
		for i in range(1, NL):
			u0 = u[i-1]
			u1 = np.fft.fftn(u0)
			uf = np.fft.fftn(u0*u0)
			u2 = A * (u1 + dt*uf)
			u[i] = np.fft.ifftn(u2).real

		print(u)



test = FourierSpaceTest()

if sys.argv[1] == 'linear_1d':
	test.linear_equation_fft_solver_1d_test(6)

if sys.argv[1] == 'linear_2d':
	test.linear_equation_fft_solver_2d_test(6)

if sys.argv[1] == 'linear_3d':
	test.linear_equation_fft_solver_3d_test(6)

if sys.argv[1] == 'parabolic_wu':
	test.parabolic_equation_solver_wu_test(4, 10)

if sys.argv[1] == 'parabolic_uu':
	test.parabolic_equation_solver_uu_test(4, 10)

if sys.argv[1] == 'squad':
	qmesh = StructureQuadMesh(box, N, N)
	mi = qmesh.multi_index()
	fig = plt.figure()
	axes = fig.gca()
	qmesh.add_plot(axes)
	qmesh.find_node(axes, showindex=True)
	plt.show()
