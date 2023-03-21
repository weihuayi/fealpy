from sympy import Matrix, sqrt, Rational
from sympy.vector import CoordSys3D

# Define the tetrahedron vertices
x0 = Matrix([0, 0, 0])
x1 = Matrix([1, 0, 0])
x2 = Matrix([Rational(1, 2), sqrt(3)/2, 0])
x3 = Matrix([Rational(1, 2), sqrt(3)/6, sqrt(Rational(2, 3))])

# Define the vectors v10, v20, v30
v10 = x0 - x1
v20 = x0 - x2
v30 = x0 - x3

# Create a coordinate system
N = CoordSys3D('N')

# Convert matrices to vectors in the coordinate system
v10_vec = v10[0] * N.i + v10[1] * N.j + v10[2] * N.k
v20_vec = v20[0] * N.i + v20[1] * N.j + v20[2] * N.k
v30_vec = v30[0] * N.i + v30[1] * N.j + v30[2] * N.k

# Calculate the squared magnitudes of the vectors
v10_sq = v10_vec.magnitude().simplify() ** 2
v20_sq = v20_vec.magnitude().simplify() ** 2
v30_sq = v30_vec.magnitude().simplify() ** 2

# Calculate the cross products and the final result
d0 = (v30_sq * v10_vec.cross(v20_vec) +
      v10_sq * v20_vec.cross(v30_vec) +
      v20_sq * v30_vec.cross(v10_vec)).simplify()

c23 = d0.dot(v20_vec.cross(v30_vec)).simplify()
c31 = d0.dot(v30_vec.cross(v10_vec)).simplify()
c12 = d0.dot(v10_vec.cross(v20_vec)).simplify()
print("d0:", d0)
print("d0:", d0.evalf())
print("c23:", c23)
print("c23 (evaluated):", c23.evalf())
print("c23:", c23)
print("c31 (evaluated):", c31.evalf())
print("c23:", c23)
print("c12 (evaluated):", c12.evalf())


