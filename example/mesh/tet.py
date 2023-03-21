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
v21 = x1 - x2
v31 = x1 - x3
v32 = x2 - x3
# Create a coordinate system
N = CoordSys3D('N')

# Convert matrices to vectors in the coordinate system
v10_vec = v10[0] * N.i + v10[1] * N.j + v10[2] * N.k
v20_vec = v20[0] * N.i + v20[1] * N.j + v20[2] * N.k
v30_vec = v30[0] * N.i + v30[1] * N.j + v30[2] * N.k
v21_vec = v21[0] * N.i + v21[1] * N.j + v21[2] * N.k
v31_vec = v31[0] * N.i + v31[1] * N.j + v31[2] * N.k
v32_vec = v32[0] * N.i + v32[1] * N.j + v32[2] * N.k

# Calculate the squared magnitudes of the vectors
v10_sq = v10_vec.magnitude().simplify() ** 2
v20_sq = v20_vec.magnitude().simplify() ** 2
v30_sq = v30_vec.magnitude().simplify() ** 2
v21_sq = v21_vec.magnitude().simplify() ** 2
v31_sq = v31_vec.magnitude().simplify() ** 2
v32_sq = v32_vec.magnitude().simplify() ** 2

# Calculate the cross products and the final result
d0 = (v30_sq * v10_vec.cross(v20_vec) +
      v10_sq * v20_vec.cross(v30_vec) +
      v20_sq * v30_vec.cross(v10_vec)).simplify()

c23 = d0.dot(v20_vec.cross(v30_vec)).simplify()
c31 = d0.dot(v30_vec.cross(v10_vec)).simplify()
c12 = d0.dot(v10_vec.cross(v20_vec)).simplify()


# Calculate the area of face
s0 = Rational(1, 2) * v21_vec.cross(v31_vec).magnitude()
s1 = Rational(1, 2) * v20_vec.cross(v30_vec).magnitude()
s2 = Rational(1, 2) * v30_vec.cross(v10_vec).magnitude()
s3 = Rational(1, 2) * v10_vec.cross(v20_vec).magnitude()
s = s0+s1+s2+s3

# Calculate the values of p0, p1, p2, and p3
p0 = (v31_sq / (4*s2)) + (v21_sq / (4*s3)) + (v32_sq / (4*s1))
p1 = (v32_sq / (4*s0)) + (v20_sq / (4*s3)) + (v30_sq / (4*s2))
p2 = (v30_sq / (4*s1)) + (v10_sq / (4*s3)) + (v31_sq / (4*s0))
p3 = (v10_sq / (4*s2)) + (v20_sq / (4*s1)) + (v21_sq / (4*s0))

# Calculate q10,q20,q30,q21,q31,q32
q10 = -((v31.dot(v30))/(4*s2) + (v21.dot(v20))/(4*s3))
q20 = -((v32.dot(v30))/(4*s1) + (-v21.dot(v10))/(4*s3))
q30 = -((-v32.dot(v20))/(4*s1) + (-v31.dot(v10))/(4*s2))
q21 = -((v32.dot(v31))/(4*s0) + (v20.dot(v10))/(4*s3))
q31 = -((v30.dot(v10))/(4*s2) + (-v32.dot(v21))/(4*s0))
q32 = -((v31.dot(v21))/(4*s0) + (v30.dot(v20))/(4*s1))

print("d0:", d0)
print("d0:", d0.evalf())
print("c23:", c23)
print("c23 (evaluated):", c23.evalf())
print("c23:", c23)
print("c31 (evaluated):", c31.evalf())
print("c23:", c23)
print("c12 (evaluated):", c12.evalf())
print("p0 = ", p0.evalf())
print("p1 = ", p1.evalf())
print("p2 = ", p2.evalf())
print("p3 = ", p3.evalf())
print("q10",q10.evalf())
print("q20",q20.evalf())
print("q30",q30.evalf())
print("q21",q21.evalf())
print("q31",q31.evalf())
print("q32",q32.evalf())
