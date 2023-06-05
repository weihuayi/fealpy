from sympy import * 


h = symbols('h')

# 定义杨氏模量的向量
E = Matrix(symbols('E_x E_y E_z'))

# 定义泊松比的向量
v = Matrix(symbols('v_xy v_yz v_xz'))

# 定义剪切模量的向量
G = Matrix(symbols('G_xy G_yz G_xz'))

# 定义相应的泊松比的向量
vp = Matrix(symbols('v_yx v_zy v_zx'))

# 定义等式
eq1 = Eq(vp[0]/E[1], v[0]/E[0])
eq2 = Eq(vp[2]/E[2], v[2]/E[0])
eq3 = Eq(vp[1]/E[2], v[1]/E[1])

DInv = Matrix.zeros(6, 6)
DInv[0, 0] = 1/E[0]
DInv[1, 1] = 1/E[1]
DInv[2, 2] = 1/E[2]
DInv[0, 1] = -v[0]/E[0]
DInv[1, 0] = -vp[0]/E[1]
DInv[0, 2] = -v[2]/E[0]
DInv[2, 0] = -vp[2]/E[2]
DInv[1, 2] = -v[1]/E[1]
DInv[2, 1] = -vp[1]/E[2]

DInv[3, 3] = 1/G[0]
DInv[4, 4] = 1/G[1]
DInv[5, 5] = 1/G[2]

D = DInv.inv()

D = D.subs({vp[0] : E[1]*v[0]/E[0], vp[2] : E[2]*v[2]/E[0], vp[1] :
    E[2]*v[1]/E[1]})

c = -(-1 + E[2]*v[1]**2/E[1] + E[1]*v[0]**2/E[0] + E[2]*v[2]**2/E[0] + 2*E[2]*v[0]*v[1]*v[2]/E[0])

D = D.subs({ c : h})


print(latex(c))
print(latex(DInv))
print(latex(D))
