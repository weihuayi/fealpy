from fealpy.decorator import cartesian

@cartesian
def AA(p):
    x = p[..., 0]
    y = p[..., 1]
    val = np.zeros(x.shape+(2, 2), dtype=np.float_)
    val[..., 0, 0] = 
    val[..., 0, 1] = 
    val[..., 1, 0] = 
    val[..., 1, 1] = 
    return val

S = space.stiff_matrix(c=AA)
