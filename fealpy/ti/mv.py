import taichi as ti


@ti.func
def lagrange_shape_function(bcs : ti.template(), p : ti.int32):
    print(bcs.shape)
    print(p)

@ti.kernel
def lagrange_cell_stiff_matrix_0(node : ti.template(), cell : ti.template(),
        S : ti.template()):
    for c in range(cell.shape[0]):
        x0 = node[cell[c, 0], 0]
        y0 = node[cell[c, 0], 1]

        x1 = node[cell[c, 1], 0]
        y1 = node[cell[c, 1], 1]

        x2 = node[cell[c, 2], 0]
        y2 = node[cell[c, 2], 1]

        l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) 
        gphi = ti.Matrix([
            [(y1 - y2)/l, (x2 - x1)/l], 
            [(y2 - y0)/l, (x0 - x2)/l],
            [(y0 - y1)/l, (x1 - x0)/l]])
        l *= 0.5
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                S[c, i, j] = l*(gphi[i, 0]*gphi[j, 0] +
                        gphi[i, 1]*gphi[j, 1])


@ti.kernel
def lagrange_cell_stiff_matrix_1(cnode : ti.template(), S : ti.template()):
    for c in range(cnode.shape[0]):
        x0 = cnode[c, 0, 0] 
        y0 = cnode[c, 0, 1]

        x1 = cnode[c, 1, 0] 
        y1 = cnode[c, 1, 1]

        x2 = cnode[c, 2, 0] 
        y2 = cnode[c, 2, 1]


        l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) 
        gphi = ti.Matrix([
            [(y1 - y2)/l, (x2 - x1)/l], 
            [(y2 - y0)/l, (x0 - x2)/l],
            [(y0 - y1)/l, (x1 - x0)/l]])
        l *= 0.5
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                S[c, i, j] = l*(gphi[i, 0]*gphi[j, 0] +
                        gphi[i, 1]*gphi[j, 1])
