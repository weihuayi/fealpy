import taichi as ti

ti.init()

x = ti.field(ti.f32)
block = ti.root.pointer(ti.ij, (4,4))
pixel = block.dense(ti.ij, (2,2))
pixel.place(x)

@ti.kernel
def activate():
    x[2,3] = 1.0
    x[2,4] = 2.0

@ti.kernel
def print_active():
    for i, j in block:
        print("Active block", i, j)
    # output: Active block 1 1
    #         Active block 1 2
    for i, j in x:
        print('field x[{}, {}] = {}'.format(i, j, x[i, j]))
    # output: field x[2, 2] = 0.000000
    #         field x[2, 3] = 1.000000
    #         field x[3, 2] = 0.000000
    #         field x[3, 3] = 0.000000
    #         field x[2, 4] = 2.000000
    #         field x[2, 5] = 0.000000
    #         field x[3, 4] = 0.000000
    #         field x[3, 5] = 0.000000

activate()
print_active()
