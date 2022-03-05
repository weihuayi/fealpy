
import taichi as ti

ti.init()

x = ti.field(ti.f32)
block = ti.root.pointer(ti.ij, (4,4))
pixel = block.bitmasked(ti.ij, (2,2))
pixel.place(x)

@ti.kernel
def activate():
    x[2,3] = 1.0
    x[2,4] = 2.0

@ti.kernel
def print_active():
    for i, j in block:
        print("Active block", i, j)
    for i, j in x:
        print('field x[{}, {}] = {}'.format(i, j, x[i, j]))

activate()
print_active()
