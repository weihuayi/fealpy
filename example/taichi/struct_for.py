
import taichi as ti

ti.init()

x = ti.field(dtype=ti.i32)
block1 = ti.root.pointer(ti.ij, (3, 3))
block2 = block1.pointer(ti.ij, (2, 2))
pixel = block2.bitmasked(ti.ij, (2, 2))
pixel.place(x)


@ti.kernel
def activity_checking(snode: ti.template(), i: ti.i32, j: ti.i32):
    print(ti.is_active(snode, [i, j]))

for i in range(3):
    for j in range(3):
        activity_checking(block1, i, j)

for i in range(6):
    for j in range(6):
        activity_checking(block2, i, j)

for i in range(12):
    for j in range(12):
        activity_checking(pixel, i, j)

@ti.kernel
def activate_snodes()
    ti.activate(block1, [1, 0])
    ti.activate(block2, [3, 1])
    ti.activate(pixel, [7, 3])

activity_checking(block1, [1, 0]) # output: 1
activity_checking(block2, [3, 1]) # output: 1
activity_checking(pixel, [7, 3])  # output: 1
