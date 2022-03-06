import taichi as ti

ti.init()

x = ti.field(ti.i32)
block = ti.root.dense(ti.i, 5)
pixel = block.dynamic(ti.j, 5) # 最大的长度是 5 
pixel.place(x)
l = ti.field(ti.i32)
ti.root.dense(ti.i, 5).place(l)

@ti.kernel
def make_lists():
    for i in range(5):
        for j in range(i):
            ti.append(x.parent(), i, j * j)  # ti.append(pixel, i, j * j)
        l[i] = ti.length(x.parent(), i)  # [0, 1, 2, 3, 4]
