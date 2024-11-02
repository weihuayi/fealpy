import taichi as ti

ti.init()

@ti.kernel
def func(v: ti.template()):
    for I in ti.grouped(v):
        v[I] += 1

fb1 = ti.FieldsBuilder()
x = ti.field(dtype=ti.f32)
fb1.dense(ti.ij, (5, 5)).place(x)
fb1_snode_tree = fb1.finalize()  # Finalizes the FieldsBuilder and returns a SNodeTree func(x) ...
fb1_snode_tree.destroy() # Destruction

fb2 = ti.FieldsBuilder()
y = ti.field(dtype=ti.f32)
fb2.dense(ti.i, 5).place(y)
fb2_snode_tree = fb2.finalize()  # Finalizes the FieldsBuilder and returns a SNodeTree func(y) ...
fb2_snode_tree.destroy() # Destruction

ti.init()  # default: packed=False
a = ti.field(ti.i32, shape=(18, 65))  # padded to (32, 128)

ti.init(packed=True)
a = ti.field(ti.i32, shape=(18, 65))  # no padding
