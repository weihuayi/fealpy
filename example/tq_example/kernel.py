
import taichi as ti

ti.init(arch=ti.gpu)

@ti.kernel
def hello(i: ti.i32):

