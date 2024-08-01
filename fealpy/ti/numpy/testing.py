import taichi as ti

@ti.kernel
def assert_array_equal(x: ti.template(), y: ti.template(), error_flag: ti.template()):
    if x.shape != y.shape:
        print(f"Assertion failed: Shape mismatch: {x.shape} != {y.shape}.")
        ti.atomic_add(error_flag[None], 1)  # 增加错误计数

    for I in ti.grouped(x):
        if x[I] != y[I]:
            ti.atomic_add(error_flag[None], 1)  # 增加错误计数
            print(f"Assertion failed: Arrays are not equal at {I}: {x[I]} != {y[I]}.")

@ti.kernel
def assert_array_almost_equal(
        x: ti.template(),
        y: ti.template(),
        error_flag: ti.template(),
        tol: float = 1e-12):
    if x.shape != y.shape:
        ti.atomic_add(error_flag[None], 1)  # 增加错误计数
        print(f"Assertion failed: Shape mismatch: {x.shape} != {y.shape}.")

    for I in ti.grouped(x):
        # 使用绝对差的方式检查两个元素是否接近
        if ti.abs(x[I] - y[I]) > tol:
            ti.atomic_add(error_flag[None], 1)  # 增加错误计数
            print(f"Assertion failed: Arrays are not almost equal at {I}"
                f": {x[I]} != {y[I]}, diff={ti.abs(x[I] - y[I])}")
