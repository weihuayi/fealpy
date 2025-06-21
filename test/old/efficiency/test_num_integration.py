import numpy as np
import itertools
import time
import pytest


@pytest.mark.parametrize("NC, NQ, ldof, TD",
        [(1000000, 20, 10, 2), (1000000, 20, 10, 3)])
def test_scalar_fun_num_inegration(NC, NQ, ldof, TD):


    # 假设你已经有了积分点的坐标 (NQ, 3)
    integ_points = np.random.rand(NQ, TD+1)

    # 计算积分点处的权重，这里假设已经计算好了权重
    weights = np.random.rand(NQ)


    plan = ["NC", "NQ", "ldof"]
    shape0 = (NC, NQ, ldof)
    shape1 = (1, NQ, 1)
    numbers = [0, 1, 2]
    func_values = np.random.rand(shape0[0], shape0[1], shape0[2])
    for p in itertools.permutations(numbers):
        fval = func_values.transpose(p[0], p[1], p[2]).copy()
        weights_reshaped = weights.reshape(shape1[p[0]], shape1[p[1]], shape1[p[2]])
        axis = p.index(1)
        start = time.time()
        results = np.sum(weights_reshaped * fval, axis=axis)
        end = time.time()
        print(results)
        diff = end - start
        print(f"方案{(plan[p[0]], plan[p[1]], plan[p[2]])} : {(shape0[p[0]], shape0[p[1]], shape0[p[2]])} 用时: {diff:.6f} 秒")

@pytest.mark.parametrize("NC, NQ, ldof, TD, GD",
        [(1000000, 20, 10, 2, 3), (1000000, 20, 10, 3, 3)])
def test_vector_fun_num_inegration(NC, NQ, ldof, TD, GD):


    # 假设你已经有了积分点的坐标 (NQ, 3)
    integ_points = np.random.rand(NQ, TD+1)

    # 计算积分点处的权重，这里假设已经计算好了权重
    weights = np.random.rand(NQ)


    plan = ["NC", "NQ", "ldof", "GD"]
    shape0 = (NC, NQ, ldof, GD)
    shape1 = (1, NQ, 1, 1)
    numbers = [0, 1, 2, 3]
    func_values = np.random.rand(shape0[0], shape0[1], shape0[2], shape0[3])
    for p in itertools.permutations(numbers):
        fval = func_values.transpose(p[0], p[1], p[2], p[3]).copy()
        weights_reshaped = weights.reshape(shape1[p[0]], shape1[p[1]],
                shape1[p[2]], shape1[p[3]])
        axis = p.index(1)
        start = time.time()
        results = np.sum(weights_reshaped * fval, axis=axis)
        end = time.time()
        diff = end - start
        print(f"方案{(plan[p[0]], plan[p[1]], plan[p[2]], plan[p[3]])} : {(shape0[p[0]], shape0[p[1]], shape0[p[2]], shape0[p[3]])} 用时: {diff:.6f} 秒")
        print(results.shape)

if __name__ == '__main__':
    test_vector_fun_num_inegration(1000000, 10, 20, 2, 3)

