import dask.array as da
import numpy as np

# 从 Numpy 数组创建 Dask 数组
np_array = np.arange(100).reshape(10, 10)
dask_array = da.from_array(np_array, chunks=(5, 5))

print(dask_array)

mean = dask_array.mean().compute()
std_dev = dask_array.std().compute()

print(f"Mean: {mean}, Standard Deviation: {std_dev}")
