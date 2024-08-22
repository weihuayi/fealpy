import dask.array as da
import numpy as np

array1 = da.from_array(np.arange(10), chunks=5)
array2 = da.from_array(np.arange(10, 20), chunks=5)

sum_array = array1 + array2
product_array = array1 * array2

print(f"Sum: {sum_array.compute()}")
print(f"Product: {product_array.compute()}")

