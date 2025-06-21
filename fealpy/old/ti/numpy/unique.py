from typing import Optional, Tuple, TypeVar
import taichi as ti

Field = TypeVar('Field')

def unique(a: Field, return_index: bool = False, return_inverse: bool = False, axis: int = 0) -> Tuple:
    """
    Find the unique elements of an array.
    
    Parameters:
        a (Field): Input field.
        return_index (bool): If True, also return the indices of `a` that result in the unique array.
        return_inverse (bool): If True, also return the indices of the unique array that can be used to reconstruct `a`.
        axis (int): The axis to operate on. Currently only axis=0 is supported.
    
    Returns:
        Tuple: Unique elements, indices, and inverse indices if requested.
    """
    M = a.shape[0]
    N = a.shape[1]

    # Allocate fields to store unique elements, their count, indices and inverse indices
    unique_elements = ti.field(a.dtype, shape=(M, N))
    unique_count = ti.field(ti.i32, shape=())
    inverse_indices = ti.field(ti.i32, shape=(M,))
    index_list = ti.field(ti.i32, shape=(M,))

    @ti.kernel
    def find_unique_and_inverse():
        unique_count[None] = 0
        for i in range(M):
            is_unique = True
            # Check if the current row is already in the list of unique elements
            for j in range(unique_count[None]):
                match = True
                for k in range(N):
                    if a[i, k] != unique_elements[j, k]:
                        match = False
                        break
                if match:
                    is_unique = False
                    inverse_indices[i] = j
                    break
            # If the row is unique, add it to the list of unique elements
            if is_unique:
                for k in range(N):
                    unique_elements[unique_count[None], k] = a[i, k]
                index_list[unique_count[None]] = i
                inverse_indices[i] = unique_count[None]
                unique_count[None] += 1

    # Initialize and execute the kernel to find unique elements and compute inverse indices
    find_unique_and_inverse()
    
    # Create fields for return values with the correct sizes
    result_unique_elements = ti.field(a.dtype, shape=(unique_count[None], N))
    result_index_list = ti.field(ti.i32, shape=(unique_count[None],))
    result_inverse_indices = ti.field(ti.i32, shape=(M,))

    @ti.kernel
    def copy_results():
        for i in range(unique_count[None]):
            for j in range(N):
                result_unique_elements[i, j] = unique_elements[i, j]
            result_index_list[i] = index_list[i]
        for i in range(M):
            result_inverse_indices[i] = inverse_indices[i]

    # Copy the results to the new fields
    copy_results()
    
    if return_index and return_inverse:
        return result_unique_elements, result_index_list, result_inverse_indices
    elif return_index:
        return result_unique_elements, result_index_list
    elif return_inverse:
        return result_unique_elements, result_inverse_indices
    else:
        return result_unique_elements
