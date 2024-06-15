import taichi as ti

def zeros(shape, dtype=ti.f32):
    """
    Create a Taichi field filled with zeros.

    Parameters:
        shape (tuple): Shape of the field.
        dtype (taichi.DataType): Data type of the field. Default is ti.f32.

    Returns:
        field: Taichi field filled with zeros.
    """
    field = ti.field(dtype, shape=shape)
    field.fill(0)
    return field

def ones(shape, dtype=ti.f32):
    """
    Create a Taichi field filled with ones.

    Parameters:
        shape (tuple): Shape of the field.
        dtype (taichi.DataType): Data type of the field. Default is ti.f32.

    Returns:
        field: Taichi field filled with ones.
    """
    field = ti.field(dtype, shape=shape)
    field.fill(1)
    return field

def arange(start: Union[int, float], 
           stop: Optional[Union[int, float]] = None, 
           step: Union[int, float] = 1, 
           dtype=ti.f32, *, like=None):
    """
    Create a Taichi field with evenly spaced values within a given interval.

    Parameters:
        start (int or float): Start of the interval.
        stop (int or float, optional): End of the interval. If not provided, start is treated as 0 and start is used as stop.
        step (int or float, optional): Spacing between values. Default is 1.
        dtype (taichi.DataType): Data type of the field. Default is ti.f32.
        like: Ignored, for NumPy compatibility.

    Returns:
        field: Taichi field with evenly spaced values.
    """
    if stop is None:
        start, stop = 0, start

    num_elements = int((stop - start) / step)
    field = ti.field(dtype, shape=(num_elements,))

    @ti.kernel
    def fill_arange():
        for i in range(num_elements):
            field[i] = start + i * step

    fill_arange()
    return field
