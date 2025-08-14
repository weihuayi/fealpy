from ....backend import backend_manager as bm
from .cec_base import CECBase


def bounds_2022(func_num, dim):
    """
    Return the lower and upper bounds for the 2022 CEC benchmark functions.

    For most functions, the bounds are `[-100, 100]`.  
    For function number 11, the bounds are expanded to `[-600, 600]`.

    Parameters:
        func_num (int): The CEC function number.
        dim (int): The problem dimension.

    Returns:
        tuple[Tensor, Tensor]: A tuple `(lb, ub)` where:
            - `lb`: Lower bound tensor.
            - `ub`: Upper bound tensor.
    """
    if func_num == 11:
        return -600 * bm.ones(dim), 600 * bm.ones(dim)
    else:
        return -100 * bm.ones(dim), 100 * bm.ones(dim)


class CEC2022(CECBase):
    """
    CEC 2022 benchmark problem interface.

    Inherits from `CECBase` and provides configuration specific to the 2022 CEC
    benchmark suite. This includes:
        - Allowed problem dimensions: `[2, 10, 20]`.
        - Function number range: `1-12`.
        - Special bound case for function 11: `[-600, 600]`.
        - Default bounds for other functions: `[-100, 100]`.

    Parameters:
        func_num (int): The CEC 2022 function number to evaluate (1-12 inclusive).
        dim (int): Problem dimension. Must be one of `[2, 10, 20]`.

    Inherited Attributes:
        func_num (int): Selected benchmark function number.
        dim (int): Problem dimension.
        lb (Tensor): Lower bound tensor for decision variables.
        ub (Tensor): Upper bound tensor for decision variables.
        lib (ctypes.CDLL): Loaded CEC benchmark dynamic library.
        _prefix (str): Function name prefix used for C library calls.

    Inherited Methods:
        evaluate(x: np.ndarray) -> float:
            Evaluate the benchmark function for a single decision vector.

        fun(x0: np.ndarray) -> np.ndarray:
            Apply `evaluate` to each row of a 2D array.

        evaluate_batch(xs: np.ndarray) -> np.ndarray:
            Evaluate the benchmark function for multiple decision vectors.

        __del__():
            Clean up and release library resources.

        __enter__() -> CECBase:
            Enable usage as a context manager.

        __exit__(exc_type, exc_val, exc_tb):
            Ensure cleanup when exiting a context block.
    """

    def __init__(self, func_num: int, dim: int):
        """
        Initialize the CEC 2022 benchmark interface.

        This sets the problem year to 2022, enforces the allowed dimensions
        and function number range for the 2022 suite, and applies the special
        bound case for function 11.

        Parameters:
            func_num (int): The CEC 2022 function number to use (1-12 inclusive).
            dim (int): Problem dimension. Must be one of `[2, 10, 20]`.
        """
        super().__init__(func_num, dim, 2022,
                         dim_choices=[2, 10, 20],
                         func_num_range=(1, 12),
                         bounds_func=bounds_2022)
