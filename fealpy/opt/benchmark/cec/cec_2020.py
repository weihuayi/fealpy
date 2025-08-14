from ....backend import backend_manager as bm
from .cec_base import CECBase


def bounds_2020(func_num, dim):
    """
    Return the lower and upper bounds for the 2020 CEC benchmark functions.

    All functions in the 2020 suite use the same bounds of `[-100, 100]`.

    Parameters:
        func_num (int): The CEC function number.
        dim (int): The problem dimension.

    Returns:
        tuple[Tensor, Tensor]: A tuple `(lb, ub)` where:
            - `lb`: Lower bound tensor.
            - `ub`: Upper bound tensor.
    """
    return -100 * bm.ones(dim), 100 * bm.ones(dim)


class CEC2020(CECBase):
    """
    CEC 2020 benchmark problem interface.

    Inherits from `CECBase` and provides configuration specific to the 2020 CEC
    benchmark suite. This includes:
        - Allowed problem dimensions: `[2, 5, 10, 15, 20, 30, 50, 100]`.
        - Function number range: `1-10`.
        - Uniform bounds for all functions: `[-100, 100]`.

    Parameters:
        func_num (int): The CEC 2020 function number to evaluate (1-10 inclusive).
        dim (int): Problem dimension. Must be one of `[2, 5, 10, 15, 20, 30, 50, 100]`.

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
        Initialize the CEC 2020 benchmark interface.

        This sets the problem year to 2020, enforces the allowed dimensions
        and function number range for the 2020 suite, and applies the uniform
        bounds of `[-100, 100]` for all functions.

        Parameters:
            func_num (int): The CEC 2020 function number to use (1-10 inclusive).
            dim (int): Problem dimension. Must be one of
                       `[2, 5, 10, 15, 20, 30, 50, 100]`.
        """
        super().__init__(func_num, dim, 2020,
                         dim_choices=[2, 5, 10, 15, 20, 30, 50, 100],
                         func_num_range=(1, 10),
                         bounds_func=bounds_2020)
