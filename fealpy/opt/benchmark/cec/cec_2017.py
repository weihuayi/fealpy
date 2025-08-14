from ....backend import backend_manager as bm
from .cec_base import CECBase


def bounds_2017(func_num, dim):
    """
    Return the lower and upper bounds for the 2017 CEC benchmark functions.

    Parameters:
        func_num (int): The CEC function number (unused for 2017, as bounds are constant).
        dim (int): The problem dimension.

    Returns:
        tuple[Tensor, Tensor]: A tuple `(lb, ub)` where:
            - `lb`: Lower bound tensor, filled with `-100`.
            - `ub`: Upper bound tensor, filled with `100`.
    """
    return -100 * bm.ones(dim), 100 * bm.ones(dim)


class CEC2017(CECBase):
    """
    CEC 2017 benchmark problem interface.

    Inherits from `CECBase` and provides configuration specific to the 2017 CEC
    benchmark suite. This includes:
        - Allowed problem dimensions.
        - Function number range.
        - Constant lower and upper bounds of `[-100, 100]` for all problems.

    Parameters:
        func_num (int): The CEC 2017 function number to evaluate (1-30 inclusive).
        dim (int): Problem dimension. Must be one of `[2, 10, 30, 50, 100]`.

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
        Initialize the CEC 2017 benchmark interface.

        This sets the problem year to 2017, enforces the allowed dimensions
        and function number range for the 2017 suite, and uses fixed bounds
        of `[-100, 100]` for all problems.

        Parameters:
            func_num (int): The CEC 2017 function number to use (1-30 inclusive).
            dim (int): Problem dimension. Must be one of `[2, 10, 30, 50, 100]`.
        """
        super().__init__(func_num, dim, 2017,
                         dim_choices=[2, 10, 30, 50, 100],
                         func_num_range=(1, 30),
                         bounds_func=bounds_2017)
