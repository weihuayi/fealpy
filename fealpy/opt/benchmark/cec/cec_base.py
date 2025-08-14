import os
import sys
import ctypes
import numpy as np
from typing import Tuple


class CECBase:
    """
    A base class for interfacing with CEC benchmark functions for optimization.

    This class dynamically loads the appropriate CEC dynamic library (`.dll` on Windows,
    `.so` on Unix-like systems) for a given competition year, initializes the function, 
    and provides evaluation methods for single or batch inputs.

    It abstracts away the direct C function calls using `ctypes` and ensures that inputs 
    are validated and converted to the correct contiguous `numpy` arrays before passing 
    them to the compiled benchmark function.

    Parameters:
        func_num (int): The CEC function number to evaluate. Must be within `func_num_range`.
        dim (int): The dimensionality of the optimization problem. Must be in `dim_choices`.
        year (int): The CEC benchmark year (e.g., 2017, 2020).
        dim_choices (list[int]): Allowed problem dimensions.
        func_num_range (Tuple[int, int]): Inclusive range of valid function numbers.
        bounds_func (Callable): A callable returning lower and upper bounds for a given
            `(func_num, dim)`.

    Attributes:
        func_num (int): Selected benchmark function number.
        dim (int): Problem dimension.
        lb (float | np.ndarray): Lower bound(s) for the decision variables.
        ub (float | np.ndarray): Upper bound(s) for the decision variables.
        lib (ctypes.CDLL): Loaded CEC benchmark dynamic library.
        _prefix (str): Prefix used for locating function symbols in the library.
        _bounds_func (Callable): Function to retrieve problem bounds.

    Methods:
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

    def __init__(self, func_num: int, dim: int, year: int,
                 dim_choices: list[int], func_num_range: Tuple[int, int],
                 bounds_func):
        """
        Initialize the CEC benchmark interface and load the dynamic library.

        Validates the function number and problem dimension, sets up the corresponding
        C function pointers for initialization, evaluation, and cleanup.

        Parameters:
            func_num (int): The CEC function number to use.
            dim (int): Problem dimension.
            year (int): Year of the CEC benchmark set.
            dim_choices (list[int]): Allowed problem dimensions.
            func_num_range (Tuple[int, int]): Valid range of function numbers (inclusive).
            bounds_func (Callable): Function returning `(lb, ub)` bounds for the problem.

        Raises:
            AssertionError: If `dim` is not in `dim_choices` or `func_num` is out of range.
            OSError: If the corresponding CEC library cannot be loaded.
        """
        assert dim in dim_choices, f"Dimension must be one of {dim_choices}"
        assert func_num_range[0] <= func_num <= func_num_range[1], \
            f"Function number must be {func_num_range[0]}-{func_num_range[1]}"

        self.func_num = func_num
        self.dim = dim
        self._bounds_func = bounds_func

        if sys.platform.startswith("win"):
            lib_name = f"cec{year % 100}_func.dll"
        else:
            lib_name = f"cec{year % 100}_func.so"

        lib_dir = os.path.dirname(__file__)
        lib_path = os.path.join(lib_dir, lib_name)

        os.chdir(lib_dir)
        self.lib = ctypes.CDLL(lib_path)

        prefix = f"cec{year % 100}_"

        # Initialize function pointers
        init_func = getattr(self.lib, f"{prefix}init")
        init_func.argtypes = [ctypes.c_int, ctypes.c_int]
        init_func.restype = None

        evaluate_func = getattr(self.lib, f"{prefix}evaluate")
        evaluate_func.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            ctypes.c_int
        ]
        evaluate_func.restype = ctypes.c_double

        cleanup_func = getattr(self.lib, f"{prefix}cleanup")
        cleanup_func.argtypes = []
        cleanup_func.restype = None

        # Call CEC initialization
        init_func(func_num, dim)

        self.lb, self.ub = self._bounds_func(func_num, dim)
        self._prefix = prefix

    def evaluate(self, x: np.ndarray) -> np.ndarray | float:
        """
        Evaluate the benchmark function for a single decision vector or a batch of vectors.

        If `x` is 1D with shape `(dim,)`, returns a single float.  
        If `x` is 2D with shape `(N, dim)`, returns an array of length N.

        Parameters:
            x (np.ndarray): Decision vector `(dim,)` or batch `(N, dim)`.

        Returns:
            float | np.ndarray: Function value(s) for the given input(s).

        Raises:
            ValueError: If the shape is invalid.
        """
        x = np.ascontiguousarray(x, dtype=np.float64)
        eval_func = getattr(self.lib, f"{self._prefix}evaluate")

        if x.ndim == 1:
            if x.shape != (self.dim,):
                raise ValueError(f"Expected shape ({self.dim},) for 1D input, got {x.shape}")
            return eval_func(x, self.dim)

        elif x.ndim == 2:
            if x.shape[1] != self.dim:
                raise ValueError(f"Expected shape (N, {self.dim}) for 2D input, got {x.shape}")
            results = np.empty(x.shape[0], dtype=np.float64)
            for i in range(x.shape[0]):
                results[i] = eval_func(x[i], self.dim)
            return results

        else:
            raise ValueError("Input must be 1D or 2D array.")

    def __del__(self):
        """Clean up by calling the CEC library's `cleanup` function if loaded."""
        try:
            if hasattr(self, 'lib'):
                getattr(self.lib, f"{self._prefix}cleanup")()
        except Exception:
            pass

    def __enter__(self):
        """
        Enter the context manager for `CECBase`.

        Returns:
            CECBase: The current instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager and ensure cleanup is performed.

        Parameters:
            exc_type (type): Exception type, if any.
            exc_val (Exception): Exception instance, if any.
            exc_tb (traceback): Traceback object, if any.
        """
        self.__del__()
