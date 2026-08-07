"""Shape optimization specific L-BFGS state helpers.

This module keeps the implementation focused on mesh/state-aware optimization
objects instead of the generic ``x/f/g`` optimizer interface used elsewhere in
FEALPy.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Number
from typing import Any

import numpy as np


def _is_sequence(value: Any) -> bool:
    """Return whether ``value`` should be treated as an ordered sequence."""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _negate_value(value: Any) -> Any:
    """Best-effort negation that preserves the input structure."""
    if isinstance(value, np.ndarray):
        return -np.asarray(value, dtype=float)
    if isinstance(value, Mapping):
        return {key: _negate_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(map(_negate_value, value))
    if isinstance(value, list):
        return list(map(_negate_value, value))
    try:
        return -value
    except Exception:
        return value


def _flatten_value(value: Any) -> tuple[np.ndarray, tuple[Any, ...]]:
    """Flatten an arbitrary gradient-like structure into a 1D vector."""
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=float)
        return array.reshape(-1), ("ndarray", tuple(array.shape))
    if isinstance(value, Mapping):
        try:
            keys = tuple(sorted(value.keys()))
        except Exception:
            keys = tuple(value.keys())
        flattened = [_flatten_value(value[key]) for key in keys]
        parts = [item[0] for item in flattened]
        descriptors = [item[1] for item in flattened]
        if parts:
            return np.concatenate(parts).astype(float, copy=False), ("mapping", keys, tuple(descriptors))
        return np.zeros(0, dtype=float), ("mapping", keys, tuple(descriptors))
    if isinstance(value, tuple):
        flattened = [_flatten_value(item) for item in value]
        parts = [item[0] for item in flattened]
        descriptors = [item[1] for item in flattened]
        if parts:
            return np.concatenate(parts).astype(float, copy=False), ("tuple", len(value), tuple(descriptors))
        return np.zeros(0, dtype=float), ("tuple", 0, tuple(descriptors))
    if isinstance(value, list):
        flattened = [_flatten_value(item) for item in value]
        parts = [item[0] for item in flattened]
        descriptors = [item[1] for item in flattened]
        if parts:
            return np.concatenate(parts).astype(float, copy=False), ("list", len(value), tuple(descriptors))
        return np.zeros(0, dtype=float), ("list", 0, tuple(descriptors))
    if isinstance(value, Number) or np.isscalar(value):
        return np.asarray([float(value)], dtype=float), ("scalar", type(value))

    try:
        array = np.asarray(value, dtype=float)
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise TypeError(f"Unsupported L-BFGS vector value type: {type(value)!r}") from exc

    if array.ndim == 0:
        return array.reshape(1).astype(float, copy=False), ("scalar", type(value))
    return array.reshape(-1).astype(float, copy=False), ("ndarray", tuple(array.shape))


def _restore_value(vector: np.ndarray, descriptor: tuple[Any, ...], offset: int = 0) -> tuple[Any, int]:
    """Restore a flattened vector back to the recorded structure."""
    kind = descriptor[0]
    if kind == "scalar":
        return float(vector[offset]), offset + 1
    if kind == "ndarray":
        shape = tuple(descriptor[1])
        size = int(np.prod(shape, dtype=int))
        array = np.asarray(vector[offset : offset + size], dtype=float).reshape(shape)
        return array.copy(), offset + size
    if kind in {"tuple", "list"}:
        length = int(descriptor[1])
        subdescriptors = descriptor[2]
        values = []
        current_offset = offset
        for index in range(length):
            item, current_offset = _restore_value(vector, subdescriptors[index], current_offset)
            values.append(item)
        if kind == "tuple":
            return tuple(values), current_offset
        return values, current_offset
    if kind == "mapping":
        keys = descriptor[1]
        subdescriptors = descriptor[2]
        values: dict[Any, Any] = {}
        current_offset = offset
        for key, subdescriptor in zip(keys, subdescriptors):
            item, current_offset = _restore_value(vector, subdescriptor, current_offset)
            values[key] = item
        return values, current_offset
    raise TypeError(f"Unsupported descriptor kind: {kind!r}")


@dataclass(slots=True)
class ShapeLBFGSState:
    """L-BFGS history and two-loop recursion for shape optimization."""

    memory_size: int = 10
    initial_inverse_hessian_scale: float = 1.0
    curvature_tolerance: float = 1.0e-12
    _s_history: deque[np.ndarray] = field(default_factory=deque, init=False, repr=False)
    _y_history: deque[np.ndarray] = field(default_factory=deque, init=False, repr=False)
    _rho_history: deque[float] = field(default_factory=deque, init=False, repr=False)
    _vector_size: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.memory_size < 1:
            raise ValueError("memory_size must be positive")
        self._s_history = deque(maxlen=int(self.memory_size))
        self._y_history = deque(maxlen=int(self.memory_size))
        self._rho_history = deque(maxlen=int(self.memory_size))

    @property
    def history_size(self) -> int:
        """Number of stored curvature pairs."""
        return len(self._s_history)

    def reset(self) -> None:
        """Clear the accumulated curvature information."""
        self._s_history.clear()
        self._y_history.clear()
        self._rho_history.clear()
        self._vector_size = None

    def _inverse_hessian_scale(self) -> float:
        """Return the scalar factor used for the initial Hessian approximation."""
        if not self._s_history:
            return float(self.initial_inverse_hessian_scale)
        s_last = self._s_history[-1]
        y_last = self._y_history[-1]
        yy = float(np.dot(y_last, y_last))
        if yy <= self.curvature_tolerance:
            return float(self.initial_inverse_hessian_scale)
        scale = float(np.dot(s_last, y_last) / yy)
        if not np.isfinite(scale) or scale <= 0.0:
            return float(self.initial_inverse_hessian_scale)
        return scale

    def compute_direction(self, gradient: Any) -> Any:
        """Apply the L-BFGS two-loop recursion to the provided gradient."""
        try:
            gradient_vector, descriptor = _flatten_value(gradient)
        except Exception:
            return _negate_value(gradient)

        if gradient_vector.size == 0:
            return _negate_value(gradient)

        if self._vector_size is not None and self._vector_size != gradient_vector.size:
            self.reset()
            return _negate_value(gradient)

        if not self._s_history:
            return _negate_value(gradient)

        if any(s_vec.size != gradient_vector.size or y_vec.size != gradient_vector.size for s_vec, y_vec in zip(self._s_history, self._y_history)):
            self.reset()
            return _negate_value(gradient)

        q = gradient_vector.astype(float, copy=True)
        history_size = len(self._s_history)
        alpha = np.zeros(history_size, dtype=float)
        rho_values = list(self._rho_history)

        for index, (s_vec, y_vec, rho_value) in enumerate(zip(reversed(self._s_history), reversed(self._y_history), reversed(rho_values))):
            reverse_index = history_size - 1 - index
            alpha[reverse_index] = rho_value * float(np.dot(s_vec, q))
            q = q - alpha[reverse_index] * y_vec

        r = self._inverse_hessian_scale() * q
        for index, (s_vec, y_vec, rho_value) in enumerate(zip(self._s_history, self._y_history, rho_values)):
            beta = rho_value * float(np.dot(y_vec, r))
            r = r + (alpha[index] - beta) * s_vec

        direction = -r
        restored, final_offset = _restore_value(direction, descriptor)
        if final_offset != direction.size:  # pragma: no cover - sanity guard
            return _negate_value(gradient)
        return restored

    def update_history(self, step: Any, previous_gradient: Any, current_gradient: Any) -> bool:
        """Store one curvature pair from an accepted step."""
        try:
            step_vector, _ = _flatten_value(step)
            previous_vector, _ = _flatten_value(previous_gradient)
            current_vector, _ = _flatten_value(current_gradient)
        except Exception:
            self.reset()
            return False

        if step_vector.size == 0 or previous_vector.size == 0 or current_vector.size == 0:
            return False

        if step_vector.size != previous_vector.size or step_vector.size != current_vector.size:
            self.reset()
            return False

        if self._vector_size is not None and self._vector_size != step_vector.size:
            self.reset()

        y_vector = current_vector - previous_vector
        curvature = float(np.dot(step_vector, y_vector))
        if not np.isfinite(curvature) or curvature <= self.curvature_tolerance:
            return False

        if not np.all(np.isfinite(step_vector)) or not np.all(np.isfinite(y_vector)):
            return False

        self._vector_size = int(step_vector.size)
        self._s_history.append(np.asarray(step_vector, dtype=float))
        self._y_history.append(np.asarray(y_vector, dtype=float))
        self._rho_history.append(1.0 / curvature)
        return True
