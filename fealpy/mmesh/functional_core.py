"""
Reusable functional core for metric-tensor based mesh adaptivity.

Provide T, TdA, Tdg, lam (and optional jac_functional) so that
MetricTensorAdaptive / MetricTensorAdaptiveX can be configured without
rewriting the geometric discretization code.
"""
from dataclasses import dataclass
from fealpy.backend import backend_manager as bm
from typing import Callable, Optional


@dataclass
class MetricFunctionalCore:
    """
    Abstract bundle of functional ingredients.

    Implement T, TdA, Tdg, lam; optionally override jac_functional.
    If jac_functional is None, caller should fall back to the class' own
    JAC_functional implementation (e.g., the default in MetricTensorAdaptiveX).
    """
    T: Callable
    TdA: Callable
    Tdg: Callable
    lam: Callable
    jac_functional: Optional[Callable] = None

    def attach(self, obj):
        """Attach core methods to a metric object instance (monkey patch)."""
        obj.T = self.T.__get__(obj, obj.__class__)
        obj.TdA = self.TdA.__get__(obj, obj.__class__)
        obj.Tdg = self.Tdg.__get__(obj, obj.__class__)
        obj.lam = self.lam.__get__(obj, obj.__class__)
        if self.jac_functional is not None:
            obj.JAC_functional = self.jac_functional.__get__(obj, obj.__class__)
        return obj


@dataclass
class PowerLawFunctionalCore(MetricFunctionalCore):
    """
    Default core matching existing MetricTensorAdaptiveX behavior.

    Parameters
    ----------
    gamma : float
        Same gamma used in metric classes.
    d : int
        Geometric dimension.
    """
    gamma: float = 1.0
    d: int = 2

    def __post_init__(self):
        # bind methods
        self.T = self._T
        self.TdA = self._TdA
        self.Tdg = self._Tdg
        self.lam = self._lam
        self.jac_functional = None  # use host class default

    @property
    def p(self):
        return self.d * self.gamma / 2.0

    def _T(self, theta, trA, detA):
        p = self.p
        return trA**p - self.d**p * self.gamma / 2.0 * theta**p * bm.log(detA)

    def _TdA(self, trA):
        p = self.p
        return p * (trA ** (p - 1))[..., None, None]

    def _Tdg(self, theta, detA):
        p = self.p
        return - self.d**p * self.gamma / 2.0 * theta**p * detA**(-1)

    def _lam(self, theta):
        p = self.p
        return self.d**p * theta**p * (1.0 - p * bm.log(theta))


def make_powerlaw_core(gamma: float, d: int) -> PowerLawFunctionalCore:
    """Convenience factory for the default core."""
    return PowerLawFunctionalCore(gamma=gamma, d=d)


def attach_core(metric_obj, core: MetricFunctionalCore):
    """Attach a functional core to an existing metric object instance."""
    return core.attach(metric_obj)
