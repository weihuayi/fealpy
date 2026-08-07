from .mesh_quality import RadiusRatioQuality
from .radius_ratio_objective import RadiusRatioSumObjective

_shapeopt_exports: list[str] = []

try:  # pragma: no cover - optional legacy export surface
    from .shapeopt import (
        AdjointSolveResult2D,
        BoundaryContract2D,
        DofMap2D,
        DualRegionPipeShapeOptimizer2D,
        FluidAdjointSolveInput2D,
        FluidGeometryGradientInput2D,
        FluidStateSolveInput2D,
        FlowObjectiveEvaluationInput2D,
        FlowObjectiveEvaluationResult2D,
        GeometryObjectiveEvaluationInput2D,
        GeometryObjectiveEvaluationResult2D,
        GeometryGradientResult2D,
        IterationState2D,
        MeshContract2D,
        MeshDeformationResult2D,
        PhysicsHooks2D,
        PropagationCache2D,
        OptimizationSetup2D,
        StateSolveResult2D,
        OptimizationResult2D,
    )
except ImportError:
    pass
else:
    _shapeopt_exports = [
        "AdjointSolveResult2D",
        "BoundaryContract2D",
        "DofMap2D",
        "DualRegionPipeShapeOptimizer2D",
        "FluidAdjointSolveInput2D",
        "FluidGeometryGradientInput2D",
        "FluidStateSolveInput2D",
        "FlowObjectiveEvaluationInput2D",
        "FlowObjectiveEvaluationResult2D",
        "GeometryObjectiveEvaluationInput2D",
        "GeometryObjectiveEvaluationResult2D",
        "GeometryGradientResult2D",
        "IterationState2D",
        "MeshContract2D",
        "MeshDeformationResult2D",
        "PhysicsHooks2D",
        "PropagationCache2D",
        "OptimizationSetup2D",
        "StateSolveResult2D",
        "OptimizationResult2D",
    ]

__all__ = [
    "RadiusRatioQuality",
    "RadiusRatioSumObjective",
    *_shapeopt_exports,
]
