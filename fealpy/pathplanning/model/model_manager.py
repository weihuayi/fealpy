from ...model import ModelManager

__all__ = ["PathPlanningManager"]

class PathPlanningModelManager(ModelManager):
    _registry = {
        "travelling_salesman_prob": "fealpy.pathplanning.model.travelling_salesman_prob",
        "route_planning": "fealpy.pathplanning.model.route_planning",
    }