from ...model import ModelManager

__all__ = ["OPTModelManager"]

class OPTModelManager(ModelManager):
    _registry = {
        "single": "fealpy.opt.model.single",
        "multi": "fealpy.opt.model.multi",
        "constrained": "fealpy.opt.model.constrained"
    }