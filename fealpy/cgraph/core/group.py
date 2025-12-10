
__all__ = ["NodeGroup", "LoopGroup"]

from . import edge as _E
from ._types import NodeExecutionError
from .node import CNode
from .graph import Graph


class NodeGroup(CNode):
    def __init__(self, graph: Graph):
        self._initialized = False
        super(NodeGroup, self).__init__(var_in=False, var_out=False)
        self.graph = graph

        for name in self.graph._source_slots:
            self.register_input(name)

        for name in self.graph._drain_slots:
            self.register_output(name)

        self._initialized = True

    def register_input(
        self,
        name: str,
        variable: bool = False,
        parameter: str | None = None,
        **kwargs
    ):
        if self._initialized:
            raise TypeError("NodeGroup does not support input registration")
        return super(NodeGroup, self).register_input(
            name, variable, parameter, **kwargs
        )

    def register_output(self, name: str):
        if self._initialized:
            raise TypeError("NodeGroup does not support output registration")
        return super(NodeGroup, self).register_output(name)

    def run(self, **kwargs):
        g = self.graph
        try:
            g.execute(source=kwargs, capture_err=False)
        except Exception as e:
            raise NodeExecutionError(
                f"subgraph {g} execution failed in {self}"
            ) from e
        result = tuple(g.get().values())
        g.reset()
        return result[0] if len(result) == 1 else result

    def __repr__(self):
        return "$node group at " + hex(id(self)) + "$"


class LoopGroup(NodeGroup):
    def __init__(self, graph: Graph):
        super(NodeGroup, self).__init__(var_in=False, var_out=False)
        self.graph = graph
        self.validate_graph(graph)
        self.register_output("iters")

        for name in self.graph._source_slots:
            if name == "step":
                continue
            self.register_input(name)

        for name in self.graph._drain_slots:
            if name == "stop":
                continue
            self.register_output(name)

    @staticmethod
    def validate_graph(graph: Graph):
        # check
        if "maxit" not in graph._source_slots:
            raise ValueError("source 'maxit' is required for LoopGroup")
        if "step" not in graph._source_slots:
            raise ValueError("source 'step' is required for LoopGroup")
        if "stop" not in graph._drain_slots:
            raise ValueError("drain 'stop' is required for LoopGroup")
        if "iters" in graph._drain_slots:
            raise ValueError("drain 'iters' is RESERVED for LoopGroup")

    def run(self, **kwargs):
        g = self.graph
        maxit = kwargs.get("maxit", 1)

        for i in range(maxit):
            kwargs["step"] = i
            try:
                g.execute(source=kwargs, capture_err=False)
            except Exception as e:
                raise NodeExecutionError(
                    f"subgraph {g} execution failed in {self} (step {i})"
                ) from e
            result = g.get().values()
            result = {name: value for name, value in zip(g._drain_slots, result)}

            if result.get("stop", False):
                break

            for key in kwargs.keys():
                if key in result:
                    kwargs[key] = result[key]
        # `iters` is the first output
        result = (i+1,) + tuple(result.values())

        return result[0] if len(result) == 1 else result

    def __repr__(self):
        return "$loop group at " + hex(id(self)) + "$"
