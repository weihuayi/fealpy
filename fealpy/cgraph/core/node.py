from typing import Any
from collections import OrderedDict
from collections.abc import Callable

from . import edge as _E
from ._types import InputSlot, OutputSlot

__all__ = [
    "CNode",
    "Sequential",
]


class CNode():
    _input_slots: OrderedDict[str, InputSlot]
    _output_slots: OrderedDict[str, OutputSlot]

    def __init__(self, target: Callable | None = None, variable=False):
        r"""Initialize a compute node."""
        super().__init__()
        self._input_slots = OrderedDict()
        self._output_slots = OrderedDict()
        self._connection_hooks = OrderedDict()
        self._status_hooks = OrderedDict()
        self._target = target
        self._variable = variable

    def register_input(
            self,
            name: str,
            variable: bool = False,
            parameter: str | None = None,
            **kwargs
    ):
        """Add an input slot to the node."""
        if "_input_slots" not in self.__dict__:
            raise AttributeError(
                "cannot assign inputs before CNode.__init__() call"
            )
        elif not isinstance(name, str):
            raise TypeError(
                f"input name should be a string, but got {name.__class__.__name__}"
            )
        elif "." in name:
            raise KeyError('input name can\'t contain "."')
        elif name == "":
            raise KeyError('input name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._input_slots:
            raise KeyError(f"attribute '{name}' already exists")
        else:
            if parameter is None:
                parameter = name

            self._input_slots[name] = InputSlot(
                has_default="default" in kwargs,
                default=kwargs.get("default", None),
                param_name=parameter,
                variable=variable
            )

    def register_output(self, name: str) -> None:
        """Add an output slot to the node."""
        if "_output_slots" not in self.__dict__:
            raise AttributeError(
                "cannot assign outputs before CNode.__init__() call"
            )
        elif not isinstance(name, str):
            raise TypeError(
                f"output name should be a string, but got {name.__class__.__name__}"
            )
        elif "." in name:
            raise KeyError('output name can\'t contain "."')
        elif name == "":
            raise KeyError('output name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._output_slots:
            raise KeyError(f"attribute '{name}' already exists")
        else:
            self._output_slots[name] = OutputSlot()

    @property
    def input_slots(self):
        return self._input_slots

    @property
    def output_slots(self):
        return self._output_slots

    def run(self, *args, **kwargs):
        if self._target is not None:
            return self._target(*args, **kwargs)

    def __call__(self, **kwargs: _E.AddrHandler | Any):
        if self._variable:
            for name in kwargs.keys():
                if name not in self.input_slots:
                    self.register_input(name, variable=False)
        _E.connect_from_address(self.input_slots, kwargs)
        return _E.AddrHandler(self, None)


class Container(CNode):
    nodes : tuple[CNode, ...]

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, index: int) -> CNode:
        return self.nodes[index]


class Sequential(Container):
    def __init__(self, *args: CNode):
        super().__init__()
        self.nodes = tuple(args)

        for name, inslots in args[0].input_slots.items():
            if inslots.has_default:
                self.register_input(name, default=inslots.default)
            else:
                self.register_input(name)

        for name in args[-1].output_slots:
            self.register_output(name)

    def __repr__(self):
        return " >> ".join([repr(n) for n in self.nodes])

    def run(self, *args, **kwargs) -> Any:
        for node in self.nodes:
            args = node.run(*args, **kwargs)
            if not isinstance(args, tuple):
                args = (args,)
            kwargs = {}
        return args
