from typing import Any
from collections import OrderedDict

from . import edge as _E
from ._types import NodeTopologyError, InputSlot, OutputSlot

__all__ = [
    "Node",
    "DataSource",
    "Const",
    "Sequential"
]


class Node():
    _input_slots : OrderedDict[str, InputSlot]
    _output_slots : OrderedDict[str, OutputSlot]

    def __init__(
        self,
        target=None,
        inputs: tuple[str, ...] = (),
        defaults: dict[str, Any] = {},
        outputs: tuple[str, ...] = (),
        variable: bool = False
    ):
        r"""Initialize a compute node."""
        super().__init__()
        self._target = target
        self._variable = variable
        self._input_slots = OrderedDict()
        self._output_slots = OrderedDict()
        self._connection_hooks = OrderedDict()
        self._status_hooks = OrderedDict()

        if target:
            for in_arg in inputs:
                if in_arg in defaults:
                    self.register_input(in_arg, default=defaults[in_arg])
                else:
                    self.register_input(in_arg)

            for out_arg in outputs:
                self.register_output(out_arg)

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
                "cannot assign inputs before Node.__init__() call"
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
                "cannot assign outputs before Node.__init__() call"
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

    def get_input(self, name: str):
        """Return the input slot given by `name`. Raises NodeTopologyError if not exists."""
        if name not in self._input_slots:
            raise NodeTopologyError(f"no input named {name}")
        return self._input_slots[name]

    def get_output(self, name: str):
        """Return the output slot given by `name`. Raises NodeTopologyError if not exists."""
        if name not in self._output_slots:
            raise NodeTopologyError(f"no output named {name}")
        return self._output_slots[name]

    def dump_key(self, slot: str):
        return (self, slot)

    @property
    def input_slots(self):
        return self._input_slots

    @property
    def output_slots(self):
        return self._output_slots

    @property
    def is_variable(self):
        return self._variable

    def run(self, *args, **kwargs):
        if self._target:
            return self._target(*args, **kwargs)

    def __call__(self, **kwargs: _E.AddrHandler):
        if self._variable:
            for name in kwargs.keys():
                if name not in self.input_slots:
                    self.register_input(name, variable=False)
        _E.connect_from_address(self.input_slots, kwargs)
        return _E.AddrHandler(self, None)


class DataSource(Node):
    def __init__(self, value: Any):
        super().__init__()
        self._value = value
        self.register_output("value")

    def __repr__(self):
        return f"Const({self._value})"

    def run(self) -> Any:
        return self._value
    
Const = DataSource


class Container(Node):
    nodes : tuple[Node, ...]

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, index: int) -> Node:
        return self.nodes[index]


class Sequential(Container):
    def __init__(self, *args: Node):
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
