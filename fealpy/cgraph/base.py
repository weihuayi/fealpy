from typing import (
    Tuple, Dict, Any, Callable,
    Generic, TypeVar, overload,
    Optional
)


_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_Self = TypeVar("_Self")


class OutputSlot():
    def __init__(self, name: str, node):
        self.name = name
        self.node = node

    def __hash__(self):
        return id(self)


class InputSlot():
    _name : str
    _source : Optional[OutputSlot]

    @overload
    def __init__(self, name: str): ...
    @overload
    def __init__(self, name: str, *, default: Any): ...
    def __init__(self, name: str, **kwargs):
        self._name = name
        self._empty = object()
        if "default" in kwargs:
            self._default = kwargs["default"]
        else:
            self._default = self._empty
        self._source = None

    @property
    def name(self):
        return self._name

    @property
    def source(self):
        return self._source

    @property
    def has_default(self):
        return self._default is not self._empty

    @property
    def default(self):
        return self._default

    def connect(self, source: OutputSlot) -> None:
        self._source = source

    def disconnect(self) -> None:
        self._source = None


class NodeBase():
    r"""Nodes in computational graph are expected to provide dataflow from
    input slots to output slots.
    """
    @property
    def input_slots(self) -> Dict[str, InputSlot]:
        raise NotImplementedError

    @property
    def output_slots(self) -> Dict[str, OutputSlot]:
        raise NotImplementedError

    @property
    def IN(self): return _ConnPortIn(self.input_slots)
    @property
    def OUT(self): return _ConnPortOut(self.output_slots)

    def execute(self, context: Dict) -> None:
        input_values = {}

        for input_name, input_slot in self.input_slots.items():
            src_slot = input_slot.source
            if src_slot is None:
                if input_slot.has_default:
                    input_values[input_name] = input_slot.default
                else:
                    raise RuntimeError(f"The '{input_name}' input of node "
                                       f"'{self}' is not connected "
                                       "and not having a default value.")
            elif src_slot in context:
                input_values[input_name] = context[src_slot]
            else:
                raise RuntimeError

        results = self.run(**input_values)
        if not isinstance(results, tuple):
            results = (results,)

        if len(results) != len(self.output_slots):
            raise RuntimeError()

        for output_slot, result in zip(self.output_slots.values(), results):
            context[output_slot] = result

    # def __rshift__(self, other):
    #     if isinstance(other, Node): # *Node >> Node -> Node
    #         self.OUT.sent(other.IN)
    #         return other
    #     return self.OUT.sent(other) # *Node >> port -> self.OUT

    # def __lshift__(self, other):
    #     if isinstance(other, Node): # *Node << Node -> Node
    #         self.IN.recv(other.OUT)
    #         return other
    #     self.IN.recv(other) # *Node << port -> None
    #     return None

    # def __rlshift__(self, other):
    #     self.OUT.sent(other) # port << *Node -> *Node
    #     return self

    # def __rrshift__(self, other):
    #     self.IN.recv(other) # port >> *Node -> *Node
    #     return self


class Node(NodeBase):
    _input_slots : Dict[str, InputSlot]
    _output_slots : Dict[str, OutputSlot]

    def __init__(self):
        super().__init__()
        self._input_slots = {}
        self._output_slots = {}

    @overload
    def add_input(self: _Self, name: str) -> _Self: ...
    @overload
    def add_input(self: _Self, name: str, *, default: Any) -> _Self: ...
    def add_input(self, name: str, **kwargs):
        self._input_slots[name] = InputSlot(name, **kwargs)
        return self

    def add_output(self, name: str):
        self._output_slots[name] = OutputSlot(name, self)
        return self

    def set_default(self, **kwargs):
        for name, value in kwargs.items():
            input_slot = self._input_slots[name]
            input_slot._default = value

    @property
    def input_slots(self):
        return self._input_slots

    @property
    def output_slots(self):
        return self._output_slots


class _ConnPort():
    def __init__(self, container: Dict):
        self._container = container
        self._port = []

    def __getattr__(self, name: str):
        self._port.append(name)
        return self

    def __getitem__(self, name: str):
        if isinstance(name, tuple):
            self._port = list(names)
        else:
            names = name.split(",")
            self._port = [n.strip() for n in names]
        return self

    def get_slot(self):
        if len(self._port) == 0:
            return list(self._container.values())
        return [self._container[p] for p in self._port]


class _ConnPortIn(_ConnPort):
    def recv(self, other: "_ConnPortOut") -> None:
        self_slots = self.get_slot()
        other_slots = other.get_slot()

        for self_slot, other_slot in zip(self_slots, other_slots):
            if isinstance(self_slot, InputSlot) and isinstance(other_slot, OutputSlot):
                self_slot.connect(other_slot)
            else:
                raise RuntimeError(f"Input should receives data from an output.")

    __lshift__ = recv


class _ConnPortOut(_ConnPort):
    def sent(self, other: "_ConnPortIn"):
        self_slots = self.get_slot()
        other_slots = other.get_slot()

        for self_slot, other_slot in zip(self_slots, other_slots):
            if isinstance(self_slot, OutputSlot) and isinstance(other_slot, InputSlot):
                other_slot.connect(self_slot)
            else:
                raise RuntimeError(f"Output should sent data to an input.")

        return self

    __rshift__ = sent


class DataSource(Node):
    def __init__(self, value: Any, /):
        super().__init__()
        self._value = value
        self.add_output("out")

    def __repr__(self):
        return f"Source({self._value})"

    def run(self) -> Any:
        return self._value

Const = DataSource


class Identity(Node):
    def __init__(self, var_name: str, node_name: Optional[str] = None, **kwargs):
        super().__init__()
        self.name = node_name if node_name else "Identity"
        self.var_name = var_name
        self.add_input(var_name, **kwargs)
        self.add_output(var_name)

    def __repr__(self):
        return f"{self.name}({self.var_name})"

    def run(self, x: Any) -> Any:
        return x


class FuncNode(Node, Generic[_T1, _T2]):
    def __init__(self, func: Callable[[_T1], _T2]):
        super().__init__()
        import inspect
        sig = inspect.signature(func)

        for name, param in sig.parameters.items():
            if param.kind in (param.POSITIONAL_ONLY, param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if param.default is param.empty:
                self.add_input(name)
            else:
                self.add_input(name, default=param.default)

        self._func = func
        self.add_output("output")

    def __repr__(self):
        return f"FuncNode({self._func.__name__})"

    def run(self, *args, **kwargs) -> _T2:
        return self._func(*args, **kwargs)


class Container(Node):
    nodes : Tuple[Node, ...]

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, index: int) -> Node:
        return self.nodes[index]


class Sequential(Container):
    def __init__(self, *args: Node):
        super().__init__()
        self.nodes = tuple(args)

        for name, inslots in args[0]._input_slots.items():
            if inslots.has_default:
                self.add_input(name, default=inslots.default)
            else:
                self.add_input(name)

        for name in args[-1]._output_slots:
            self.add_output(name)

    def __repr__(self):
        return " -> ".join([repr(n) for n in self.nodes])

    def run(self, *args, **kwargs) -> Any:
        for node in self.nodes:
            args = node.run(*args, **kwargs)
            if not isinstance(args, tuple):
                args = (args,)
            kwargs = {}
        return args

    # def __rshift__(self, other: Node) -> "Sequential":
    #     if isinstance(other, Sequential):
    #         return Sequential(*self.nodes, *other.nodes)
    #     return Sequential(*self.nodes, other)

class OutputNode(Node):
    def __init__(self):
        super().__init__()
        self.add_input("value")
        self.add_output("out")

    def run(self, value):
        self._value = value
        return value

    @property
    def value(self):
        return self._value