"""Codec for the supported subset of Abaqus INP physical records."""

from __future__ import annotations

from math import isfinite
from pathlib import Path

from ._model import AbaqusElementBlock, AbaqusInpModel, AbaqusNodeBlock


_MAPPED_KEYWORDS = {"PART", "END PART", "NODE", "ELEMENT"}


class AbaqusInpError(ValueError):
    """Base error raised while reading the supported Abaqus INP subset."""


class AbaqusInpSyntaxError(AbaqusInpError):
    """An INP record is malformed or appears in an invalid context."""


class UnsupportedAbaqusElementError(AbaqusInpError):
    """An element block uses a type outside the current support coverage."""


def _location(path: Path, line_number: int, message: str) -> str:
    return f"{path}:{line_number}: {message}"


def _parse_keyword(
    path: Path,
    line_number: int,
    line: str,
) -> tuple[str, dict[str, str | None]]:
    fields = [field.strip() for field in line[1:].split(",")]
    keyword = fields[0].upper()
    if not keyword:
        raise AbaqusInpSyntaxError(
            _location(path, line_number, "empty Abaqus keyword")
        )

    options: dict[str, str | None] = {}
    for field in fields[1:]:
        if not field:
            continue
        if "=" in field:
            name, value = field.split("=", 1)
            name = name.strip().upper()
            value = value.strip()
            if not name or not value:
                raise AbaqusInpSyntaxError(
                    _location(path, line_number, f"invalid keyword option {field!r}")
                )
            options[name] = value
        else:
            options[field.upper()] = None
    return keyword, options


def _parse_positive_int(
    path: Path,
    line_number: int,
    value: str,
    description: str,
) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise AbaqusInpSyntaxError(
            _location(path, line_number, f"invalid {description}: {value!r}")
        ) from exc
    if result <= 0:
        raise AbaqusInpSyntaxError(
            _location(path, line_number, f"{description} must be positive, got {result}")
        )
    return result


class _InpParser:
    def __init__(self, path: Path):
        self.path = path
        self.model = AbaqusInpModel()
        self.part_name: str | None = None
        self.current_block: AbaqusNodeBlock | AbaqusElementBlock | None = None
        self.node_labels: dict[str | None, set[int]] = {}
        self.element_labels: dict[str | None, set[int]] = {}

    def parse(self) -> AbaqusInpModel:
        with self.path.open("r", encoding="utf-8") as source:
            for line_number, raw_line in enumerate(source, start=1):
                line = raw_line.strip()
                if not line or line.startswith("**"):
                    continue
                if line.startswith("*"):
                    self._start_keyword(line_number, line)
                elif isinstance(self.current_block, AbaqusNodeBlock):
                    self._parse_node(line_number, line, self.current_block)
                elif isinstance(self.current_block, AbaqusElementBlock):
                    self._parse_element(line_number, line, self.current_block)

        if self.part_name is not None:
            raise AbaqusInpSyntaxError(
                f"{self.path}: unterminated Abaqus part {self.part_name!r}"
            )
        self._validate_node_references()
        return self.model

    def _start_keyword(self, line_number: int, line: str) -> None:
        keyword, options = _parse_keyword(self.path, line_number, line)
        self.model.encountered_keywords.add(keyword)
        self.current_block = None

        if keyword not in _MAPPED_KEYWORDS:
            self.model.unmapped_keywords.add(keyword)
            return
        if keyword == "PART":
            self._start_part(line_number, options)
        elif keyword == "END PART":
            self._end_part(line_number)
        elif keyword == "NODE":
            block = AbaqusNodeBlock(part_name=self.part_name, options=options)
            self.model.node_blocks.append(block)
            self.current_block = block
        elif keyword == "ELEMENT":
            self._start_element_block(line_number, options)

    def _start_part(
        self,
        line_number: int,
        options: dict[str, str | None],
    ) -> None:
        if self.part_name is not None:
            raise AbaqusInpSyntaxError(
                _location(
                    self.path,
                    line_number,
                    f"part {self.part_name!r} is still open",
                )
            )
        name = options.get("NAME")
        if name is None:
            raise AbaqusInpSyntaxError(
                _location(self.path, line_number, "*Part requires a NAME option")
            )
        self.part_name = name

    def _end_part(self, line_number: int) -> None:
        if self.part_name is None:
            raise AbaqusInpSyntaxError(
                _location(self.path, line_number, "*End Part without an open part")
            )
        self.part_name = None

    def _start_element_block(
        self,
        line_number: int,
        options: dict[str, str | None],
    ) -> None:
        element_type = options.get("TYPE")
        if element_type is None:
            raise AbaqusInpSyntaxError(
                _location(self.path, line_number, "*Element requires a TYPE option")
            )
        element_type = element_type.upper()
        if element_type != "C3D4":
            raise UnsupportedAbaqusElementError(
                _location(
                    self.path,
                    line_number,
                    f"unsupported Abaqus element type {element_type!r}; "
                    "current support is limited to C3D4",
                )
            )

        block = AbaqusElementBlock(
            element_type=element_type,
            part_name=self.part_name,
            options=options,
        )
        self.model.element_blocks.append(block)
        self.current_block = block

    def _parse_node(
        self,
        line_number: int,
        line: str,
        block: AbaqusNodeBlock,
    ) -> None:
        fields = [field.strip() for field in line.split(",")]
        if not 2 <= len(fields) <= 4 or any(not field for field in fields):
            raise AbaqusInpSyntaxError(
                _location(
                    self.path,
                    line_number,
                    "a *Node record requires one label and one to three coordinates",
                )
            )

        label = _parse_positive_int(self.path, line_number, fields[0], "node label")
        labels = self.node_labels.setdefault(block.part_name, set())
        if label in labels:
            raise AbaqusInpSyntaxError(
                _location(
                    self.path,
                    line_number,
                    f"duplicate node label {label} in part {block.part_name!r}",
                )
            )

        try:
            coordinates = tuple(float(value) for value in fields[1:])
        except ValueError as exc:
            raise AbaqusInpSyntaxError(
                _location(self.path, line_number, f"invalid node coordinates in {line!r}")
            ) from exc
        if not all(isfinite(value) for value in coordinates):
            raise AbaqusInpSyntaxError(
                _location(self.path, line_number, "node coordinates must be finite")
            )

        labels.add(label)
        block.labels.append(label)
        block.coordinates.append(coordinates)

    def _parse_element(
        self,
        line_number: int,
        line: str,
        block: AbaqusElementBlock,
    ) -> None:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5 or any(not field for field in fields):
            raise AbaqusInpSyntaxError(
                _location(
                    self.path,
                    line_number,
                    "a C3D4 record requires one element label and four node labels",
                )
            )

        label = _parse_positive_int(self.path, line_number, fields[0], "element label")
        labels = self.element_labels.setdefault(block.part_name, set())
        if label in labels:
            raise AbaqusInpSyntaxError(
                _location(
                    self.path,
                    line_number,
                    f"duplicate element label {label} in part {block.part_name!r}",
                )
            )
        connectivity = tuple(
            _parse_positive_int(self.path, line_number, value, "node label")
            for value in fields[1:]
        )

        labels.add(label)
        block.labels.append(label)
        block.connectivity.append(connectivity)

    def _validate_node_references(self) -> None:
        for block in self.model.element_blocks:
            known_labels = self.node_labels.get(block.part_name, set())
            for element_label, connectivity in zip(block.labels, block.connectivity):
                missing = sorted(set(connectivity) - known_labels)
                if missing:
                    raise AbaqusInpError(
                        f"{self.path}: element {element_label} in part "
                        f"{block.part_name!r} references undefined node labels {missing}"
                    )


def read_inp_model(filename: str | Path) -> AbaqusInpModel:
    """Read the currently supported Abaqus INP subset.

    The file is consumed line by line.  Node and element records are preserved
    in an Abaqus-specific external representation; no FEALPy mesh is created by
    this codec.
    """

    return _InpParser(Path(filename)).parse()
