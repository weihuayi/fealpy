"""Minimal external representation for the first Abaqus INP vertical slice.

These objects preserve the Abaqus labels and keyword context needed by the INP
codec.  They deliberately do not depend on FEALPy domain objects; converting
them into a mesh is the responsibility of a later adapter layer.
"""

from dataclasses import dataclass, field


KeywordOptions = dict[str, str | None]


@dataclass(slots=True)
class AbaqusNodeBlock:
    """Node records declared by one Abaqus ``*Node`` keyword."""

    part_name: str | None = None
    options: KeywordOptions = field(default_factory=dict)
    labels: list[int] = field(default_factory=list)
    coordinates: list[tuple[float, ...]] = field(default_factory=list)


@dataclass(slots=True)
class AbaqusElementBlock:
    """Element records declared by one Abaqus ``*Element`` keyword."""

    element_type: str
    part_name: str | None = None
    options: KeywordOptions = field(default_factory=dict)
    labels: list[int] = field(default_factory=list)
    connectivity: list[tuple[int, ...]] = field(default_factory=list)


@dataclass(slots=True)
class AbaqusInpModel:
    """External representation produced from the supported INP subset."""

    node_blocks: list[AbaqusNodeBlock] = field(default_factory=list)
    element_blocks: list[AbaqusElementBlock] = field(default_factory=list)
    encountered_keywords: set[str] = field(default_factory=set)
    unmapped_keywords: set[str] = field(default_factory=set)
