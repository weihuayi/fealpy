
from collections.abc import Iterable

from .entity_schema import EntitySchema
from .classic import *

__all__ = ["SCHEMA_REGISTRY"]


SCHEMA_REGISTRY: dict[str, type[EntitySchema]] = {
	"point": PointSchema,
	"segment": SegmentSchema,
	"tri": TriangleSchema,
	"quad": QuadrilateralSchema,
	"tet": TetrahedronSchema,
	"prism": PrismSchema,
	"pyramid": PyramidSchema,
	"hex": HexahedronSchema,
}


def ensure_positive_topdim(top_dim: int, highest_dim: int) -> int:
    if top_dim < -highest_dim - 1 or top_dim > highest_dim:
        raise ValueError(f"top dimension {top_dim} is out of range "
                         f"[-{highest_dim + 1}, {highest_dim}]")
    if top_dim < 0:
        top_dim += highest_dim + 1

    return top_dim


def string_to_etype_and_idx(etype_string: str, highest_dim: int) -> tuple[int, int]:
    etype_idx = etype_string.split(":")

    if len(etype_idx) == 1:
        etype = etype_idx[0].strip()
        idx = 0
    else:
        etype, idx = etype_idx
        idx = int(idx.strip())

    topdim = etype_to_topdim(etype, highest_dim)
    return topdim, idx


def etype_to_topdim(etype: str, highest_dim: int) -> int:
    etype = etype.upper()

    if etype == "CELL":
        return highest_dim
    if etype == "FACE":
        return highest_dim - 1
    if etype == "EDGE":
        return 1
    if etype == "NODE":
        return 0

    raise ValueError(f"etype name {etype} is not supported, "
					 "available options are: cell, face, edge, node.")


def topdim_to_names(top_dim: int, range: Iterable[str]) -> list[str]:
	result = []
	for name in range:
		if SCHEMA_REGISTRY[name].top_dim == top_dim:
			result.append(name)
	return result


def schema_name_single_parser(name_or_topdim: str | int, idx: int, highest_dim: int, range: Iterable[str]) -> str:
    if isinstance(name_or_topdim, int):
        topdim = ensure_positive_topdim(name_or_topdim, highest_dim)
    else:
        if not isinstance(name_or_topdim, str):
            raise TypeError(f"Expected str or int for name_or_topdim, got {type(name_or_topdim)}")
        if ":" in name_or_topdim:
            topdim, idx = string_to_etype_and_idx(name_or_topdim, highest_dim)
        else:
            if name_or_topdim in range:
                return name_or_topdim
            try:
                topdim = etype_to_topdim(name_or_topdim, highest_dim)
            except ValueError as e:
                raise ValueError(
                    f"Name {name_or_topdim} is not supported, "
                    f"available names are: {list(range)}, "
                    f"or the top dimension symbols: cell, face, edge, node.") from e

    return topdim_to_names(topdim, range)[idx]


def schema_name_multi_parser(name_or_topdim: str | int, highest_dim: int, range: Iterable[str]) -> list[str]:
    if isinstance(name_or_topdim, int):
        topdim = ensure_positive_topdim(name_or_topdim, highest_dim)
    else:
        if not isinstance(name_or_topdim, str):
            raise TypeError(f"Expected str or int for name_or_topdim, got {type(name_or_topdim)}")
        if name_or_topdim in range:
            return [name_or_topdim]
        try:
            topdim = etype_to_topdim(name_or_topdim, highest_dim)
        except ValueError as e:
            raise ValueError(
                f"Name {name_or_topdim} is not supported, "
                f"available names are: {list(range)}, "
                f"or the top dimension symbols: cell, face, edge, node.") from e

    return topdim_to_names(topdim, range)
