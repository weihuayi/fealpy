
from dataclasses import dataclass, field
from typing import Any

from ...backend import Tensor
from .relation import Relation

__all__ = ["EntityContext", "EntitySector", "MeshBlock"]


@dataclass(slots=True)
class EntitySector:
    schema_name: str
    indices: Tensor
    indptr: Tensor | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def schema(self):
        from ..schema import SCHEMA_REGISTRY
        return SCHEMA_REGISTRY[self.schema_name]


@dataclass(slots=True)
class MeshBlock:
    positions: Tensor
    sectors: dict[str, EntitySector] = field(default_factory=dict)
    relations: dict[tuple[str, str], Relation] = field(default_factory=dict)
    root_entity_names: list[str] = field(default_factory=list)
    _cache_boundary_info: dict[str, Any] | None = None

    def add_sector(self, sec: EntitySector, *, root: bool = False) -> None:
        self.sectors[sec.schema_name] = sec
        if root and sec.schema_name not in self.root_entity_names:
            self.root_entity_names.append(sec.schema_name)

    def get_sector(self, name: str, /) -> EntitySector:
        return self.sectors[name]

    def has_sector(self, name: str, /) -> bool:
        return name in self.sectors


@dataclass(slots=True, frozen=True)
class EntityContext:
    block: MeshBlock
    sector: EntitySector
