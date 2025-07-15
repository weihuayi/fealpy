import importlib
from typing import Type, Dict, Tuple


def example_import_util(
        category: str,
        key: str,
        data_table: Dict[str, Tuple[str, str]]
    ) -> Type:
    assert isinstance(data_table, dict)
    file_name, class_name = data_table[key]
    m = importlib.import_module(f"fealpy.model.{category}.{file_name}")

    return getattr(m, class_name)