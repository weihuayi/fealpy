from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, List, Dict, Literal, Type, Union, Sequence
import matplotlib.colors as mcolors
from fealpy import logger


# 辅助函数：十六进制颜色转RGB
def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))


def parse_color(color: Union[str, Sequence[float]]) -> tuple:
    """
    将颜色名称、简写或十六进制转换为 RGB 浮点数（0.0~1.0）

    支持：
    - 简写：'r'（红）、'g'（绿）、'b'（蓝）、'c'（青）、'm'（品红）、'y'（黄）、'k'（黑）、'w'（白）
    - 颜色名称：'red', 'blue', 'steelblue' 等
    - 十六进制字符串：'#FF0000'
    - RGB 元组：(1.0, 0.0, 0.0)

    Returns
    -------
    tuple
        RGB 浮点数值（范围 0.0~1.0）
    """
    # 如果是 RGB 元组/列表，直接返回
    if isinstance(color, (tuple, list)):
        if len(color) != 3:
            raise ValueError("RGB 格式需为长度为3的元组或列表。")
        return tuple(color)

    # 处理字符串（名称、简写、十六进制）
    try:
        return mcolors.to_rgb(color)
    except ValueError:
        raise ValueError(f"无效颜色格式：'{color}'。支持颜色名称（如 'red'）、简写（如 'r'）或十六进制（如 '#FF0000'）。")


def _make_default_mapping(*names: str):
    return {k: k for k in names}

ATTRIBUTE_MAPPING = _make_default_mapping(

)

FUNCTION_MAPPING = _make_default_mapping(

)

TRANSFORMS_MAPPING = _make_default_mapping(

)


class ModuleProxy():
    @classmethod
    def attach_attributes(cls, mapping: Dict[str, str], source: Any, /):
        for target_key, source_key in mapping.items():
            if (source_key is None) or (source_key == ''):
                continue
            if hasattr(source, source_key):
                setattr(cls, target_key, getattr(source, source_key))

    @classmethod
    def attach_methods(cls, mapping: Dict[str, str], source: Any, /):
        for target_key, source_key in mapping.items():
            if (source_key is None) or (source_key == ''):
                continue
            if hasattr(cls, target_key):
                # Methods will not be copied from source if implemented manually.
                logger.debug(f"`{target_key}` already defined. "
                             f"Skip the copy from {source.__name__}.")
                continue
            if hasattr(source, source_key):
                setattr(cls, target_key, staticmethod(getattr(source, source_key)))
            else:
                logger.info(f"`{source_key}` not found in {source.__name__}. "
                            f"Method `{target_key}` remains unimplemented.")

    @classmethod
    def show_unsupported(cls, signal: bool, function_name: str, arg_name: str) -> None:
        if signal:
            logger.warning(f"{cls.__name__} does not support the "
                           f"'{arg_name}' argument in the function {function_name}. "
                           f"The argument will be ignored.")


class GeometryKernelAdapterBase(ABC, ModuleProxy):
    """几何内核适配器基类，定义所有内核必须实现的接口"""
    _available_adapters: Dict[str, Type["GeometryKernelAdapterBase"]] = {}

    def __init_subclass__(cls, adapter_name: str, **kwargs):
        super().__init_subclass__(**kwargs)

        if adapter_name != "":
            cls._available_adapters[adapter_name.lower()] = cls
            cls.backend_name = adapter_name
        else:
            raise ValueError("Backend name cannot be empty.")

    @abstractmethod
    def initialize(self, config: Optional[dict] = None) -> None:
        """初始化几何内核（如加载依赖库、配置参数）"""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """清理内核资源"""
        pass