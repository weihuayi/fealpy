from abc import ABC, abstractmethod
from typing import Dict, Any, Type

class SimulationParameters(ABC):

    def __str__(self) -> str:
        from textwrap import indent
        """以树形结构可视化所有参数，支持嵌套字典和列表"""
        def format_value(v):
            if isinstance(v, (list, tuple)):
                return f"[{', '.join(map(str, v))}]"
            elif isinstance(v, dict):
                return "{\n" + indent(",\n".join(
                    f"{k}: {format_value(v)}" for k, v in v.items()
                ), "  ") + "\n}"
            elif isinstance(v, str):
                return f'"{v}"'
            else:
                return str(v)

        sections = []
        for section, params in self._params.items():
            items = [f"{k}: {format_value(v)}" for k, v in params.items()]
            sections.append(f"{section}:\n" + indent("\n".join(items), "  "))

        return "=== Simulation Parameters: ===\n" + indent("\n\n".join(sections), "  ")



class SimulationBase(ABC):
    def __init__(self, method,  output=False):
        """
        初始化模拟器
        """
        self.method = method
        self.equation = method.equation  
        #self._initialize()
    
    @abstractmethod
    def _initialize_variables(self):
        """抽象方法：子类需实现具体的方程创建逻辑"""
        pass
    
    @abstractmethod
    def run_one_step(self):
        """执行单步求解"""
        pass

    @abstractmethod
    def run(self):
        """执行全流程求解"""
        pass

