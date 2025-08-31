from abc import ABC, abstractmethod
from enum import Enum


class ConstitutiveEquation(Enum):
    """Defines the formulation for the viscous term."""
    SIMPLIFIED_LAPLACIAN = 1  # 代表 mu * nabla^2(u)
    FULL_STRESS_TENSOR = 2    # 代表 div(mu * (grad(u) + grad(u)^T))
    CUSTOM = 3



class BaseEquation(ABC):
    """ Base class for simulation """
    def __init__(self, pde):
        self.pde = pde 
        self.constitutive = ConstitutiveEquation.FULL_STRESS_TENSOR
    
    def set_constitutive(self, model_identifier):
        
        new_model = None
        
        if isinstance(model_identifier, ConstitutiveEquation):
            new_model = model_identifier
        elif isinstance(model_identifier, str):
            try:
                new_model = ConstitutiveEquation[model_identifier.upper()]
            except KeyError:
                pass
        elif isinstance(model_identifier, int):
            try:
                new_model = ConstitutiveEquation(model_identifier)
            except ValueError:
                pass

        if new_model:
            self.constitutive = new_model
            print(f"本构方程已成功切换为: {self.constitutive.name} (值为: {self.constitutive.value})")
        else:
            # 1. 动态地从枚举中生成所有有效选项的描述
            valid_options = []
            for member in ConstitutiveEquation:
                # 格式化成 '名称' (数字: 值) 的形式
                option_desc = f"'{member.name.lower()}' (数字: {member.value})"
                valid_options.append(option_desc)
            
            # 2. 将所有选项用逗号连接成一个字符串
            options_str = ", ".join(valid_options)
            
            # 3. 抛出带有详细提示的 ValueError
            raise ValueError(
                f"无法识别的模型标识: '{model_identifier}'.\n"
                f"请输入一个有效的选项。可用选项为: {options_str}"
            )
            
    @property
    def coefs(self):
        """ Coefficients for equation  """
        pass

    @property
    def variables(self):
        """ Variables for equation """
        pass
    
    def __str__(self) -> str:
        """ String representation of the equation """
        pass
