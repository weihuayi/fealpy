import numpy as np

class BeamStructureIntegrator:
    def __init__(self, E: float, I: float, q :int = 3):
        """
        BeamStructureIntegrator 类的初始化

        参数:
        E -- 杨氏模量
        I -- 惯性矩
        q -- 积分公式的等级，默认值为3
        """
        self.E = E  # 杨氏模量
        self.I = I  # 惯性矩
        self.q = q # 积分公式
