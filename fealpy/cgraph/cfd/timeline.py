from ..nodetype import CNodeType, PortConf, DataType

class CFDTimeline(CNodeType):
    r"""Uniform time discretization for CFD simulations.

    Inputs:
        T0 (float): Initial time.
        T1 (float): Final time.
        NT (int): Number of time intervals.
    
    Outputs:
        T0 (float): Initial time.
        T1 (float): Final time.
        NL (int): Number of time levels.
    """
    TITLE: str = "均匀时间剖分"
    PATH: str = "流体.时间剖分"
    INPUT_SLOTS = [
        PortConf("T0", DataType.FLOAT, 0, title="初始时间",default=0.0),
        PortConf("T1", DataType.FLOAT, 0, title="结束时间", default=1.0),
        PortConf("NT", DataType.INT, 0, title="时间剖分数", default=1000, min_val=1)
    ]
    OUTPUT_SLOTS = [
        PortConf("T0", DataType.FLOAT, title="初始时间"),
        PortConf("T1", DataType.FLOAT, title="结束时间"),
        PortConf("NL", DataType.INT, title="时间层数")
    ]
    @staticmethod
    def run(T0, T1, NT):
        from fealpy.cfd.simulation.time import UniformTimeLine
        timeline = UniformTimeLine(T0, T1, NT)
        return (timeline.T0, timeline.T1, timeline.NL)