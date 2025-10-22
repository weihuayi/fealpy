from .nodetype import CNodeType, PortConf, DataType

class Sample(CNodeType):        
    r"""Generates sample points or configuration points within a rectangular region, 
    with the option to sample from either the interior or the boundary.

    The total number of points generated depends on the sampling mode ('mode'),
    the number of points ('n'), and whether boundary sampling is enabled ('boundary').

    Inputs:
        domain (list | tensor): A sequence defining the rectangular domain, e.g., [x_min, x_max, y_min, y_max].
        mode (str): Sampling method. Either 'random' or 'linspace'. Defaults to 'linspace'.
        n (int): Controls the number of samples. The exact count depends on 'mode' and 'boundary':
            If the mode='linspace', the total number of samples is n^d if boundary=False or n*d if boundary=True
            If the mode='random', the total number of samples is n if boundary=False or n*d if boundary=True.
            where d is the axis of the domain.
        boundary (bool): If True, samples are generated on the boundary. 
                         If False (default), samples are generated in the interior.
        dtype: Data type for the samples. Currently only supports float64.

    Outputs:
        Tensor: A tensor containing the coordinates of the generated sample points.
    """

    TITLE: str = "矩形区域采样器"
    PATH: str = "机器学习.采样"
    DESC: str = "在矩形区域的内部或边界上获取采样点或配置点."

    INPUT_SLOTS = [
        PortConf("domain", DataType.NONE, ttype=1, desc="区域由 PDE 提供", title="计算区域", default=[0,1,0,1]),
        PortConf("mode", DataType.MENU, ttype=0, title="采样模式", default="linspace", items=["random", "linspace"]),
        PortConf("n", DataType.INT, ttype=0, title="分段数", default=10, min_val=1),
        PortConf("boundary", DataType.BOOL, ttype=0, title="是否是在边界采样", default=False),
    ]

    OUTPUT_SLOTS = [PortConf("samples", DataType.TENSOR, title="采样点的笛卡尔坐标")]

    @staticmethod
    def run(domain, mode, n, boundary):
        if boundary:
            from ..ml.sampler import BoxBoundarySampler as Sa
        else:
            from ..ml.sampler import ISampler as Sa

        sampler = Sa(domain, mode=mode)
        samples = sampler.run(n)
        return samples
