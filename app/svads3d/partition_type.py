import numpy as np
from enum import Enum

class PartitionType(Enum):
    # 无重叠
    NONE = 1
    # 重叠1
    OVERLAP1 = 2
    # 重叠2
    OVERLAP2 = 3
