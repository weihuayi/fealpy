import numpy as np
from dataclasses import dataclass

@dataclass
class PartitionType():
    def __init__(self, partition_type, *parameters):
        """
        @param partition_type: 分区类型. "nonoverlap" or "overlap0" or "overlap1".
        @param parameters: 分区参数.
        """
        self.partition_type = partition_type
        self.parameters = parameters

    def __eq__(self, other):
        return self.partition_type == other

