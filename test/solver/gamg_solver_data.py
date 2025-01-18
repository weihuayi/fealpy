import numpy as np


init_data = [{'theta': 0.025, # 粗化系数
              'csize': 50, # 最粗问题规模
              'ctype': 'C', # 粗化方法
              'itype': 'T', # 插值方法
              'ptype': 'W', # 预条件类型
              'sstep':  2, # 默认光滑步数
              'isolver': 'CG', # 默认迭代解法器
              'maxit':  200,   # 默认迭代最大次数
              'csolver': 'direct', # 默认粗网格解法器
              'rtol': 1e-8,      # 相对误差收敛阈值
              'atol': 1e-8,      # 绝对误差收敛阈值
             }]
