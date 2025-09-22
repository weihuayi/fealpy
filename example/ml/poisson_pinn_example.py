from fealpy.backend import bm
bm.set_backend('pytorch')  
from fealpy.ml import PoissonPINNModel

options = PoissonPINNModel.get_options()   
options['pde'] = 12  # PDE 算例
options['mesh_size'] = 30  # 网格剖分数：30 个
options['hidden_size'] = (50, 50, 50, 50)  # 网络结构
options['lr'] = 0.005   # 学习率
options['epochs'] = 2000  # 迭代次数
options['sampling_mode'] = 'linspace'  # 均匀采样模式
options['npde'] = 28  # 区域内部的采样点：28*28 个
options['nbc'] = 30  # 4 条边界的采样点：30*4 个
model = PoissonPINNModel(options=options)
model.run()   
model.show()  


