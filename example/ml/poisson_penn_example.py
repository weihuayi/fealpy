from fealpy.backend import bm
bm.set_backend('pytorch')
from fealpy.ml import PoissonPENNModel

options = PoissonPENNModel.get_options()  
options['pde'] = 12  # PDE 算例
options['mesh_size'] = 30  # 网格剖分数：30 个，网格节点即为 PENN 的输入数据
options['hidden_size'] = (50, 50, 50, 50)  # 网络结构
options['lr'] = 0.005   # 学习率
options['epochs'] = 2000  # 迭代次数
model = PoissonPENNModel(options=options)
model.run()   
model.show()   

