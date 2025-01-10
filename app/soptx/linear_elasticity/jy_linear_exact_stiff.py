import re

from fealpy.backend import backend_manager as bm

from fealpy.mesh import HexahedronMesh
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.fem.vector_source_integrator import VectorSourceIntegrator
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.linear_form import LinearForm
from fealpy.fem.dirichlet_bc import DirichletBC
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian

def read_mtx_file_global(filename):
    """
    读取 Matrix Market (.mtx) 文件并将数据转换为二维数组，
    当值为 1.000000000000000e+36 时，将其赋值为 0。
        
    返回:
    numpy.ndarray
        形状为 (48, 48) 的二维数组
    """
    # 初始化一个48x48的零矩阵
    result = bm.zeros((48, 48))
    
    # 读取文件
    with open(filename, 'r') as file:
        for line in file:
            # 跳过空行和注释行
            if not line.strip() or line.startswith('%'):
                continue
                
            # 将每行分割成数组
            parts = line.strip().split()
            if len(parts) == 3:  # 确保行格式正确
                # 解析数据
                i = int(parts[0]) - 1          # 行索引 (0-47)
                j = int(parts[1]) - 1          # 列索引 (0-47)
                value = float(parts[2])        # 值
                
                # 检查索引是否在范围内
                if 0 <= i < 48 and 0 <= j < 48:
                    # 如果值为 1.000000000000000e+36，则赋值为 0，否则赋原值
                    if value == 1.000000000000000e+36:
                        result[i, j] = 1.0
                        result[j, i] = 1.0  # 处理对称矩阵
                    else:
                        result[i, j] = value
                        result[j, i] = value  # 处理对称矩阵
        
    return result

def read_mtx_file(filename):
    """
    读取 mtx 文件并将数据转换为三维数组
        
    返回:
    numpy.ndarray
        形状为 (7, 24, 24) 的三维数组
    """
    result = bm.zeros((7, 24, 24))
    
    # 读取文件
    with open(filename, 'r') as file:
        for line in file:
            # 跳过空行
            if not line.strip():
                continue
                
            # 将每行分割成数组
            parts = line.strip().split()
            if len(parts) == 4:  # 确保行格式正确
                # 解析数据
                matrix_idx = int(parts[0])     # 矩阵索引 (0-6)
                i = int(parts[1]) - 1          # 行索引 (0-23)
                j = int(parts[2]) - 1          # 列索引 (0-23)
                value = float(parts[3])        # 值
                
                # 将值存入对应位置
                if 0 <= matrix_idx < 7 and 0 <= i < 24 and 0 <= j < 24:
                    result[matrix_idx, i, j] = value
    
    return result

def extract_stiff_matrix_ansys_by_elem(file_path, elem_number, elem_type="SOLID185"):

    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 动态构建目标字符串
    target_string = f"STIFFNESS MATRIX FOR ELEMENT        {elem_number}          {elem_type}"

    start_index = None
    for i, line in enumerate(lines):
        if target_string in line:
            start_index = i + 1  # 矩阵数据从下一行开始
            break
    
    if start_index is None:
        raise ValueError("未找到目标字符串。")
    
    matrix = bm.zeros((24, 24), dtype=bm.float64)
    
    for row_num in range(24):
        row_data = []
        for j in range(4):  # 4行组成一个矩阵行
            current_line = lines[start_index + row_num * 4 + j].strip()
            parts = current_line.split()
            # 第一行包含行号，忽略第一个元素
            if j == 0:
                data = parts[1:]
            else:
                data = parts
            row_data.extend(data)
        
        if len(row_data) != 24:
            raise ValueError(f"第 {row_num+1} 行的数据不完整。")
        
        # 转换为浮点数
        row_data = [float(value.replace('D', 'E').replace('d', 'E')) for value in row_data]
        matrix[row_num] = row_data
    
    return matrix

class BoxDomainLinear3d():
    def __init__(self):
        self.eps = 1e-12

    def domain(self):
        return [0, 1, 0, 1, 0, 1]

    @cartesian
    def source(self, points: TensorLike):
        val = bm.zeros(points.shape, 
                       dtype=points.dtype, device=bm.get_device(points))
        
        return val

    @cartesian
    def solution(self, points: TensorLike):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        val = bm.zeros(points.shape, 
                       dtype=points.dtype, device=bm.get_device(points))
        
        val[..., 0] = 1e-3 * (2*x + y + z) / 2
        val[..., 1] = 1e-3 * (x + 2*y + z) / 2
        val[..., 2] = 1e-3 * (x + y + 2*z) / 2

        return val

    @cartesian
    def dirichlet(self, points: TensorLike) -> TensorLike:

        result = self.solution(points)

        return result

node = bm.array([[0.249, 0.342, 0.192],
                [0.826, 0.288, 0.288],
                [0.850, 0.649, 0.263],
                [0.273, 0.750, 0.230],
                [0.320, 0.186, 0.643],
                [0.677, 0.305, 0.683],
                [0.788, 0.693, 0.644],
                [0.165, 0.745, 0.702],
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1]],
            dtype=bm.float64)

cell = bm.array([[0, 1, 2, 3, 4, 5, 6, 7],
                [0, 3, 2, 1, 8, 11, 10, 9],
                [4, 5, 6, 7, 12, 13, 14, 15],
                [3, 7, 6, 2, 11, 15, 14, 10],
                [0, 1, 5, 4, 8, 9, 13, 12],
                [1, 2, 6, 5, 9, 10, 14, 13],
                [0, 4, 7, 3, 8, 12, 15, 11]],
                dtype=bm.int32)

file_name_global = "/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/mtx/lin_exa_abaqus_global_STIF2.mtx"
K_abaqus_proto = read_mtx_file_global(file_name_global)

file_name_local = "/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/mtx/lin_exa_abaqus_local_STIF2.mtx"
KE_abaqus_proto = read_mtx_file(file_name_local)
KE_abaqus_utri = bm.triu(KE_abaqus_proto, 1)
KE_abaqus_ltri = bm.transpose(KE_abaqus_utri, (0, 2, 1))
KE_abaqus = KE_abaqus_proto + KE_abaqus_ltri
KE0_abaqus = KE_abaqus[0]

mesh = HexahedronMesh(node, cell)

GD = mesh.geo_dimension()
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()
print(f"NN = {NN}, NC = {NC}")
cm = mesh.cell_volume()
node = mesh.entity('node')
cell = mesh.entity('cell')

p = 1
q = 2
space = LagrangeFESpace(mesh, p=p, ctype='C')
cell2dof = space.cell_to_dof()
sgdof = space.number_of_global_dofs()
print(f"sgdof: {sgdof}")
tensor_space = TensorFunctionSpace(space, shape=(-1, 3)) # gd_priority
cell2tdof = tensor_space.cell_to_dof()
map = [ 0,  1,  2, 12, 13, 14,  9, 10, 11, 21, 22, 23,  3,  4,  5, 15, 16,
       17,  6,  7,  8, 18, 19, 20]
# map = cell2tdof[0]
tldof = tensor_space.number_of_local_dofs()
tgdof = tensor_space.number_of_global_dofs()

KE_abaqus_map = KE_abaqus[:, :, map]
KE_abaqus_map = KE_abaqus_map[:, map, :]
KE0_abaqus_map = KE_abaqus_map[0]

pde = BoxDomainLinear3d()

# 刚度矩阵
E = 2.1e5
nu = 0.3
lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
linear_elastic_material = LinearElasticMaterial(name='E_nu', 
                                                elastic_modulus=E, poisson_ratio=nu, 
                                                hypo='3D', device=bm.get_device(mesh))
integrator_K_bbar = LinearElasticIntegrator(material=linear_elastic_material, 
                                            q=q, method='C3D8_BBar')
KE_bbar = integrator_K_bbar.c3d8_bbar_assembly(space=tensor_space)
KE0_bbar = KE_bbar[0]

error_KE = bm.max(bm.abs(KE_abaqus_map - KE_bbar))
print(f"error_KE: {error_KE}")

bform = BilinearForm(tensor_space)
bform.add_integrator(integrator_K_bbar)
K_bbar = bform.assembly(format='csr')
K_bbar_dense1 = K_bbar.to_dense()

integrator_F = VectorSourceIntegrator(source=pde.source, q=q)
lform = LinearForm(tensor_space)    
lform.add_integrator(integrator_F)
F = lform.assembly()

# 边界条件处理
uh_bd = bm.zeros(tensor_space.number_of_global_dofs(), 
                dtype=bm.float64, device=bm.get_device(mesh))
uh_bd, isDDof = tensor_space.boundary_interpolate(gd=pde.dirichlet, 
                                                uh=uh_bd, threshold=None, method='interp')
bd_indcies = bm.where(isDDof)[0]
K_bbar_dense2 = bm.copy(K_bbar_dense1)
K_bbar_dense2[bd_indcies, bd_indcies] = 1

dbc = DirichletBC(space=tensor_space, 
                gd=pde.dirichlet, 
                threshold=None, 
                method='interp')
K_bbar, F = dbc.apply(K_bbar, F)
K_bbar_dense3 = K_bbar.to_dense()
error_K = bm.max(bm.abs(K_bbar_dense2 - K_abaqus_proto))
print(f"error_K: {error_K}")

print("--------------------------------")