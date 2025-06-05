
from fealpy.backend import backend_manager as bm
from fealpy.ml.sampler import CircleCollocator
from fealpy.typing import TensorLike

from lafemims.data_generator import NearFieldDataFEMGenerator2d
from lafemims.functional import levelset_fourier, generate_scatterers_fourier


# 设置随机种子
SEED = 2025

# 定义计算域
domain = [-6, 6, -6, 6]

# 定义入射波函数
u_inc = 'cos(d_0*k*x + d_1*k*y) + sin(d_0*k*x + d_1*k*y) * 1j'

# 定义波矢量方向和波数
d = [[-bm.sqrt(0.5), bm.sqrt(0.5)]]
k = [2 * bm.pi]

#选择散射体复杂度、散射体中心点以及生成散射体个数
M = 20
origin = bm.array([0.0, 0.0])
num_of_scatterers = 40000

#确定外层接收点
num_of_receiver_points = 50
receiver_points = CircleCollocator(0, 0, 5).run(num_of_receiver_points).detach().numpy()

#选择某个散射体
coefficients = generate_scatterers_fourier(M, num_of_scatterers, SEED)          # shape == [2 * M + 1]

def main(idx: int, complexity: int, coefficient: TensorLike, origin:TensorLike, save_path: str, test_type: str):

    # 定义指示函数
    ind_func = lambda p: levelset_fourier(p, complexity, coefficient, origin)

    # 创建近场数据生成器
    generator = NearFieldDataFEMGenerator2d(
        domain=domain,
        mesh='UniformMesh',
        nx=100,
        ny=100,
        p=1,
        q=3,
        u_inc=u_inc,
        levelset=ind_func,
        d=d,
        k=k,
        receiver_points=receiver_points
    )
    if test_type == "save":
        generator.save(save_path, idx)
    elif test_type == "visualization":
        return generator
    else:
        raise ValueError("test_type must be 'save' or 'visualization'")
    

if __name__ == '__main__':

    SAVE_PATH = 'D:/ims_problem/test'
    TEST_TYPE = 'visualization'

    if TEST_TYPE == 'save':
        from multiprocessing import Pool
        from tqdm import tqdm
        import time

        NUM_OF_POOL = 8         # 并行进程数
        AMOUNT_OF_DATA = 100    # 生成数据数量

        def update_progress_bar(*_):
            progress_bar.update()

        pool = Pool(NUM_OF_POOL)
        processes = []

        # 创建进度条
        with tqdm(total=AMOUNT_OF_DATA, desc="数据生成进度") as progress_bar:
            start_time = time.time()
            for idx_ in range(AMOUNT_OF_DATA):
                p = pool.apply_async(main, args=(idx_, M, coefficients[idx_, ...], origin, SAVE_PATH, TEST_TYPE), callback=update_progress_bar)
                processes.append(p)

            pool.close()
            pool.join()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"生成数据时间: {elapsed_time} 秒")

    else:
        IDX = 99                # 选择要可视化的数据索引
        generator = main(IDX, M, coefficients[IDX, ...], origin, SAVE_PATH, TEST_TYPE)
        generator.visualization_of_nearfield_data(k=k[-1], d=d[0])
