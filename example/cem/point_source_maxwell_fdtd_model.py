from fealpy.cem.model.point_source_maxwell import PointSourceMaxwell
from fealpy.cem.point_source_maxwell_fdtd_model import PointSourceMaxwellFDTDModel
import matplotlib.pyplot as plt


def test_Wave_Interference():
    pde = PointSourceMaxwell(eps=1.0, mu=1.0, domain=[0, 5e-6, 0, 5e-6])

    pde.add_source(
        position=(2e-6,2.5e-6),
        comp='Ez', 
        waveform='sinusoid',
        waveform_params={'freq': 6e14},
        amplitude=1.0
    )

    pde.add_source(
        position=(3e-6,2e-6),
        comp='Ez', 
        waveform='sinusoid',
        waveform_params={'freq': 3e14},
        amplitude=2.0
    )

    # 配置选项
    options = {
        # 'dt': 1e-12,  # 1皮秒
        'R': 0.5,     # Courant数
        'boundary': 'UPML',
        'pml_width': 10,
        'maxstep': 5000,
        'save_every': 1
    }

    # 创建计算模型
    model = PointSourceMaxwellFDTDModel(pde, n=100, options=options)
    nt = 200
    # 运行模拟 - 只需要指定时间步数
    field_history = model.run(nt=nt)

    print(model)  # 打印模型信息


    # 方式2：显示2D切片
    fig, axes = model.show_field(
        field_history,
        step_index = nt-1,
        field_component='Ez', 
        plot_type='imshow',     # 2D图像显示
        cmap='rainbow'
    )

    # 方式3：显示其他方向的切片
    fig, axes = model.show_field(
        field_history,
        step_index = nt-1,
        field_component='Hx',
        plot_type='imshow',
        cmap='coolwarm'
    )

    plt.show()

    # 2. 生成动画
    ani = model.show_animation(
        field_history,
        field_component='Ez',
        frames=nt-1,
        interval=50,
        fname='test_Wave_Interference.mp4',
        cmap='rainbow',
        vmin = -0.2,
        vmax = 0.2
    )

    # 3. 绘制时间序列
    fig, ax = model.plot_time_series(
        field_history,
        positions=[ (3.5e-6,2.5e-6)],
        field_component='Ez',
    )
    plt.show()

    return None

# 波的散射
def test_Wave_Scattering():
    pde = PointSourceMaxwell(eps=1.0, mu=1.0, domain=[0, 5e-6, 0, 5e-6])

    pde.add_source(
        position=(3e-6,2e-6),
        comp='Ez', 
        waveform='sinusoid',
        waveform_params={'freq': 6e14},
        amplitude=1.0
    )

    pde.add_object([3e-6,4e-6,1e-6,2e-6],eps = 4)
    pde.add_object([3e-6,3.5e-6,2.5e-6,3.5e-6],eps = 10)

    # 配置选项
    options = {
        # 'dt': 1e-12,  # 1皮秒
        'R': 0.5,     # Courant数
        'boundary': 'UPML',
        'pml_width': 10,
        'maxstep': 5000,
        'save_every': 1
    }

    # 创建计算模型
    model = PointSourceMaxwellFDTDModel(pde, n=100, options=options)
    nt = 200
    # 运行模拟 - 只需要指定时间步数
    field_history = model.run(nt=nt)

    print(model)  # 打印模型信息


    # 方式2：显示2D切片
    fig, axes = model.show_field(
        field_history,
        step_index = nt-1,
        field_component='Ez', 
        plot_type='imshow',     # 2D图像显示
        cmap='rainbow'
    )

    # 方式3：显示其他方向的切片
    fig, axes = model.show_field(
        field_history,
        step_index = nt-1,
        field_component='Hx',
        plot_type='imshow',
        cmap='coolwarm'
    )

    plt.show()

    # 2. 生成动画
    ani = model.show_animation(
        field_history,
        field_component='Ez',
        frames=nt-1,
        interval=50,
        fname='test_Wave_Scattering.mp4',
        cmap='rainbow',
        vmin = -0.2,
        vmax = 0.2
    )

    # 3. 绘制时间序列
    fig, ax = model.plot_time_series(
        field_history,
        positions=[ (3.5e-6,2.5e-6)],
        field_component='Ez',
    )
    plt.show()

    return None

def test_Wave_Diffraction():
    pde = PointSourceMaxwell(eps=1.0, mu=1.0, domain=[0, 5e-6, 0, 5e-6])
    model = PointSourceMaxwellFDTDModel(pde, n=100)
    pde.add_source(
        position=(2e-6, 2.5e-6),
        comp='Ez', 
        waveform='gaussian_enveloped_sine',
        waveform_params={'freq':6e14,'t0':10 * model.fdtd.dt,'tau':3 * model.fdtd.dt },
        amplitude=2.0)
    

    pde.add_object([0,2.10e-6,2.5e-6,2.6e-6],eps = 10000)
    pde.add_object([2.35e-6,2.65e-6,2.5e-6,2.6e-6],eps = 10000)
    pde.add_object([2.9e-6,5e-6,2.5e-6,2.6e-6],eps = 10000)

    # 配置选项
    options = {
        # 'dt': 1e-12,  # 1皮秒
        'R': 0.5,     # Courant数
        'boundary': 'UPML',
        'pml_width': 10,
        'maxstep': 5000,
        'save_every': 1
    }

    # 创建计算模型
    model = PointSourceMaxwellFDTDModel(pde, n=100, options=options)


    
    nt = 200

    # 运行模拟 - 只需要指定时间步数
    field_history = model.run(nt=nt)

    print(model)  # 打印模型信息


    # 2. 生成动画
    ani = model.show_animation(
        field_history,
        field_component='Ez',
        frames=nt-1,
        interval=50,
        fname='test_Wave_Diffraction.mp4',
        cmap='rainbow',
        vmin = -0.2,
        vmax = 0.2
    )

    # 3. 绘制时间序列
    fig, ax = model.plot_time_series(
        field_history,
        positions=[ (3.5e-6,2.5e-6)],
        field_component='Ez',
    )
    plt.show()

    return None


if __name__ == "__main__":
    test_Wave_Interference()

    test_Wave_Scattering()

    test_Wave_Diffraction()




