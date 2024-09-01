# run_mbb_beam.py

from fealpy.experimental.backend import backend_manager as bm
from app.stopt.soptx.cases.mbb_beam_case import MBBBeamCase
from app.stopt.soptx.optalg.optimizer_oc import OCOptimizer  # 使用 OC 方法优化

def main():
    # 设置后端
    bm.set_backend('numpy')  # 根据需求选择 'numpy' 或 'pytorch'

    # 实例化 MBB 梁案例
    mbb_case = MBBBeamCase("top88")
    print(mbb_case)

    # 初始化优化器 (使用 OCOptimizer 类)
    optimizer = OCOptimizer(
        material_properties=mbb_case.material_properties,
        geometry_properties=mbb_case.geometry_properties,
        filter_properties=mbb_case.filter_properties,
        constraint_conditions=mbb_case.constraint_conditions,
        boundary_conditions=mbb_case.boundary_conditions,
        termination_criteria=mbb_case.termination_criterias
    )

    # 运行优化
    optimizer.run()
    
    # 输出结果
    results = optimizer.get_results()
    print("Optimization Results:")
    print(results)

    # 可选：保存结果或进行可视化
    # optimizer.save_results("output_file_path")
    # optimizer.plot_results()

if __name__ == "__main__":
    main()
