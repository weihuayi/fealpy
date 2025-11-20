"""
examples/example_sources_demo.py

演示：在你的 YeeUniformMesher 上创建多个源并注入（2D 与 3D）。
依赖：
    - fealpy.cem.sources.Source, SourceManager
    - fealpy.cem.mesh.YeeUniformMesher  (你已提供)
    - fealpy.backend.backend_manager as bm
"""

from fealpy.cem.model.source import Source, SourceManager, gaussian_pulse, gaussian_enveloped_sine, ricker_wavelet
from fealpy.backend import backend_manager as bm

# 你的 Yee 网格（示例）
from fealpy.cem.mesh import YeeUniformMesher

def infer_field_shapes_from_yee(E_fields, H_fields):
    """尝试从 yee 对象获取常用 field shapes；否则使用常见回退约定。
    返回 (E_shapes, H_shapes) 字典，E_shapes 有键 'x','y','z'，H_shapes 同理。
    """

    Es = {}
    Hs = {}
    key_e = E_fields.keys()
    key_h = H_fields.keys()

    for xt in key_e:
        Es[xt] = E_fields[xt].shape
    for xt in key_h:
        Hs[xt] = H_fields[xt].shape

    return Es, Hs

def example_2d_multi_sources():
    print("=== 2D example: multiple sources demo ===")
    # 构造 Yee 网格（你给的）
    yee2d = YeeUniformMesher((0.0,1.0,0.0,1.0), nx=32, ny=32)
    
    
    E_fields,_ = yee2d._init_fields_dict("E", ["z"], num_frames=0, axis_type=False)
    H_fields,_ = yee2d._init_fields_dict("H", ["x", "y"], num_frames=0, axis_type=False)

    E_shapes, H_shapes = infer_field_shapes_from_yee(E_fields, H_fields)

    print("Inferred E shapes:", E_shapes)
    print("Inferred H shapes:", H_shapes)

    # 创建 SourceManager
    sm = SourceManager()

    # 添加多个 source（示例：两个 soft Gaussian 在不同位置，一个 hard 正弦在边上）
    # 推荐使用网格索引定位（可靠）
    s1 = Source(position=(8, 8), comp='Ez', waveform=lambda t: gaussian_pulse(t, t0=10.0, tau=3.0),
                amplitude=1.0, spread=1, injection='soft')
    s2 = Source(position=(24, 20), comp='Ez', waveform=lambda t: gaussian_enveloped_sine(t, freq=0.05, t0=12.0, tau=4.0),
                amplitude=0.8, spread=2, injection='soft')
    s3 = Source(position=(0, 16), comp='Ez', waveform=lambda t: 0.5 * (1.0 if (t % 20.0) < 5.0 else 0.0),
                amplitude=2.0, spread=0, injection='hard')  # 边上的 hard source

    sm.add(s1); sm.add(s2); sm.add(s3)
    print("Added 3 sources: s1(small gaussian), s2(enveloped sine), s3(hard pulse at boundary)")

    # 一个非常简单的 time loop：仅演示 source 注入（真实仿真应在每步先 update H -> 注入 -> update E）
    nt = 40
    dt = 1.0
    for n in range(nt):
        t = n * dt
        # 在此处通常应先 update_H(), 然后 apply sources（某些注入方式注入到 E 或 H 在不同时间位置）
        sm.apply_all(t, yee2d, E_fields, H_fields)

        if n % 10 == 0:
            # 输出 Ez 的统计信息，直观展示注入生效
            ez = E_fields['z']
            print(f" step {n:03d}, t={t:.1f}, Ez max={float(bm.max(bm.abs(ez))):.6e}, Ez mean={float(bm.mean(ez)):.6e}")

    # 最终快照统计
    ez = E_fields['z']
    print("Final Ez snapshot stats: max, min, mean ->", float(bm.max(ez)), float(bm.min(ez)), float(bm.mean(ez)))
    return yee2d, E_fields, H_fields

def example_3d_multi_sources():
    print("=== 3D example: multiple sources demo ===")
    yee3d = YeeUniformMesher((0.0,1.0,0.0,1.0,0.0,1.0), nx=16, ny=16, nz=16)

    E_fields,_ = yee3d._init_fields_dict("E", ["x", "y", "z"], num_frames=0, axis_type=False)
    H_fields,_ = yee3d._init_fields_dict("H", ["x", "y", "z"], num_frames=0, axis_type=False)

    E_shapes, H_shapes = infer_field_shapes_from_yee(E_fields, H_fields)
    print("Inferred E shapes (3D):", E_shapes)
    print("Inferred H shapes (3D):", H_shapes)

   
    sm = SourceManager()

    # 3D 多源：一个中心 Ricker，两个角上小高斯
    s_center = Source(position=(8,8,8), comp='Ez', waveform=lambda t: ricker_wavelet(t, t0=8.0, f=0.25),
                      amplitude=1.0, spread=1, injection='soft')
    s_corner1 = Source(position=(3,3,3), comp='Ez', waveform=lambda t: gaussian_pulse(t, t0=6.0, tau=2.0),
                       amplitude=0.6, spread=1, injection='soft')
    s_corner2 = Source(position=(12,12,12), comp='Ez', waveform=lambda t: gaussian_pulse(t, t0=6.0, tau=2.0),
                       amplitude=0.6, spread=1, injection='soft')
    sm.add(s_center); sm.add(s_corner1); 
    sm.add(s_corner2)
    print("Added 3 sources in 3D (center Ricker + two corner Gaussians)")

    nt = 24
    dt = 1.0
    for n in range(nt):
        t = n * dt
        sm.apply_all(t, yee3d, E_fields, H_fields)
        if n % 6 == 0:
            ez = E_fields['z']
            print(f" step {n:03d}, t={t:.1f}, Ez max={float(bm.max(bm.abs(ez))):.6e}, Ez mean={float(bm.mean(ez)):.6e}")

    ez = E_fields['z']
    print("Final Ez snapshot stats (3D): max, min, mean ->", float(bm.max(ez)), float(bm.min(ez)), float(bm.mean(ez)))
    return yee3d, E_fields, H_fields

if __name__ == "__main__":
    # 运行 2D demo
    yee2d, E2, H2 = example_2d_multi_sources()

    # 运行 3D demo（可注释掉以节约时间）
    yee3d, E3, H3 = example_3d_multi_sources()

    print("Example finished. You can inspect returned E_fields/H_fields or integrate with solver update steps.")
