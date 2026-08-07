# FEALPy | Old Type Dependency Report | mesh_05b_ai_correction

## 扫描范围与命令

扫描范围：`fealpy/`、`app/`、`example/`、`test/`、`tests/`。
关键词：`from fealpy.mesh import`、`from fealpy.mesh_old`、`TriangleMesh`、`QuadrangleMesh`、`TetrahedronMesh`、`HexahedronMesh`、`EdgeMesh`、`IntervalMesh`、`PrismMesh`、`PyramidMesh`。

## 结论摘要

- 命中文件总数：722。
- `fealpy/`：354 个文件命中。
- `app/`：120 个文件命中。
- `example/`：67 个文件命中。
- `test/`：177 个文件命中。
- `tests/`：4 个文件命中。

按旧类型/导入关键词统计：
- `TriangleMesh`：445
- `TetrahedronMesh`：153
- `QuadrangleMesh`：108
- `EdgeMesh`：66
- `HexahedronMesh`：66
- `IntervalMesh`：59
- `PrismMesh`：8
- `PyramidMesh`：1

## 阻塞性判断

- 对本轮 schema 基本算例不构成直接阻塞：`uv run python -m pytest tests/mesh/unit/schema -q` 已通过。
- 对后续“新网格最终替换旧网格”构成系统性迁移风险：上层模块、示例、旧测试仍大量显式引用旧 mesh 类型。
- 本任务按 A5 只识别依赖，不执行全量替换；建议后续单独建立兼容层/迁移任务。

## 样例命中清单（每个顶层目录最多 12 项）

### `fealpy/`
- `fealpy/cdg/geometry/metric.py` :: from fealpy.mesh import, TriangleMesh
- `fealpy/cdg/io/loader.py` :: TriangleMesh
- `fealpy/cdg/model/circle_harmonic_map_model.py` :: TriangleMesh
- `fealpy/cdg/operators/mesh_operator.py` :: from fealpy.mesh import, TriangleMesh
- `fealpy/cdg/solvers/base_solver.py` :: from fealpy.mesh import, TriangleMesh
- `fealpy/cdg/topology/boundary.py` :: from fealpy.mesh import, TriangleMesh
- `fealpy/cdg/topology/euler.py` :: from fealpy.mesh import, TriangleMesh
- `fealpy/cem/generator/eit_data_generator.py` :: from fealpy.mesh import
- `fealpy/cem/generator/laplace_data_generator.py` :: from fealpy.mesh import, TriangleMesh
- `fealpy/cem/mesh/metalenses_mesher.py` :: from fealpy.mesh import, TetrahedronMesh
- `fealpy/cem/mesh/metalenses_mesher_tet.py` :: from fealpy.mesh import, TetrahedronMesh
- `fealpy/cem/mesh/yee_uniform_mesher.py` :: from fealpy.mesh import

### `app/`
- `app/fluid/meshmove.py` :: from fealpy.mesh import, QuadrangleMesh
- `app/fluid/TriplePointShockInteractionModel.py` :: from fealpy.mesh import
- `app/fracturex/fracturex/cases/phase_field/model0_example.py` :: from fealpy.mesh import, TriangleMesh
- `app/fracturex/fracturex/cases/phase_field/model3d.py` :: from fealpy.mesh import, TetrahedronMesh, HexahedronMesh
- `app/fracturex/fracturex/cases/phase_field/square_domian_with_fracture.py` :: from fealpy.mesh import, TriangleMesh, QuadrangleMesh
- `app/fracturex/fracturex/tests/test_fracture_constitutive_model.py` :: from fealpy.mesh import, TriangleMesh
- `app/fracturex/fracturex/tests/test_main_solver.py` :: from fealpy.mesh import, TriangleMesh
- `app/FuelRodSim/fuel_rod_mesher.py` :: from fealpy.mesh import, TriangleMesh, TetrahedronMesh
- `app/FuelRodSim/heat_equation/box_2dexample.py` :: from fealpy.mesh import, TriangleMesh
- `app/FuelRodSim/heat_equation/box_3dexample.py` :: from fealpy.mesh import, TetrahedronMesh
- `app/FuelRodSim/heat_equation/heat_equation_solver.py` :: from fealpy.mesh import, TriangleMesh, TetrahedronMesh
- `app/FuelRodSim/heat_equation/true_solution_2dexample.py` :: from fealpy.mesh import, TriangleMesh

### `example/`
- `example/cem/metalenses_mesher.py` :: from fealpy.mesh import, TetrahedronMesh
- `example/cfd/pipe_bend_turbulent_flow.py` :: from fealpy.mesh import, TriangleMesh, TetrahedronMesh
- `example/csm/fem/bar25_lfem_example.py` :: EdgeMesh
- `example/csm/fem/bar924_lfem_example.py` :: EdgeMesh
- `example/csm/fem/channel_beam_lfem_example.py` :: EdgeMesh
- `example/csm/fem/timobeam_axle_lfem_example.py` :: EdgeMesh
- `example/csm/fem/truss_tower_lfem_example.py` :: EdgeMesh
- `example/fdm/poisson_fdm_example_1d.py` :: from fealpy.mesh import
- `example/fdm/poisson_fdm_example_2d.py` :: from fealpy.mesh import
- `example/fem/dld_microfluidic_chip_lfem_3d_example.py` :: from fealpy.mesh import, TriangleMesh
- `example/fem/dld_microfluidic_chip_lfem_example.py` :: from fealpy.mesh import, TriangleMesh
- `example/fem/level_set_lfem_example.py` :: from fealpy.mesh import, TriangleMesh

### `test/`
- `test/cem/test_eit_generator.py` :: from fealpy.mesh import, TriangleMesh
- `test/cem/test_em_fdtd_sim.py` :: from fealpy.mesh import
- `test/cem/test_mesher.py` :: from fealpy.mesh import, TetrahedronMesh
- `test/cem/text_yeemesher.py` :: from fealpy.mesh import
- `test/cgraph/test_square.py` :: TriangleMesh
- `test/csm/test_elbow_pipe_model.py` :: from fealpy.mesh import, TriangleMesh, TetrahedronMesh
- `test/fdm/test_convection_operator.py` :: from fealpy.mesh import
- `test/fdm/test_diffusion_operator.py` :: from fealpy.mesh import
- `test/fdm/test_dirichlet_bc.py` :: from fealpy.mesh import
- `test/fdm/test_laplace_operator.py` :: from fealpy.mesh import
- `test/fdm/test_reaction_operator.py` :: from fealpy.mesh import
- `test/fem/bilinear_form_data.py` :: TriangleMesh

### `tests/`
- `tests/mesh/conftest.py` :: from fealpy.mesh import, TriangleMesh
- `tests/mesh/test_vtk_writter.py` :: from fealpy.mesh import
- `tests/mesh/unit/test_template.py` :: from fealpy.mesh import, TriangleMesh
- `tests/mesh/unit/schema/test_pyramid_mesh.py` :: PyramidMesh

## 后续建议

1. 先把 `fealpy/mesher/` 与 `tests/mesh/` 中的旧类型依赖分批迁移，因为它们更接近网格模块验收路径。
2. 对 `app/`、`example/` 中大量科研/示例代码，先通过兼容层稳定运行，再逐步替换构造入口。
3. 不建议长期并行维护新旧两套 mesh；兼容层只作为过渡。