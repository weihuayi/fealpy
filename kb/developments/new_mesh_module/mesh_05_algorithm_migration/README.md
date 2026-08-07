# FEALPy | Mesh 05 Algorithm Migration - 任务导航

- **任务编号**：mesh_05_algorithm_migration
- **任务标题**：新网格模块算法实现迁移
- **版本**：v0.1
- **状态**：草案
- **创建时间**：2026-05-26
- **更新时间**：2026-05-26
- **前置任务**：mesh_03_optimize_architecture（接口与架构优化）
- **后续任务**：mesh_06_test_coverage（覆盖率补强）、mesh_07_integration_validation（系统查漏）

---

## 📋 快速导航

本任务文档体系包含以下 7 个核心文件 + 1 份协作交接说明：

1. **[Task Brief](mesh_05_algorithm_migration_task_brief.md)**  
   这个 Task 是什么、为什么成立、完成判据是什么。
2. **[Task Requirement](mesh_05_algorithm_migration_task_requirement.md)**  
   需要迁移哪些算法、必须满足哪些约束与成功条件。
3. **[Task Object Boundary](mesh_05_algorithm_migration_task_object_boundary.md)**  
   只迁移到 Entity Schema，不触及其他层的边界说明。
4. **[Task Network Design](mesh_05_algorithm_migration_task_network_design.md)**  
   多人协作下的结构拆解与依赖关系。
5. **[Task Target Asset Design](mesh_05_algorithm_migration_task_target_asset_design.md)**  
   预期形成的目标资产、命名与入库位置。
6. **[Task Validation Design](mesh_05_algorithm_migration_task_validation_design.md)**  
   迁移成果如何验证、验证子任务如何拆解。
7. **[Task Execution Workflow](mesh_05_algorithm_migration_task_execution_workflow.md)**  
   执行步骤与检查点，便于多人分工推进。
8. **[协作交接说明](mesh_05_algorithm_migration_team_handoff.md)**  
   新网格模块架构摘要、Entity Schema 接口标准、法向/切向定义准则与注意事项。

---

## 🎯 使用路径建议

- **任务负责人**：先读 Brief → Requirement → Network Design → Execution Workflow。
- **开发成员**：先读 Boundary → Handoff → Requirement → Execution Workflow。
- **验证/测试成员**：先读 Validation Design → Requirement → Handoff。

---

## 📚 相关参考

- 子任务 03 资产：`kb/developments/new_mesh_module/mesh_03_optimize_architecture/`
- Entity Schema 接口参考：`mesh_03_optimize_architecture/mesh_module_interface.md`
- 架构设计文档：`kb/design/mesh/mesh-data-structure.md`
