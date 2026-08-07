# Suanhai | Template | 任务 Agent 提示词模板

- **版本**：v0.4
- **状态**：草案
- **入库位置**：`suanhaios/templates/task/suanhai_task_agent_prompt_template.md`
- **启用条件**：当需要为某个具体 Task 提供直接执行协作输入时，本模板必须启用
- **适用范围**：SuanhaiOS 仓库；算海团队所有仓库中的 Task Agent Prompt 实例写作与修订场景

本模板对应文件 `suanhai_task_agent_prompt_template.md`，用于生成具体的 Task Agent Prompt 实例文档。该模板回答的问题是：面向执行协作的直接输入是什么。

## 一、当前目标
基于 fealpy 新 mesh 架构，实现一个最小三维四面体网格验证算例，完成：

1. 使用 meshio 读取 `.vtu` 文件
2. 构造四面体网格结构
3. 计算四面体单元半径比质量
4. 计算四面体单元二面角
5. 提供最小可运行验证流程


## 二、已知输入
- fealpy 新 mesh 模块：
  - Mesh
  - MeshBlock
  - EntitySector
  - TopologyBuilder

- 外部依赖：
  - meshio

- 输入网格：
  - `.vtu` 四面体网格文件

- 当前已实现功能：
  - `from_vtu`
  - `radius_ratio`
  - `dihedral_angle`
  - `face_area`
  - `cell_volume`
## 三、输出要求
需要输出：

1. 一个可直接运行的 Python 脚本
2. 能通过命令行：
   ```bash
   python test_mesh.py --filename xxx.vtu
## 四、禁止事项与自检要求
- 无