# $\text{SST}\,\, k-\omega$ 模型有限元求解程序设计文档

## 一、 概述

本设计严格遵循 FEALPy.CFD 的框架设计，程序框架拆分为 `Equation`（数学模型）、`Simulation`（离散算法）、`Math Model`（算例模型）和 `Computation Model`（计算模型）四个模块。通过这种解耦设计，可以确保物理边界、方程组装、代数求解与底层网格空间的完全独立，完美支撑后续流固耦合（FSI）优化任务的扩展。

---

## 二、 程序设计

### 文件结构
```

SST_k-omega_fem_solver/
├── equations/        # 控制方程模块
│   ├── stationary_incompressible_rans.py   # 稳态不可压缩 RANS 方程
│   ├── stationary_turbulent_kinetic_energy.py  # 稳态湍动能方程
│   └── stationary_specific_dissipation_rate.py  # 比耗散率方程
│
├── simulation/     #数值模拟方法目录
│   └── fem/            # 有限元
│   │   └── stationary_sst_k_omega/  # SST k-omega 模型
│   │   │   ├── stationary_incompressible_rans.py   # 稳态不可压缩 RANS 方程
│   │   │   ├── stationary_turbulent_kinetic_energy.py   # 稳态湍动能方程
│   │   │   └── stationary_specific_dissipation_rate.py   # 比耗散率方程
│
├── model/     #测试算例
│   └── stationary_sst_k_omega/
│   │  └── pipe_bend_turbulent_flow.py   # 90 度弯管湍流
│
└── stationary_incompressible_sst_k_omega_fem_model.py  # SST k-omega 计算模型
│

FEALPy/example/cfd
└── stationary_sst_k_omega.py   # SST k-omega 模型算例测试
│
```
### 模块设计

#### 1. Equation (数学模型)
**类名**： 
* `StationaryIncompressibleRANS`
* `StationaryTurbulentKineticEnergy`
* `StationarySpecificDissipationRate`

**职责**：
* 管理 $\text{SST} \,\,k-\omega$ 湍流模型各个方程的专属经验常数（如 $\beta^*$、$\sigma_k$ 等）。
* 根据当前流场状态，计算有效粘度（$\mu_{eff}$）以及湍流的生成项、耗散项和交叉扩散项。

#### 2. Simulation (离散算法)
**类名**：
* `StationaryIncompressibleRANSFEM`
* `StationaryTurbulentKineticEnergyFEM`
* `StationarySpecificDissipationRateFEM`

**职责**：
* **设定离散空间**：为各物理场（未知量）分配有限元空间。
* **提供组装组件**：拆分提供独立的刚度矩阵和右端项组装接口（分别对应 RANS方程、$k$ 方程、$\omega$ 方程）。
* **提供更新组件**：提供独立的局部系数更新接口（如更新涡粘系数 $\mu_t$ 场、更新 SUPG 稳定化参数）。

#### 3. Math Model (算例模型)
**类名**：`PipeBendTurbulentFlow`  
**职责**：
* **配置物理参数**：设定特定工况（如 Re=43000）下的流体密度与运动粘度。
* **定义边界解析式**：编写入口抛物面速度、入口 $k$ 与 $\omega$ 经验分布、出口零静压等边界的数学闭包函数。
* **封装基准算例**：将几何域标记与边界解析函数绑定，向后传递标准化的 PDE 问题对象。

#### 4. Computation Model (计算模型)
**类名**：`StationaryIncompressibleSSTkomegaFEMModel`  
**职责**：
* **管理网格与自由度**：加载弯管三维网格，完成全局自由度的分配与初始化。
* **编排求解控制流**：作为核心调度器，按顺序调用 `Simulation` 的组装与更新组件，驱动稳态分离式（Segregated）迭代求解。
* **收敛检验与后处理**：计算每步迭代残差以控制启停，并在收敛后调用导出接口输出 VTK 流场数据。
---

## 三、 Benchmark 算例验证支撑

针对雷诺数 **43000** 的 **90°** 弯管内部流动验证，这套架构的协同数据流如下：
1.  **Math Model** 加载 Benchmark 指定的粘度参数与入口抛物线映射公式。
2.  **Computation Model** 从底层 `fealpy.mesh` 读取高精度弯管网格，并分配所有的离散自由度。
3.  **Simulation** 启动分离式离散算法，高频调度 **Equation** 刷新矩阵进行非线性逼近。
4.  收敛后，**Simulation** 直接调用 `Computation Model` 的输出接口导出 VTK，供后处理工具提取 **75°** 截面数据与参考文献试验结果对比。