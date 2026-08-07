# 成熟项目自动化测试系统调研报告：面向 FEALPy 新网格模块

## 调研目标与任务边界

本报告服务于 FEALPy 新网格模块测试体系建设的前期调研阶段。目标不是立刻实现测试，也不是直接落地 CI，而是先观察成熟项目如何把“测试代码、测试数据、测试分层、失败反馈、CI 选择与本地复现”组织成一套可持续维护的工程系统，从而为后续的“最小测试矩阵”“测试用例生成”“测试系统设计”和“CI 运行策略”提供可执行输入。

本报告明确不做以下事情：不直接编写 FEALPy 新网格模块正式测试文件；不直接修改现有 CI；不把范围扩展到 FEALPy 其它模块；不以“覆盖率指标最大化”为目标；不把调研写成 pytest 或 GitHub Actions 工具教程。报告中所有建议都以“支持新网格模块 MVP 测试系统”为边界，优先考虑低成本、可回归、可定位、可本地复现。

从问题性质看，FEALPy 新网格模块最需要的不是“一次性多写点测试”，而是尽快建立一套小而稳的公共约束：哪些对象先测、哪些断言必须稳定、哪些输入必须固定、哪些失败信息必须保留、哪些测试能进 PR、哪些测试应当留到定时回归。成熟项目真正可借鉴的价值，恰恰在于这些工程边界如何被长期固化，而不只是它们“使用了 pytest”。这一点在 SciPy、scikit-learn 和 DOLFINx 的官方开发者文档、仓库测试目录与 CI workflow 中都体现得很清楚。[1](https://docs.scipy.org/doc/scipy-1.16.1/dev/contributor/devpy_test.html)

## 调研对象选择

| 项目 | 类型 | 选择理由 | 重点调研内容 |
|---|---|---|---|
| SciPy | Python 科学计算基础库 | 代表数值计算项目的典型测试工程化做法；官方开发者文档完整，仓库中根级 `conftest.py`、测试入口、慢测试与 array backend 测试组织都很成熟。[1](https://docs.scipy.org/doc/scipy-1.16.1/dev/contributor/devpy_test.html) | 测试分层、慢测试与 xslow、浮点断言、共享 fixture、array backend 参数化、CI 中 fail-slow 与 wheel/nightly 组织 |
| scikit-learn | 大型 Python 科学计算 / 机器学习项目 | 其测试体系兼顾“公共测试规范”和“大规模协作”，对参数化、共享 fixture、随机种子、非回归测试、贡献者工作流和 CI 分流策略尤其有参考价值。[2](https://scikit-learn.org/stable/developers/contributing.html) | 公共测试工具、`parametrize_with_checks`、全局随机种子策略、float32 测试、路径/场景选择、失败报告与本地复现 |
| DOLFINx | 有限元 / 网格 / 数值模拟项目 | 与 FEALPy 新网格模块的领域距离最近；其 Python 与 C++ 并行测试、MPI 语义、网格/几何/IO/演示算例回归、临时目录与并行清理策略具有直接借鉴意义。[3](https://github.com/FEniCS/dolfinx/tree/main/python/test/unit) | 网格对象测试组织、MPI/串行分层、测试数据目录、临时文件与并行安全、端到端 demo 回归、C++/Python 双层 CI |

这三个项目的组合是互补的。SciPy 解决“数值库如何把测试工程做稳”的问题；scikit-learn 解决“大型社区项目如何让贡献者可预测地写、跑、复现测试”的问题；DOLFINx 解决“网格与有限元对象如何在串行/MPI、数值/IO、单元/回归之间分层”的问题。对 FEALPy 新网格模块而言，这比单独调研某一个项目更有价值。[1](https://docs.scipy.org/doc/scipy-1.16.1/dev/contributor/devpy_test.html)

## 各项目测试系统实践分析

**SciPy**

**测试层级划分。** SciPy 的开发者文档把本地运行分成默认测试、按子模块/类/单测精确选择运行、带 coverage 的运行，以及包含 `slow` 测试的 “full” 模式；根级 `conftest.py` 又进一步区分了 `slow` 和 `xslow`，其中 `xslow` 默认跳过，只有设置 `SCIPY_XSLOW=1` 才运行。与此同时，CI 文档说明有 “fast” 与 “full” 两种时长阈值，并通过 `pytest-fail-slow` 约束普通测试与慢测试的执行时间。换句话说，SciPy 的分层不是单独维护 “unit/integration” 目录，而是在同一测试体系中用选择入口、标记和运行模式完成分层。对于 FEALPy，这种“同源测试代码 + 不同运行层级”的方式比一开始硬拆很多目录更实用。[1](https://docs.scipy.org/doc/scipy-1.16.1/dev/contributor/devpy_test.html)

**测试目录与命名组织。** SciPy 的文档鼓励按子模块、测试文件、测试类、测试函数精确运行测试，这说明其测试是与模块结构紧密绑定、按 `scipy.<module>.tests.<test_file>` 组织的，而不是把所有测试集中到单一大目录。根级 `conftest.py` 统一注册 markers、共享 autouse fixture、array backend fixture 与线程限制逻辑，这种“局部测试文件 + 全局测试基础设施”的组合，对于 FEALPy 新网格模块尤其值得借鉴。[1](https://docs.scipy.org/doc/scipy-1.16.1/dev/contributor/devpy_test.html)

**fixture、测试数据与可重复性。** SciPy 在根级 `conftest.py` 中做了几件很工程化的事情：一是通过 `check_fpu_mode` autouse fixture 检查测试是否改变 FPU 模式；二是通过线程池限制避免并行测试导致 OpenMP/BLAS 过度抢占；三是通过 `xp` fixture 把 NumPy、Torch、CuPy、JAX、Dask 等后端统一纳入一套测试入口，并用 backend-specific 的 skip/xfail marker 来约束可运行边界。文档对断言函数也给出明确建议：浮点比较优先用 `xp_assert_close`，整数数组用 `xp_assert_equal`。这套做法的底层思想非常适合 FEALPy：把“浮点容忍、共享前置条件、环境隔离、公共后端/输入变体”放到共享层，而不是散落在单个测试文件里。[4](https://github.com/scipy/scipy/raw/refs/heads/main/scipy/conftest.py)

**参数化测试实践。** SciPy 的代表性做法不是炫技式的大型 `parametrize`，而是把“参数化入口”包装成具名 fixture。`xp` fixture 通过 `pytest.param(..., id='numpy')` 等方式给每个 backend 明确命名，再结合 `array_api_backends`、`skip_xp_backends`、`xfail_xp_backends` 等 marker，使失败能精确定位到具体后端与具体约束条件。这种模式对 FEALPy 非常有启发：网格类型、维度、存储布局、边界条件不要直接塞进一层又一层匿名 `for` 循环，而应当给每个维度一个清晰名字和可选择性。[4](https://github.com/scipy/scipy/raw/refs/heads/main/scipy/conftest.py)

**回归测试策略。** SciPy 官方文档没有把“回归测试”单独做成一个独立目录或独立工具链，但从它支持按具体测试点精确运行、对 32 位平台、array backend、超慢测试分别设 marker、并在 CI 中对测试时长进行门控，可以推断其回归策略更偏向“把历史失败场景沉淀为精确、可单独调用的测试项”，而不是维护一个与普通测试分离的回归目录。这对 FEALPy 很重要：新网格模块的回归测试更适合围绕“具体 bug / 具体网格 / 具体拓扑关系 / 具体数值边界”沉淀成最小用例，而不是先建立一个庞大的“regression framework”。[1](https://docs.scipy.org/doc/scipy-1.16.1/dev/contributor/devpy_test.html)

**CI 组织方式。** SciPy 的 CI 文档表明其既使用 GitHub Actions，也有专门的文档与 benchmark 任务；并明确支持通过 commit message 跳过部分 CI，wheel 构建则由定时、手动、标签与特定 commit marker 触发。当前仓库的 `pixi.toml` 还把本地测试任务抽象成 `test`、`test-coverage` 和 `test-xslow`，分别对应默认测试、覆盖率测试和包含 `slow`/`xslow` 的完整测试。这种“开发者本地命令与 CI 运行模式一一对应”的做法，对 FEALPy 后续实现“一条命令复现 CI”非常有价值。[5](https://docs.scipy.org/doc/scipy-1.16.1/dev/contributor/continuous_integration.html)

**对 FEALPy 新网格模块的启发。** SciPy 最可迁移的不是它庞大的 CI，而是三件事：第一，用 marker 和公共 fixture 做分层，而不是一开始就设计复杂目录体系；第二，把浮点比较、环境保护、共享输入变体上提到公共层；第三，把“默认测试”和“完整回归/慢测试”拆成不同运行模式，但保持同一套测试代码源。[5](https://docs.scipy.org/doc/scipy-1.16.1/dev/contributor/continuous_integration.html)

**不适合当前阶段照搬的做法。** SciPy 面向多后端数组库、不同 BLAS/LAPACK 变体、wheel、GPU 与 nightly 生态，CI 与本地任务矩阵天然复杂。FEALPy 新网格模块当前若直接照搬这类矩阵，会把“先把网格语义测稳”变成“先把基础设施铺满”，投入产出比过低。尤其是 GPU/backends、多 BLAS 变体、周度 wheel / nightly 构建并不属于新网格模块 MVP 的第一优先级。[6](https://github.com/scipy/scipy/blob/main/.github/workflows/linux_blas.yml)

**scikit-learn**

**测试层级划分。** scikit-learn 把高质量单元测试明确写成开发流程基石，测试函数放在各子包的 `tests` 子目录中；同时在开发者文档和当前 workflow 中把普通单元测试、doctest、软依赖测试、文档构建、性能 benchmark 和 nightly 额外检查分开组织。它没有把所有层级都靠目录隔离，而是靠不同任务入口来完成：PR 主要跑与改动相关的测试，nightly/manual 则额外启用随机种子扰动、float32 测试和网络测试。对 FEALPy 来说，这说明“测试层级”本质上是运行策略，不一定等于目录数量。[2](https://scikit-learn.org/stable/developers/contributing.html)

**测试目录与命名组织。** 开发者文档明确说明：运行某个文件夹下的 `pytest` 会执行相应子包测试；PR checklist 则建议按源码文件、测试文件、整个子模块、相关 rst 文档四个粒度去运行。这种“源码—测试—文档示例”三位一体的组织非常成熟，也意味着单元测试与 doctest 并不是完全割裂的。对于 FEALPy 新网格模块，值得借鉴的是“测试与被测模块紧邻组织 + 文档示例作为补充层”，而不是文档示例取代正式测试。[2](https://scikit-learn.org/stable/developers/contributing.html)

**fixture、测试数据与可重复性。** scikit-learn 在 `sklearn/conftest.py` 和并行/配置文档中，把可重复性做得非常系统。它有 `global_dtype` fixture，通过 `SKLEARN_RUN_FLOAT32_TESTS=1` 把同一批测试自动扩展到 float32；有 `global_random_seed` fixture，通过 `SKLEARN_TESTS_GLOBAL_RANDOM_SEED` 统一控制种子，并明确要求使用该 fixture 的测试在 0 到 99 的任意种子下都应稳定通过；对需要网络下载的数据集，默认跳过，并只在指定环境变量开启时下载，同时在并行收集阶段统一处理下载，避免 xdist 线程不安全；`pyplot` fixture 则负责软依赖跳过与图像资源清理。换言之，scikit-learn 把“环境变量 + fixture + 收集期处理 + 软依赖跳过”合成了一套可重复性制度。[7](https://github.com/scikit-learn/scikit-learn/raw/refs/heads/main/sklearn/conftest.py)

**参数化测试实践。** scikit-learn 最有代表性的不是普通的 `@pytest.mark.parametrize`，而是 `parametrize_with_checks`：它把一组公共 API 兼容性检查参数化到不同 estimator 上，并把每个测试的 id 设成“estimator 的 pprint + check 名称 + 关键字参数”，从而支持用 `pytest -k` 精确筛选失败项。当前 `conftest.py` 又把 `global_random_seed` 与 `global_dtype` 的参数化放到生成阶段统一处理。对 FEALPy 而言，这意味着参数化的价值不仅是“少写重复代码”，更是“把失败显示成可读的测试名字”。[8](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html)

**回归测试策略。** scikit-learn 的 PR checklist 说得非常直接：对 bug fix，非回归测试在 PR 时应该在 `main` 上失败、在修复代码上通过。这是成熟项目中最清晰、也最适合迁移的一条制度性约束之一。它把“遇到 bug 就顺手修掉”升级成“遇到 bug 必须留下可长期执行的最小失败样例”。对 FEALPy 新网格模块来说，这应当成为后续测试建设的硬规则。[2](https://scikit-learn.org/stable/developers/contributing.html)

**CI 组织方式。** 当前 `unit-tests.yml` 非常值得细读：它在 `push`、`pull_request`、`schedule` 和 `workflow_dispatch` 下触发；主矩阵覆盖 Linux、macOS、Windows、ARM、32 位、最小依赖、最新依赖、MKL/OpenBLAS、SciPy dev 等场景；PR 默认不启用随机种子扰动和网络测试，而 nightly/manual 会随机抽取 `SKLEARN_TESTS_GLOBAL_RANDOM_SEED`、打开 float32 测试并恢复网络测试；workflow 还支持从 commit message 中提取选定测试集合，并在 nightly 失败时把 JUnit 文件回写到 tracking issue。覆盖率报告上传到 entity["company","Codecov","coverage platform"]，日志中显式打印“复现请设置的环境变量”。这是“快速 PR 测试”和“系统性回归探索”分离得非常清楚的做法。[9](https://github.com/scikit-learn/scikit-learn/actions/runs/24871091644/workflow)

**对 FEALPy 新网格模块的启发。** scikit-learn 给 FEALPy 的最大启发有四点。其一，回归测试必须与 bug 修复绑定。其二，随机种子和 dtype 扰动应当进入“定时回归”而不是一上来压到每个 PR。其三，参数化必须有清晰 id，否则日志会迅速失去可读性。其四，本地运行路径要和开发者平时修改文件的方式对应，支持按文件、按子模块、按单测精确运行。[2](https://scikit-learn.org/stable/developers/contributing.html)

**不适合当前阶段照搬的做法。** scikit-learn 的多平台、多依赖、32 位、ARM、free-threaded、SciPy dev、软依赖、文档、benchmark、tracking issue 全套机制，适合大型社区项目的长期维护，但对 FEALPy 新网格模块前期调研阶段明显过重。尤其是随机种子 0–99 合同、float32 全面扩展、网络数据管理、跨平台特殊依赖和 nightly 跟踪问题自动回写，这些都应排在“网格语义正确性已经稳定之后”。[9](https://github.com/scikit-learn/scikit-learn/actions/runs/24871091644/workflow)

**DOLFINx**

**测试层级划分。** DOLFINx 的分层比前两者更贴近 FEALPy 所关心的网格与数值对象。仓库中既有 `python/test/unit` 下的 Python 单元测试，又有 C++ `ctest` 单元测试，还在 CI 中显式运行 C++ demo regression tests、Python demo tests、Python unit tests 的串行与 MPI 版本，以及跨平台 smoke test。也就是说，它把“单元测试、并行环境测试、IO 测试、demo 级回归测试”明确视为不同层级。对于 FEALPy，新网格模块最值得借鉴的是：单元语义测试与 demo/算例级回归应当区分；而并行/MPI 语义如果未来需要，也应单独分层。[3](https://github.com/FEniCS/dolfinx/tree/main/python/test/unit)

**测试目录与命名组织。** `python/test/unit` 下按 `common`、`fem`、`geometry`、`graph`、`io`、`la`、`mesh`、`nls`、`refinement` 组织，根测试目录和 unit 目录各自有 `conftest.py`。这说明 DOLFINx 并没有用一个巨型共享测试文件，而是把“跨整个测试树共享的基础设施”与“仅 unit 层共享的 fixture”分开。这对 FEALPy 很关键：`tests/mesh` 层面应该有自己的共享夹具与数据，而不是直接依赖仓库全局的杂糅工具。[3](https://github.com/FEniCS/dolfinx/tree/main/python/test/unit)

**fixture、测试数据与可重复性。** DOLFINx 的顶层 `conftest.py` 非常有领域特色。它在每个测试 teardown 后主动 `gc.collect()` 并进行 `MPI.COMM_WORLD.Barrier()`，以便及时触发可能具有 collectiveness 语义的析构；提供 `datadir` fixture 定位共享测试数据目录；提供 MPI-safe 的 `tempdir` fixture，为每个测试函数实例、每个 xdist worker 乃至参数化实例生成唯一目录，并保留目录以便失败后人工检查。具体测试文件中则常用固定随机种子、`np.allclose(..., atol=1e-5)` 之类的容忍比较，以及构造最小网格对象后直接断言几何与拓扑性质。对 FEALPy 新网格模块来说，这几乎就是可直接迁移的模板：小型标准网格、稳定临时目录、失败可检查、浮点容忍断言、避免环境依赖。[10](https://github.com/FEniCS/dolfinx/raw/refs/heads/main/python/test/conftest.py)

**参数化测试实践。** DOLFINx 大量使用堆叠式参数化：`dim`、`simplex`、`dtype`、`cell_type`、`order` 一层层组合，同时把常见网格类型包装成 `parametrize_cell_types` 这样的可复用装饰器。`test_interpolation.py` 里还会在参数化前加 `skip_in_parallel` 标记，将仅适合串行验证的测试从 MPI 环境中排除。它的优点是覆盖范围大、复用度高；弱点是相较 scikit-learn，自定义参数 id 的使用不算充分，失败定位更多还是依赖 pytest 自动输出参数值。对 FEALPy 而言，应吸收其“参数化 + 复用装饰器”的优点，但最好补上更清晰的 `id=` 命名。[11](https://github.com/FEniCS/dolfinx/blob/main/python/test/unit/fem/test_interpolation.py)

**回归测试策略。** DOLFINx 的回归思路非常明确：除了 unit tests，它还把 demo 级别的 serial/MPI 运行放进 CI，并在 C++ 侧显式运行 regression tests。对网格/有限元项目来说，这种做法很有代表性，因为很多问题不会在纯 API 单测中暴露，而会在“网格 + 函数空间 + 边界条件 + IO/求解”的组合路径中出现。不过，DOLFINx 这种 demo regression 的价值，在于它建立在坚实的单元测试底座之上，而不是反过来。FEALPy 当前阶段可以学习其“保留少量端到端回归”的方向，但不应一开始就把大规模 demo 作为主测试层。[12](https://github.com/FEniCS/dolfinx/blob/main/.github/workflows/ccpp.yml)

**CI 组织方式。** DOLFINx 当前 workflow 清楚展示了它怎样分层运行测试。`conda.yml` 用 `schedule` 和 `workflow_dispatch` 做跨 OS、Python 版本、real/complex 标量类型的安装与 smoke test；`macos.yml` 和 `ccpp.yml` 则同时跑 C++ 单测、MPI 单测、Python serial/MPI 单测、demo tests，并在 `ctest` 中开启 `-V --output-on-failure`，在 pytest 中输出 `--durations` 统计。workflow 还会在运行前写入统一的 JIT 配置文件，以减少编译噪声。这对 FEALPy 的直接启发是：哪怕暂时不做复杂 CI，也应先把“串行快速测试”“少量 demo 回归”“失败时输出耗时与关键日志”分开。[13](https://github.com/FEniCS/dolfinx/blob/main/.github/workflows/conda.yml)

**对 FEALPy 新网格模块的启发。** DOLFINx 最值得 FEALPy 借鉴的是四件事：按领域对象拆测试树；把串行与并行语义明确区分；为测试数据目录和临时目录提供共享 fixture；保留一小层端到端回归，但不让它挤占核心单元测试的优先级。其临时目录与清理策略，对未来涉及网格导出、文件写入、中间产物检查的测试尤其有参考意义。[3](https://github.com/FEniCS/dolfinx/tree/main/python/test/unit)

**不适合当前阶段照搬的做法。** DOLFINx 依赖 C++/Python 双栈、MPI、PETSc、ADIOS2、SLEPc、平台差异与 JIT 配置，其 workflow 天生比纯 Python 网格模块重很多。FEALPy 新网格模块若在前期调研后直接照搬这类串行/MPI 双跑、C++ demo regression、跨平台 real/complex 安装矩阵，会明显超出当前任务边界。应当借鉴其“分层思想”和“目录/fixture 组织”，而不是照搬其依赖复杂度。[13](https://github.com/FEniCS/dolfinx/blob/main/.github/workflows/conda.yml)

## 横向对比总结

| 维度 | SciPy | scikit-learn | DOLFINx | 对 FEALPy 的启发 |
|---|---|---|---|---|
| 测试层级 | 默认 / full / `slow` / `xslow` 运行模式 | 单元测试 + doctest + 夜间扩展检查 | unit + demo regression + serial/MPI + C++/Python | 先做“同源测试代码 + 分层运行”，再扩展目录 |
| fixture 组织 | 根级 `conftest.py` 集中管理 marker、backend、线程与环境保护 | 根级 `conftest.py` 统一 dtype、seed、网络数据、绘图资源 | 顶层与 unit 层双 `conftest.py`，兼顾全局与局部共享 | `tests/mesh` 需要独立共享层，不要把逻辑散到单测里 |
| 测试数据管理 | 更强调程序生成与共享输入变体 | 网络数据默认跳过，fixture 负责下载与跳过 | `datadir` + `tempdir`，MPI-safe 文件目录管理 | 小型固定网格数据应版本化；文件型测试要统一目录策略 |
| 参数化测试 | backend fixture + marker 限定边界 | `parametrize_with_checks` + 明确 id | 维度/单元/阶次/类型堆叠参数化 | 参数化必须有清晰名字与切片能力 |
| 回归测试 | 偏向把具体问题沉淀为精确测试项 | 明确要求 bug fix 必须附非回归测试 | 保留 demo / regression 层 | 每个 bug 对应一个最小回归用例 |
| 慢测试 / 性能测试 | `slow`、`xslow`、benchmark 分离 | benchmark 与 nightly 分离 | demo / MPI / unit 分层 | MVP 阶段先不强上全量 benchmark |
| CI 触发 | 文档、wheel、nightly、手动与常规 CI 分流 | `push` / `pull_request` / `schedule` / `workflow_dispatch` 完整分流 | 常规 workflow + schedule/manual 的安装验证 | PR 快速测试与定时回归必须分离 |
| 失败报告 | fail-slow、coverage、本地精确命令 | JUnit、coverage、tracking issue、复现环境变量 | `--output-on-failure`、`--durations`、serial/MPI 分开 | 失败信息至少要包含测试参数、最小输入、断言依据 |
| 本地复现 | 支持按子模块/类/函数精确运行 | 支持按文件/模块/fixture 环境变量复现 | workflow 命令几乎可直接本地照跑 | 必须提供仓库级单命令入口 |

从横向比较看，三者共同点非常稳定：都把“共享 fixture、可选择运行层级、可重复输入、局部可精确复现”放在测试系统核心；差异则主要来自领域与项目规模。SciPy 更强调数值库与多后端；scikit-learn 更强调大社区协作和贡献者体验；DOLFINx 更强调网格/有限元对象与串行-MPI 语义。FEALPy 新网格模块不需要照搬三者的全部复杂度，但应该同时吸收这三个方向的“低成本高稳定性”做法。[5](https://docs.scipy.org/doc/scipy-1.16.1/dev/contributor/continuous_integration.html)

## 可迁移到 FEALPy 新网格模块的设计原则

**先建设最小稳定测试集**

- 成熟项目依据：SciPy 把默认测试与 `slow/xslow` 分开，scikit-learn 把 PR 与 nightly 的额外检查分开，DOLFINx 也把 smoke / unit / demo regression 分开运行。[5](https://docs.scipy.org/doc/scipy-1.16.1/dev/contributor/continuous_integration.html)
- 适用于 FEALPy 的原因：新网格模块当前最需要的是先把接口语义和核心拓扑几何关系测稳，而不是先做完大而全的矩阵。
- 在新网格模块中的落地方式：先定义一套 P0 测试集合，只覆盖最核心的网格构造、拓扑查询、几何量计算、边界识别。
- 后续对应任务：最小测试矩阵、CI MVP。

**单元测试、集成测试、回归测试要分层运行，而不必一开始分成过多目录**

- 成熟项目依据：三者都更强调“运行层级”而不是“目录数量”来区分测试层。[1](https://docs.scipy.org/doc/scipy-1.16.1/dev/contributor/devpy_test.html)
- 适用于 FEALPy 的原因：当前阶段若过早细分目录，会增加维护负担；但运行层级不分，会导致 PR 太慢、回归不稳定。
- 在新网格模块中的落地方式：先统一放在 `tests/mesh` 范围内，再用 marker 或任务入口区分 `quick`、`regression`、`slow`。
- 后续对应任务：测试系统设计、CI 运行策略。

**共享 fixture 要承载标准网格样例，而不是让每个测试文件各自造轮子**

- 成熟项目依据：SciPy 用根级 `conftest.py` 管 markers 和共享 fixture；scikit-learn 用 `conftest.py` 统一 dtype、seed、软依赖与数据拉取；DOLFINx 用 `conftest.py` 管 `datadir`、`tempdir` 与共享网格 fixture。[4](https://github.com/scipy/scipy/raw/refs/heads/main/scipy/conftest.py)
- 适用于 FEALPy 的原因：新网格模块会反复使用三角形、四边形、四面体、结构/非结构等标准网格样例。
- 在新网格模块中的落地方式：把“最小标准网格样例”放到共享 fixture 层，并要求所有测试优先复用。
- 后续对应任务：测试目录组织、fixture 设计。

**测试数据要小型化、固定化、版本化**

- 成熟项目依据：scikit-learn 默认跳过网络数据，并把数据下载放进受控 fixture；DOLFINx 用 `datadir` 管理共享测试数据、用 `tempdir` 管理中间产物。[7](https://github.com/scikit-learn/scikit-learn/raw/refs/heads/main/sklearn/conftest.py)
- 适用于 FEALPy 的原因：网格模块测试若依赖大型外部网格文件或随机生成输入，复现成本会迅速上升。
- 在新网格模块中的落地方式：优先使用手工构造的小型网格、固定几何配置和最小失败样例；确需文件时将其版本化纳入仓库。
- 后续对应任务：测试数据设计、回归样例沉淀。

**浮点断言必须显式写出容忍误差与断言语义**

- 成熟项目依据：SciPy 明确区分浮点与整数断言函数；DOLFINx 在插值与 IO 测试中普遍使用 `np.allclose(..., atol=...)`。[14](https://docs.scipy.org/doc/scipy/dev/contributor/writing_test_tips.html)
- 适用于 FEALPy 的原因：网格几何量、法向、面积/体积、重心等天然受浮点误差影响。
- 在新网格模块中的落地方式：整数拓扑关系用精确断言；几何实数统一写清 `atol/rtol` 及其依据。
- 后续对应任务：断言规范、测试用例生成。

**参数化可以扩大覆盖面，但必须给失败定位留名字**

- 成熟项目依据：scikit-learn 的 `parametrize_with_checks` 明确为每个检查生成可读 id；SciPy 的 backend fixture 也为参数值指定明确 id；DOLFINx 则展示了多维参数化在网格测试中的自然用法。[8](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html)
- 适用于 FEALPy 的原因：网格类型、维度、结构性、边界情形天然适合参数化。
- 在新网格模块中的落地方式：参数 id 至少包含网格类型、维度、接口名；避免匿名大循环。
- 后续对应任务：测试矩阵设计、失败报告设计。

**每个 bug 修复都应留下一个最小回归测试**

- 成熟项目依据：scikit-learn 明确要求 bug fix PR 在 `main` 上失败、在 PR 上通过的非回归测试。[2](https://scikit-learn.org/stable/developers/contributing.html)
- 适用于 FEALPy 的原因：网格模块中很多问题是边界网格、拓扑退化、几何极值导致的，一旦复现成本高，就必须固化成资产。
- 在新网格模块中的落地方式：每个 bug 记录“最小输入—预期行为—断言依据—关联修复”。
- 后续对应任务：回归测试策略、缺陷闭环机制。

**随机性测试不应该污染 PR 快速测试**

- 成熟项目依据：scikit-learn 把随机种子扰动放在 nightly/manual，不放在普通 PR；DOLFINx 则在需要时显式固定随机种子。[15](https://scikit-learn.org/stable/computing/parallelism.html)
- 适用于 FEALPy 的原因：PR 阶段的第一目标是快速、稳定、可解释，而不是探索所有随机路径。
- 在新网格模块中的落地方式：PR 只跑固定输入；后续定时任务再扩展到不同随机扰动或输入扰动。
- 后续对应任务：CI MVP、后续完整回归策略。

**CI 失败必须支持本地单命令复现**

- 成熟项目依据：SciPy 提供精确到单测的本地入口；scikit-learn workflow 在日志中打印复现所需环境变量；DOLFINx workflow 命令与本地命令高度一致。[1](https://docs.scipy.org/doc/scipy-1.16.1/dev/contributor/devpy_test.html)
- 适用于 FEALPy 的原因：如果开发者不能快速复现，自动化测试就只会变成“远程红灯”。
- 在新网格模块中的落地方式：为 `tests/mesh` 提供唯一规范入口，CI 与本地都调用同一命令层。
- 后续对应任务：测试系统设计、CI 运行策略。

**失败报告要保留最小复现输入、关键中间量与断言依据**

- 成熟项目依据：scikit-learn 输出 JUnit 与覆盖率，并在 nightly 追踪中回写失败信息；DOLFINx 在 `ctest`/pytest 中开启 `--output-on-failure` 与 `--durations`，其 `tempdir` 还允许保留失败产物。[9](https://github.com/scikit-learn/scikit-learn/actions/runs/24871091644/workflow)
- 适用于 FEALPy 的原因：网格问题往往不是“结果错了”这么简单，而是某个局部拓扑、某个单元、某条边界面错了。
- 在新网格模块中的落地方式：失败时打印网格类型、维度、核心索引、拓扑连接、关键几何量与容忍阈值。
- 后续对应任务：失败报告格式、测试辅助工具。

**AI 辅助测试只能生成草案，不能跳过人工语义复核**

- 成熟项目依据：GitHub Copilot 的官方负责使用文档明确指出，生成代码必须人工审查、验收并进一步验证；对自动修复建议同样强调必须结合 AI 限制进行编辑和测试。[16](https://docs.github.com/en/enterprise-cloud%40latest/copilot/responsible-use/copilot-code-completion)
- 适用于 FEALPy 的原因：网格模块测试的难点不是“写出 assert”，而是 assert 是否真的表达了几何/拓扑语义。
- 在新网格模块中的落地方式：允许 AI 产出测试思路、边界场景、矩阵草案和初始代码，但入库前必须经过“语义正确性、稳定性、是否依赖实现细节、是否可复现”的人工清单式审查。
- 后续对应任务：AI 辅助测试流程、测试资产治理。

## 不适合当前任务直接照搬的实践

| 实践 | 来源项目 | 不适合原因 | 将来是否可考虑 |
|---|---|---|---|
| 大规模多平台、多依赖、多架构 CI 矩阵 | scikit-learn | 当前目标是新网格模块语义测试 MVP，不是仓库级质量平台；过大矩阵会显著拖慢反馈与维护。 | 可在测试系统稳定后逐步引入 |
| `slow` / `xslow` / nightly / wheel / benchmark 全套分层同时上线 | SciPy | 运行层级设计值得借鉴，但同时落地全部层级会让系统过早复杂化。 | 可分阶段引入 |
| GPU / 多 array backend / 多 BLAS 变体矩阵 | SciPy | FEALPy 新网格模块当前核心是几何与拓扑语义，不是多后端兼容性。 | 远期可按需求引入 |
| 大规模 demo regression + C++/MPI 双栈验证 | DOLFINx | 依赖复杂、运行成本高，不适合当前前期建设阶段。 | 仅在未来确有并行/高性能需求时考虑 |
| 随机种子全范围合同测试 | scikit-learn | 成本高，且要求测试设计高度成熟；当前更适合先用固定输入建立稳定基线。 | 可在 nightly 阶段引入 |
| float32 全量扩展测试 | scikit-learn | 需要所有断言与数值容差都先稳定，否则会放大噪声。 | 可在核心几何量测试稳定后引入 |
| 自动回写 tracking issue / 复杂失败汇总自动化 | scikit-learn | 前期维护收益不高，反而增加工具链复杂度。 | 中后期可考虑 |
| 完整 benchmark 体系与性能门禁 | SciPy、scikit-learn | 新网格模块当前的首要目标是正确性与回归稳定，不是性能基线治理。 | 在接口稳定后建立 |
| 依赖网络数据或大型外部测试资源 | scikit-learn 反面经验 | 会直接损害可复现性和 CI 稳定性。 | 原则上仍应慎用 |
| 以高覆盖率硬门禁替代测试设计质量 | scikit-learn 经验的反向边界 | 覆盖率有价值，但对网格模块来说，不如“是否覆盖关键语义”重要。 | 可做辅助指标，不宜先做硬门禁 |

以上“不宜照搬”的共同点是：它们都建立在项目已有比较稳的测试底座之上。FEALPy 新网格模块当前更需要的是先做“低复杂度、强可复现、能定位”的核心测试系统，再考虑把矩阵和自动化层层加厚。[5](https://docs.scipy.org/doc/scipy-1.16.1/dev/contributor/continuous_integration.html)

## 面向后续阶段的输入建议

**对最小测试矩阵的建议**

后续最小测试矩阵至少应包含以下维度：测试对象、测试层级、输入数据、预期行为、断言方式、优先级、是否进入 PR 快速测试、是否属于回归测试。对 FEALPy 新网格模块而言，建议优先把对象维度限定在“代表性网格类型 + 代表性接口”上，而不是一次性穷尽所有 API。对象维度可先覆盖三角形网格、四边形网格、四面体网格、结构网格、非结构网格；行为维度可先覆盖构造、几何量、拓扑映射、边界识别、实体索引一致性；断言维度则明确分为“精确整数断言”和“容差浮点断言”。这样的矩阵更接近 scikit-learn 的“公共检查合同”和 DOLFINx 的“网格对象 + 维度/类型参数化”组合。[8](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html)

建议在矩阵中显式增加两个治理字段。一个是“最小复现输入”，要求每条测试项都能回溯到一个小型标准网格或最小失败样例；另一个是“触发层级”，要求每条测试项标明属于 PR 快速测试、手动完整回归、还是未来定时任务。这样做的目的，是避免后续测试系统实现时再次围绕“这条测试应该在哪儿跑”反复争论。[5](https://docs.scipy.org/doc/scipy-1.16.1/dev/contributor/continuous_integration.html)

**对测试目录组织的建议**

建议后续把新网格模块测试限定在 `tests/mesh` 一棵子树中，先不要扩展到 FEALPy 其它模块。该子树内部可以按“公共夹具 / 单元语义 / 集成语义 / 回归样例 / 数据目录”做轻量组织，但不建议前期再拆出大量二三级目录。更重要的是在 `tests/mesh` 层形成自己的共享 `conftest.py` 和测试数据目录，借鉴 DOLFINx 的“双层共享”与 SciPy/scikit-learn 的“根共享逻辑上提”思路。[3](https://github.com/FEniCS/dolfinx/tree/main/python/test/unit)

在命名上，建议测试文件名直接表达被测职责，例如“构造”“几何量”“拓扑关系”“边界实体”“IO/序列化”。不要把一个文件写成网格模块的“总测试场”。成熟项目的共同经验表明，文件颗粒度适中、可按单文件精确运行，远比“测试集中在一个超大文件里”更有利于本地复现和 PR 审查。[1](https://docs.scipy.org/doc/scipy-1.16.1/dev/contributor/devpy_test.html)

**对测试数据和 fixture 的建议**

建议把测试输入分成三类。第一类是标准固定数据：例如最小三角形网格、最小四边形网格、最小四面体网格、一个带边界标签的简单几何域，它们应由共享 fixture 直接提供。第二类是临时文件数据：仅在测试 IO、导出、缓存等接口时生成，统一由共享临时目录 fixture 管理。第三类是随机数据：仅用于未来的鲁棒性或扰动测试，当前阶段不应进入 PR 默认路径。这样的分法与 scikit-learn 对网络/随机/float32 的分层控制，以及 DOLFINx 对 `datadir`/`tempdir` 的分离，非常一致。[15](https://scikit-learn.org/stable/computing/parallelism.html)

fixture 层面，建议至少预留四类公共夹具：标准网格对象夹具、标准几何/边界样例夹具、临时目录夹具、公共断言辅助函数。特别是公共断言辅助函数，应把“拓扑精确断言”“几何量近似断言”“失败时附加关键中间量打印”集中管理，避免每个测试文件自行发明断言风格。SciPy 对断言函数选择和 scikit-learn 对全局 fixture 合同的做法都支持这一做法。[14](https://docs.scipy.org/doc/scipy/dev/contributor/writing_test_tips.html)

**对 CI MVP 的建议**

MVP 阶段建议只做最小必要的三种触发：`pull_request` 用于快速回归；`push` 用于主分支或指定开发分支保护；`workflow_dispatch` 用于手工完整回归。`schedule` 可以作为后续阶段再加入，而不是当前必须。这个建议来自 scikit-learn 与 DOLFINx 的共同经验：定时回归很有价值，但前提是你已经有一套清晰的“默认 PR 测试集”。[9](https://github.com/scikit-learn/scikit-learn/actions/runs/24871091644/workflow)

运行内容上，MVP 阶段应只跑新网格模块最小稳定测试集，并且最好按路径、标签或专用任务入口限制范围，不要让其它模块测试混入。失败报告至少应输出：测试名、网格类型、测试参数、关键断言差值、关键中间量、失败产物目录位置。是否接入覆盖率报告可以做，但不建议把覆盖率门禁作为 MVP 阶段的主要目标。更重要的是保证“本地一条命令复现 CI”。这一点是 SciPy、scikit-learn、DOLFINx 三者共同强调的。[1](https://docs.scipy.org/doc/scipy-1.16.1/dev/contributor/devpy_test.html)

**对 AI 辅助测试生成的建议**

AI 在本任务中的合理位置，是“前置辅助”而不是“最终作者”。它适合用于生成测试思路、边界场景清单、参数矩阵草案、最小样例候选、重复样板的初稿，也适合帮助补全“这个接口还有哪些等价类没有覆盖”。但它不适合在无人复核的情况下直接入库，因为网格模块测试最容易出错的正是语义层：断言是否真的对应单元拓扑、边界关系、几何量定义，AI 往往会生成看上去合理、但语义上不稳的测试。官方负责使用文档对 AI 代码建议也明确要求“始终人工审查，并在接受后继续验证和测试”。[16](https://docs.github.com/en/enterprise-cloud%40latest/copilot/responsible-use/copilot-code-completion)

建议后续建立一个轻量的 AI 产物进入流程：先把 AI 生成结果归档为“候选测试草案”，再由人工按四项清单复核——断言是否符合 FEALPy 新网格模块语义、测试是否稳定、是否依赖当前实现细节、是否与真实 API 一致。只有通过复核的内容才转为正式测试资产；同时在 PR 描述或知识库记录中保留“AI 参与过哪些环节、人工删改了什么、为什么最终保留”。这样做不会把 AI 神秘化，而是把它纳入和普通代码审查一样的工程流程。[16](https://docs.github.com/en/enterprise-cloud%40latest/copilot/responsible-use/copilot-code-completion)

## 结论

综合三类成熟项目的证据，FEALPy 新网格模块当前最应该借鉴的，不是“大项目有多复杂”，而是它们怎样把复杂性约束在可维护的边界内。最值得立刻迁移的实践包括：先建立最小稳定测试集；用共享 fixture 管理标准网格样例和临时目录；把单元测试、集成测试、回归测试作为不同运行层级来组织；参数化时强制提供清晰的失败定位；对浮点结果统一使用显式容忍误差；每个 bug 修复必须沉淀一个最小回归测试；CI 失败必须支持本地单命令复现。[5](https://docs.scipy.org/doc/scipy-1.16.1/dev/contributor/continuous_integration.html)

当前阶段最应该避免的是过度设计。包括：一上来就搭建过大的 CI 矩阵、过早引入完整 benchmark 体系、把 demo 级大算例作为主测试层、为覆盖率设置高门槛硬门禁、让随机性和扩展 dtype 测试进入所有 PR、以及把 AI 生成测试直接当成最终资产。这些做法本身并不坏，但它们属于“测试系统已经稳定以后再扩展”的事情，不属于新网格模块前期调研之后的第一波落地内容。[9](https://github.com/scikit-learn/scikit-learn/actions/runs/24871091644/workflow)

如果把本报告转化为后续设计动作，最直接的落点就是四件事：先定义最小测试矩阵；再定义 `tests/mesh` 的共享 fixture 与数据边界；随后约定回归测试的沉淀规则；最后为 CI MVP 选定一条统一、本地可复现的测试入口。只要这四件事做扎实，FEALPy 新网格模块后续无论扩展到更多网格类型、更多接口还是更复杂的 CI，都能在一个稳定底座上迭代，而不是在不断返工的测试体系上堆复杂度。

**开放问题与局限**

本报告优先使用了官方开发者文档、官方仓库测试目录、`conftest.py`、当前 workflow 和官方负责使用文档；但并未穷尽每个项目的所有测试文件，因此对“所有历史 bug 如何沉淀”为回归测试的细粒度做法，更多是基于代表性制度与仓库结构所做的高置信度归纳，而非逐条 issue 级盘点。SciPy 当前仓库已经有 `pixi`/`spin` 任务，而开发者文档仍以 `dev.py` 为主入口，说明其本地开发入口处于演进中；本报告据此提炼的是“统一任务入口与 CI 模式对齐”的原则，而不是推荐 FEALPy 绑定某个具体工具。DOLFINx 的结论则更依赖仓库与 workflow 检查，而不是单独的测试指南页面，因此其“工程实践强、叙述文档相对分散”的特点需要在后续落地时加以留意。[1](https://docs.scipy.org/doc/scipy-1.16.1/dev/contributor/devpy_test.html)