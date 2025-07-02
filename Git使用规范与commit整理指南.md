# FEALPy 项目 Git 使用规范与 Commit 整理指南

> 本文档旨在规范 FEALPy 项目的 Git 使用流程，提升团队协作效率，确保代码历史清晰
> 可追溯，促进高质量科研软件开发。

---

## 一、分支命名规范

| 分支类型      | 命名格式                             | 示例 |
|---------------|--------------------------------------|------|
| 主分支        | `main`                               | `main` |
| 开发分支      | `develop`                            | `develop` |
| 功能开发      | `feature/<模块名>-<功能描述>`         | `feature/mesh-refactor` |
| 重构优化      | `refactor/<模块名>-<目的>`            | `refactor/functionspace-cleanup` |
| 发布准备      | `release/<版本号>`                    | `release/1.5.0` |
| 热修复        | `hotfix/<版本号>-<问题描述>`          | `hotfix/1.5.1-fix-crash` |
| 草稿探索      | `draft/<功能名或作者缩写>`            | `draft/spectral-diff-wy` |

---

## 二、Commit 编写规范

### 1. 语义化提交格式

使用如下格式书写提交信息：

```
<类型>: <简明描述>

[可选] 更详细的说明或变更背景。
[可选] 相关 issue、PR 链接。
```

### 2. 常见类型类型列表

| 类型      | 说明 |
|-----------|------|
| `feat`    | 新增功能 |
| `fix`     | 修复 bug |
| `refactor`| 重构（不涉及功能变更） |
| `perf`    | 性能优化 |
| `test`    | 增加或调整测试用例 |
| `docs`    | 文档更新 |
| `style`   | 格式调整（如空格、缩进等） |
| `chore`   | 构建脚本、工具配置等杂项 |
| `WIP`     | Work In Progress，用于临时同步（需后续整理） |

### 3. 示例

```bash
git commit -m "feat: add SDM basis function construction"

git commit -m "fix: correct indexing bug in cell_to_dof()"

git commit -m "WIP: early stage of LBM test case setup"
```

---

## 三、碎片化开发的推荐做法

### 1. 使用 WIP 提交保存进度

```bash
git commit -am "WIP: partial progress on optimization backend"
```

### 2. 完成功能后整理提交历史

使用交互式 rebase 合并整理：

```bash
git rebase -i develop
```

将多个碎片化的提交合并为语义清晰的一次性提交。

---

## 四、开发流程建议

```text
               +--------------+
               |  main        | <-- 稳定发布分支
               +--------------+
                      ^
                      |
               +--------------+
               |  develop     | <-- 日常开发集成分支
               +--------------+
                    ^
      +-------------+-------------+
      |             |             |
+------------+ +------------+ +------------+
| feature/*  | | refactor/* | | draft/*    | <-- 功能/重构/实验分支
+------------+ +------------+ +------------+
```

- 所有开发应基于 `develop` 创建分支；
- 完成后合并回 `develop`，PR 前整理 commit；
- 发布版本从 `develop` 分出 `release/*`；
- 紧急修复基于 `main` 创建 `hotfix/*`。

---

## 五、PR 提交前检查清单

- [ ] 是否基于 `develop` 创建分支？
- [ ] 是否完成 `git rebase -i` 整理 commit？
- [ ] 是否包含必要的单元测试或文档？
- [ ] 是否通过了 CI 自动测试？
- [ ] 是否撰写清晰的 PR 描述和变更说明？

---

## 六、附录：常用 Git 命令速查

### 合并多个提交（squash）
```bash
git rebase -i HEAD~5
# 将后续提交标记为 squash 或 fixup
```

### 修改最近一次 commit 信息
```bash
git commit --amend
```

### 暂存当前修改（不 commit）
```bash
git stash
# 恢复
git stash pop
```

### 强制推送整理后的提交
```bash
git push --force
```

---

## 七、附加建议

- **推荐工具**：使用 Fork / GitKraken / GitHub Desktop 可更方便管理分支与 rebase；
- **文档收录**：本指南应包含于 FEALPy 的开发文档中（如 `docs/dev/git_guide.md`）；
- **自动检查**：可通过 pre-commit hook 强制检查 commit 信息风格。

---

> FEALPy 是一个高质量科研基础设施项目。我们通过规范的协作流程与干净的历史，致力于打造一个可持续发展、可复用、可学习的计算平台。

