---
title: 网格数据结构基础
permalink: /docs/zh/mesh-base
key: docs-mesh-base-zh
---


$\qquad$ 在偏微分方程数值计算程序设计中， 网格是最核心的数据结构， 是下一步实现数值离散方法
的基础。FEALPy 中核心网格数据结构是用{\bf 数组}表示。
\begin{itemize}
    \item[$\bullet$] 三角形、四边形、四面体和六面体等网格，因为每个单元顶点的个
        数固定，因此可以用{\bf 节点坐标数组} node 和{\bf 单元拓扑数组} cell 来表
        示，这是一种以{\bf 单元为中心的数据结构}。
    \item[$\bullet$] 其它的如{\bf 边数组} edge、{\bf 面数组} face 都可由 cell 生
        成。
    \item[$\bullet$] FEALPy 中把 node、edge、face 和 cell 统称为网格中的实体 entity。
    \item[$\bullet$] 在一维情形下，edge，face 和 cell 的意义是相同的。
    \item[$\bullet$] 在二维情形下，edge 和 face 意义是相同的。
    \item[$\bullet$] FEALPy 中还有一种以{\bf 边中心的网格数据结构}， 称为{\bf 半边数
    据结构(Half-Edge data structure)}， 它具有更灵活和强大的网格表达能力。
