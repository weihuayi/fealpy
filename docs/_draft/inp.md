# Abaqus 模型定义

inp 格式的文件

* keyword lines 和 data lines 
* model data 和 history data


## 输入文件

1. *Heading 
1. 模型数据
    + nodes
    + elements
    + materials
    + initial conditions
    +
1. part, assembly, instance
1. 历史数据: 第一个 step 定义之前的是模型数据，以后的都是历史数据
    + analysis type
    + loading
    + output requests

## 模型数据

定义一个有限元模型

### 必需的数据

1. 几何(Geometry)
    + nodes
    + elements
1. 材料（Material)

### 可选的数据

1. Parts and assembly
1. Initial conditions
1. Boundary conditions
1. Kinematic constraints (动力学约束）
1. Interactions: 定义部件之间的接触和相互作用
1. Amplitude definitions: 振幅定义，用于指定后续时间依赖的载荷或边界条件
1. Output control：输出控制
1. Environment properties: 用于定义环境属性，比如模型周围的流体
1. Analysis continuation: 分析沿续
  

## 历史数据(history data)

分析的目的是预测一个模型对于某种形式的外部载荷或非稳态的初始条件的反应。

一个 Abaqus 的分析是基于 steps 的概念， 一个分析中可以定义多个 steps。

* general response analysis steps, which can be linear or nonlinear
* linear perturbation steps


### 必需的历史数据

反应类型

* linear
* nonlinear
* static
* dynamic

### 可选的历史数据

* Loading
* Boundary conditions
* Output control
* Contact
* Auxiliary controls

### 可选的数据




## 语法规则

* keyword lines: options and often have parameters
* data lines
* comment lines
