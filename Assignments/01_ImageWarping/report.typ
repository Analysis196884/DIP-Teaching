// Page Setup
#set page(width: 16cm, height: auto)
#set heading(numbering: "1.")

#set text(font: ("Noto Serif", "Noto Serif CJK SC"))

#set par(
  // first-line-indent: (amount: 2em, all: true),
  justify: true
)

// Title and Author
#set document(
  title: [Assignment 01: Image Warping],
)
#align(center)[#title()]
#align(center)[SA25001078]

= 任务概述
#h(2em)
本实验实现了基于控制点的图像变形算法，包括全局变换（仿射变换）和点导向变形（RBF 变形）两个部分。


= 算法介绍

== 全局仿射变换
#h(2em)
全局变换通过组合多个基本变换矩阵实现：
M = T @ R @ S @ F

其中：
- $S$ 为缩放矩阵
- $R$ 为旋转矩阵
- $T$ 为平移矩阵  
- $F$ 为翻转矩阵

== 基于 RBF 的图像变形
#h(2em)
使用基于 RBF 插值的方法进行图像变形。图像变形本质上是一个插值问题：找到一个光滑的变形函数 $f: bb(R)^2 -> bb(R)^2$，使得给定的一组控制点 $p_i$ 映射到对应的目标点 $q_i$。

在该算法中：
$ f(p) = A p + b + sum_(i=1)^n a_i phi_i (||p - p_i||) $

其中 $phi_i (r) = 1/(r^2 + d_i)$ 为径向基函数，$d_i$ 为参数，$n$ 为控制点的组数。

为了简便起见，在这里将仿射部分 $A p + b$ 设为了恒等映射 $p$：当 $p$ 远离控制点 $p_i$ 时，$f(p) approx p$，这使得这些像素近似保持不动。

根据约束条件 $f(p_j) = q_j$，有：
$ f(p_j) = p_j + sum_(i=1)^n a_i phi_i (||p_j - p_i||) = q_j, quad j = 0, 1, ..., n $

这里一共有 $2n$ 个方程，因此可以通过求解线性方程组来得到 $a_i$，进而得出插值函数 $f$。

= 实现细节
关键优化：

#h(2em)
用 numpy 的向量化操作替代 python 循环来计算所有像素到控制点的距离以及 RBF 值。numpy 底层为基于 C/C++ 的高性能库，充分利用了 SIMD 指令和多线程，能够显著提升计算效率。

= 实验结果
两个 RBF 变形的结果如下：
#image("pics/result.png")
#image("pics/result2.png")