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
  title: [Assignment 02: Image Processing with PyTorch]
)
#align(center)[#title()]
#align(center)[SA25001078]

#let st = "s.t."
#let quad = h(1em)

= 任务概述
本实验包含两个任务：
- 使用 PyTorch 实现 Poisson 图像融合。
- 使用 PyTorch 实现 Pix2Pix 图像翻译。

= Poisson 图像融合
==  算法介绍
设前景图像表示为 $f$，背景图像表示为 $b$，融合区域为 $Omega$，边界为 $partial Omega$。 Poisson 图像融合的目标是求解以下优化问题：

$ min_u integral_Omega norm(nabla u - nabla f)^2 "d" x, quad st u|_(partial Omega) = b|_(partial Omega). $

通过引入拉普拉斯算子 $Delta$，上述问题可以转化为求解以下线性方程组：$ Delta u = Delta f, quad u|_(partial Omega) = b|_(partial Omega). $
这里不进行显式的线性方程组构造与求解，而是通过定义损失函数 $L(u) = 1/(|Omega|)integral_Omega (Delta u - Delta f)^2 "d" x$，并使用 PyTorch 的自动微分功能进行迭代优化求解。

== 实现细节
- *计算区域:* 初始化时对 mask 区域计算包围盒，后续只对包围盒内的像素计算 Laplacian，减少不必要的计算。
- *Laplacian 算子:* 构建 $3 times 3$ 的卷积核 $[[0, 1, 0]; [1, -4, 1]; [0, 1, 0]]$，通过 `torch.nn.functional.conv2d` 进行通道独立卷积。
- *优化配置:* 使用 Adam 优化器（$lr=0.01$），迭代次数为 5000 次，并引入学习率衰减策略。

== 实验结果
Poisson 图像融合的结果如图所示：
#figure(
  image("figs/poisson-result.webp"),
  caption: "Poisson 图像融合结果，将鲨鱼融合到背景图中"
)

= Pix2Pix 图像翻译

== 算法介绍
Pix2Pix 是一种基于 cGAN (Conditional Generative Adversarial Network) 的图像翻译框架。其目标函数由 GAN Loss 和 L1 Loss 组成：
$ L_(c G A N)(G, D) = E_(x,y) [log D(x, y)] + E_(x,z) [log(1 - D(x, G(x, z)))] $
$ L_(L 1)(G) = E_(x,y,z) [norm(y - G(x, z))_L_1] $
其中 $x$ 表示输入图像（语义图），$y$ 表示目标图像（立面图），$z$ 表示生成器的随机噪声输入。最终生成器的损失函数为：
$ L_G = L_(c G A N)(G, D) + lambda L_(L 1)(G) $
其中 $lambda$ 是权衡 GAN Loss 和 L1 Loss 比例的超参数，在本实验设置为 100。

== 实现细节
- *网络结构:* 

    - *Generator:* 采用 U-Net 结构，包含 4 层下采样与上采样对称结构。通道数序列为 32-64-128-256。编码器与解码器之间通过 Skip Connections 连接，以融合底层位置信息与深层语义信息。
    - *Discriminator:* 采用 PatchGAN 结构（70x70），通道数为 32-64-128-256。通过对图像局部块进行真伪判定，能够更有效地引导生成器产生高频纹理细节。
- *数据增强:* 对训练数据进行随机水平翻转和随机裁剪，以增强模型的泛化能力。

== 实验结果
针对 facades 数据集 (400 张图片) 在本机 (NVIDIA RTX 5070 LAPTOP) 上训练 200 个 epoch，使用 batch_size = 4, 耗时约 12 分钟。
结果如图所示：
#figure(
  grid(
    image("figs/pix2pix-result1.png"),
    image("figs/pix2pix-result2.png"),
    image("figs/pix2pix-result3.png"),
    image("figs/pix2pix-result4.png"),
    image("figs/pix2pix-result5.png")
  ),
  caption: "Pix2Pix 图像翻译结果，左边为输入的语义图，中间为对应的真实立面图，右边为生成器模型生成的立面图"
)