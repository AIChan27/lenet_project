# 🔍 LeNet-5 手搓项目

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-1.24-green?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Status-手搓中......-brightgreen" alt="Status">
</div>

---

## 📖 项目简介
这是基于《深度学习入门：基于Python的理论与实现》学习后，**纯手搓（不使用PyTorch框架）** 的 LeNet-5 卷积神经网络实现。本项目用于深入理解卷积、池化、反向传播的底层数学原理。

> ⚠️ **声明1**：现阶段util.py和mnist.py直接照搬鱼书代码！

> ⚠️ **声明2**：仅供深度学习初学者学习底层原理使用。训练纯 CPU，可能会有点慢！

---

## 🎯 核心特性
- **纯 NumPy 实现**：不依赖任何现代深度学习框架，包括 `im2col` 和 `col2im` 展开操作。
- **数据流透明**：输入 `(N, 1, 28, 28)`，经过多层处理后输出 `(N, 10)`。
- **包含全套组件**：手写了 `Sigmoid`、`ReLU`、`Softmax`、`交叉熵`、`卷积`、`池化`、`全连接` 等类。

---

## 🧠 核心设定（数据与模型）

- **使用模型**：LeNet-5（经典卷积神经网络，1998年提出）。
- **数据集**：MNIST（手写数字识别），使用 `flatten=False` 加载（保持 4D 结构）。
- **数据初始大小**：
  - 输入形状：`(N, 1, 28, 28)`（即：100张图，1个灰度通道，28x28像素）。
  - 标签形状：`(N, 10)`（独热编码）。
- **训练批次（Batch Size）**：100
- **训练轮次（Epochs）**：10 - 20 个左右

---

## 🧱 LeNet-5 标准维度流（完整推导）

| 层名称 | 操作 | 输入维度 | 输出维度 | 参数/备注 |
| :--- | :--- | :--- | :--- | :--- |
| **C1** | 卷积 + ReLU | (N, 1, 28, 28) | (N, 6, 24, 24) | 6个 5x5，步长1，无填充 |
| **S2** | 池化 | (N, 6, 24, 24) | (N, 6, 12, 12) | 窗口 2x2，步长2 |
| **C3** | 卷积 + ReLU | (N, 6, 12, 12) | (N, 16, 8, 8) | 16个 5x5，步长1 |
| **S4** | 池化 | (N, 16, 8, 8) | (N, 16, 4, 4) | 窗口 2x2，步长2 |
| **F5** | 展平 + Affine + ReLU | (N, 256) | (N, 120) | 全连接 |
| **F6** | Affine + ReLU | (N, 120) | (N, 84) | 全连接 |
| **Out** | Affine + Softmax | (N, 84) | (N, 10) | 输出层 |

---

## 📂 目录结构

<pre>
lenet_project/
├── dataset/                  # dataset 文件夹（包含 mnist.py）
├── common/                   # common 文件夹（包含 layers.py, gradient.py, functions.py, util.py, optimizer.py）
├── pdf/                      # pdf 文件夹（存放此次项目参考论文或资料）
├── lenet5.py                 # 1. 这里写 LeNet-5 的网络类
├── trainer.py                # 2. 这里写 Trainer 类
├── save_and_load.py          # 3. 这里写 pkl 保存和加载的测试
└── main.py                   # 4. 主程序：把上面全部串起来
└── debugfromgradient.py      # 5. 测试程序，通过numerical_gradient找出梯度bug
</pre>