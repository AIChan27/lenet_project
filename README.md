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

## 踩坑指南
首先程序完成后，运行main.py，得到了不理想的结果：

<pre>
Epoch 1/10 | 训练集: 0.1136 | 验证集: 0.1064
Epoch 2/10 | 训练集: 0.1035 | 验证集: 0.1090
Epoch 3/10 | 训练集: 0.1035 | 验证集: 0.1090
Epoch 4/10 | 训练集: 0.1035 | 验证集: 0.1090
Epoch 5/10 | 训练集: 0.1035 | 验证集: 0.1090
Epoch 6/10 | 训练集: 0.1035 | 验证集: 0.1090
验证集连续多轮未提升，触发早停！
训练结束！已恢复验证集最高准确率（0.1090）时的模型参数。
✅ 模型参数已成功保存到 lenet_params.pkl！
最终测试结果为：0.1028
</pre>

程序发生了梯度消失/死区，准确率稳定在 0.1035（10.35%），验证集稳定在 0.1090（10.90%）。
MNIST 是 10 分类问题，如果瞎猜，准确率就是 10%。这说明模型目前完全没有在学，只是在“均匀乱猜”。

首先，由于 Loss 完全没有下降，所以我怀疑 layers.py 或 lenet5.py 里可能有隐藏的梯度计算 Bug。
于是我执行了debugfromgradient.py，使用里面的numerical_gradient（数值梯度）来对证，得到如下结果：

<pre>
W1 的最大误差: 6.32092435097195e-07
b1 的最大误差: 6.916253897219903e-06
W2 的最大误差: 1.24730879076222e-07
b2 的最大误差: 1.637048975216544e-05
W3 的最大误差: 3.73632748345295e-08
b3 的最大误差: 0.0002379821193276262
W4 的最大误差: 7.375460396465258e-12
b4 的最大误差: 0.0053471366801005615
W5 的最大误差: 9.962516039926472e-12
b5 的最大误差: 1.5012118034785082e-07
</pre>

显然误差处于正常范围内！通过查看论文以及与DeepSeek老师交流，了解到原版 LeNet-5 用的是 Sigmoid，而我使用了 ReLU。
根据之前对鱼书的学习，了解到当激活函数使用ReLU时，一般推荐使用ReLU专用的初始值，也就是Kaiming He等人推荐的初始值，也称为“He初始值”

原代码：
<pre>
# C1	卷积 + ReLU	(N, 1, 28, 28)	(N, 6, 24, 24)	6个 5x5，步长1，无填充
self.params["W1"] = weight_init_std * np.random.randn(6, input_dim[0], 5, 5)
self.params["b1"] = np.zeros(6)
# C3	卷积 + ReLU	(N, 6, 12, 12)	(N, 16, 8, 8)	16个 5x5，步长1
self.params["W2"] = weight_init_std * np.random.randn(16, 6, 5, 5)
self.params["b2"] = np.zeros(16)
# F5	展平+Affine + ReLU	(N, 256)	(N, 120)	全连接
# 为什么是 256 ？因为输入是 16 * 4 * 4 = 256
self.params["W3"] = weight_init_std * np.random.randn(256, hidden_size_1)
self.params["b3"] = np.zeros(hidden_size_1)
# F6	Affine + ReLU	(N, 120)	(N, 84)	全连接
self.params["W4"] = weight_init_std * np.random.randn(
    hidden_size_1, hidden_size_2
)
self.params["b4"] = np.zeros(hidden_size_2)
# Out	Affine + Softmax	(N, 84)	(N, 10)	输出层
self.params["W5"] = weight_init_std * np.random.randn(
    hidden_size_2, output_size
)
self.params["b5"] = np.zeros(output_size)
</pre>

使用“He初始值”之后的代码：
<pre>
# C1 卷积层使用 He 初始化（输入通道 * 卷积核大小 = node_num_1）
node_num_1 = input_dim[0] * 5 * 5  # 第一层：1 * 5 * 5 = 25
self.params["W1"] = np.random.randn(6, input_dim[0], 5, 5) *math.sqrt(2) / np.sqrt(node_num_1)
self.params["b1"] = np.zeros(6)
# C3 卷积层使用 He 初始化（输入通道 * 卷积核大小 = node_num_2）
node_num_2 = 6 * 5 * 5  # 第二层：6 * 5 * 5 = 150
self.params["W2"] = np.random.randn(16, 6, 5, 5) *math.sqrt(2) / np.sqrt(node_num_2)
self.params["b2"] = np.zeros(16)
# F5 展平+Affine + ReLU 全连接层使用 He 初始化（node_num = 16 * 4 * 4 = 256）
self.params["W3"] = np.random.randn(256, hidden_size_1) *math.sqrt(2) / np.sqrt(256)
self.params["b3"] = np.zeros(hidden_size_1)
# F6 Affine + ReLU 全连接层（120 -> 84）
self.params["W4"] = np.random.randn(hidden_size_1, hidden_size_2) *math.sqrt(2) / np.sqrt(hidden_size_1)
self.params["b4"] = np.zeros(hidden_size_2)
# Out Affine + Softmax 全连接层（84 -> 10）
self.params["W5"] = np.random.randn(hidden_size_2, output_size) *math.sqrt(2) / np.sqrt(hidden_size_2)
self.params["b5"] = np.zeros(output_size)
</pre>

程序运行后出现了新的问题：
<pre>
Epoch 1/10 | 训练集: 0.2204 | 验证集: 0.2301
Epoch 2/10 | 训练集: 0.2477 | 验证集: 0.2610
Epoch 3/10 | 训练集: 0.2830 | 验证集: 0.2954
Epoch 4/10 | 训练集: 0.3270 | 验证集: 0.3360
Epoch 5/10 | 训练集: 0.3673 | 验证集: 0.3800
Epoch 6/10 | 训练集: 0.4077 | 验证集: 0.4204
验证集连续多轮未提升，触发早停！
训练结束！已恢复验证集最高准确率（0.4204）时的模型参数。
✅ 模型参数已成功保存到 lenet_params.pkl！
最终测试结果为：0.4252
</pre>

显然判断早停的函数出现了逻辑错误，遂对早停代码进行修改：