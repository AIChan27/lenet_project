# 🔍 LeNet-5 手搓项目

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-1.24-green?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Status-第一阶段完成！-brightgreen" alt="Status">
</div>

---

[![📖 项目简介](https://img.shields.io/badge/📖-项目简介-blue)](README.md)
[![🚧 踩坑指南](https://img.shields.io/badge/🚧-踩坑指南-orange)](TROUBLESHOOTING.md)

---

## 📖 项目简介
这是基于《深度学习入门：基于Python的理论与实现》学习后，**纯手搓（不使用PyTorch框架）** 的 LeNet-5 卷积神经网络实现。本项目用于深入理解卷积、池化、反向传播的底层数学原理。

> ⚠️ **声明1**：现阶段util.py和mnist.py直接照搬鱼书代码！

> ⚠️ **声明2**：仅供深度学习初学者学习底层原理使用。训练纯 CPU，可能会有点慢！

---

## 🗺️ 项目阶段

### ✅ 第一阶段：初步跑通模型
- **训练配置**：`epochs=10`, `mini_batch_size=100`, `optimizer="Adam"`, `lr=0.001`（修正了此前的死区问题，改用 `He 初始化` 搭配 `ReLU`）。

- **任务进度**：第一阶段任务完成。

**📈 第一阶段训练配置与运行结果：**
训练配置：
<pre>
trainer = Trainer(
    network,
    x_train,
    t_train,
    x_val,
    t_val,
    epochs=10,
    mini_batch_size=100,
    optimizer="Adam",
    optimizer_param={"lr": 0.001},
)
</pre>
运行结果：
<pre>
Epoch 1/10 | 训练集: 0.1749 | 验证集: 0.1823
Epoch 2/10 | 训练集: 0.1913 | 验证集: 0.1992
Epoch 3/10 | 训练集: 0.2020 | 验证集: 0.2093
Epoch 4/10 | 训练集: 0.2472 | 验证集: 0.2610
Epoch 5/10 | 训练集: 0.2851 | 验证集: 0.3020
Epoch 6/10 | 训练集: 0.3173 | 验证集: 0.3355
Epoch 7/10 | 训练集: 0.3564 | 验证集: 0.3745
Epoch 8/10 | 训练集: 0.4038 | 验证集: 0.4240
Epoch 9/10 | 训练集: 0.4652 | 验证集: 0.4872
Epoch 10/10 | 训练集: 0.5298 | 验证集: 0.5511
训练结束！已恢复验证集最高准确率（0.5511）时的模型参数。
✅ 模型参数已成功保存到 lenet_params.pkl！
最终测试结果为：0.5407
</pre>

---

### 🎯 第二阶段：优化现有工程
- **调大训练轮次**：将 `epochs` 从 `10` 增加至 `30` 或 `50`，让模型有充足时间收敛至更高的精度（目标突破 90%）。
- **调整 Batch Size**：将 `mini_batch_size` 从 `100` 调大为 `256`，提高梯度稳定性。
- **加入学习率衰减**：在训练中途（如 `epoch == 15` 时）手动将 `lr` 降为原来的 1/10（例如从 `0.001` 调整为 `0.0001`）（在 Trainer 里加个简单的 `if epoch == 15: optimizer.lr *= 0.1`，或者直接改 `main.py`），实现后期精准微调。
- **尝试数据增强**：在 `main.py` 加载数据后，对训练集随机做轻微翻转或添加噪声，提升模型泛化能力。

- **任务进度**：第二阶段任务进行中。

**📈 第二阶段训练配置与运行结果：**
训练配置：
<pre>
# 调大训练轮次以及调整 Batch Size
trainer = Trainer(
    network,
    x_train,
    t_train,
    x_val,
    t_val,
    epochs=50,
    mini_batch_size=256,
    optimizer="Adam",
    optimizer_param={"lr": 0.001},
)
# 加入学习率衰减。每隔 15 个 epoch，学习率缩小 10 倍
if epoch != 0 and epoch % 15 == 0:
    self.optimizer.lr *= 0.1
    print(f"第 {epoch} 个 epoch，学习率已衰减为 {self.optimizer.lr}")
# 数据增强:几何变换 + 添加噪声
x_batch=data_augmentation(x_batch)
x_batch=add_noise(x_batch,sigma=0.02)
</pre>
运行结果：
<pre>
Epoch 1/10 | 训练集: 0.1749 | 验证集: 0.1823
Epoch 2/10 | 训练集: 0.1913 | 验证集: 0.1992
Epoch 3/10 | 训练集: 0.2020 | 验证集: 0.2093
Epoch 4/10 | 训练集: 0.2472 | 验证集: 0.2610
Epoch 5/10 | 训练集: 0.2851 | 验证集: 0.3020
Epoch 6/10 | 训练集: 0.3173 | 验证集: 0.3355
Epoch 7/10 | 训练集: 0.3564 | 验证集: 0.3745
Epoch 8/10 | 训练集: 0.4038 | 验证集: 0.4240
Epoch 9/10 | 训练集: 0.4652 | 验证集: 0.4872
Epoch 10/10 | 训练集: 0.5298 | 验证集: 0.5511
训练结束！已恢复验证集最高准确率（0.5511）时的模型参数。
✅ 模型参数已成功保存到 lenet_params.pkl！
最终测试结果为：0.5407
</pre>

---

### 📊 第三阶段：工程化完善
- **绘制训练曲线**：使用 `matplotlib` 将 `train_loss_list`、`train_acc_list` 和 `val_acc_list` 绘制成折线图，直观展示收敛过程与过拟合情况。
- **完善 README 与代码注释**：补充详细的调试记录与运行结果，让项目更具工程参考价值。

### 🌐 第四阶段：跨数据集实战
- **挑战 CIFAR-10 数据集**：将手搓代码迁移到 `32×32` 的彩色图像（3 通道，10 类）任务中。
- **技术点**：需修改 `input_dim` 为 `(3, 32, 32)`，并对第一层卷积 `pad` 参数进行适配，验证代码的通用性与扩展性。

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
| **C1** | 卷积 + ReLU | (N, 1, 28, 28) | (N, 6, 24, 24) | 6个 5x5，步幅1，无填充 |
| **S2** | 池化 | (N, 6, 24, 24) | (N, 6, 12, 12) | 窗口 2x2，步幅2 |
| **C3** | 卷积 + ReLU | (N, 6, 12, 12) | (N, 16, 8, 8) | 16个 5x5，步幅1 |
| **S4** | 池化 | (N, 16, 8, 8) | (N, 16, 4, 4) | 窗口 2x2，步幅2 |
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
├── learning/                 # learning 文件夹（以学习新知识为目的创建的文件夹，不参与项目开发）
├── lenet5.py                 # 1. 这里写 LeNet-5 的网络类
├── trainer.py                # 2. 这里写 Trainer 类
├── save_and_load.py          # 3. 这里写 pkl 保存和加载的测试
└── main.py                   # 4. 主程序：把上面全部串起来
└── debugfromgradient.py      # 5. 测试程序，通过numerical_gradient找出梯度bug
</pre>
