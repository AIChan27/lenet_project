```text

LeNet-5 手搓项目导入大纲:
1. 目录结构：
    lenet_project/
    │── dataset/             # dataset 文件夹（包含 mnist.py）
    │── common/              # common 文件夹（包含 layers.py, gradient.py, functions.py）
    │── lenet5.py            # 1. 这里写 LeNet-5 的网络类
    │── trainer.py           # 2. 这里写 Trainer 类
    │── save_and_load.py     # 3. 这里写 pkl 保存和加载的测试
    │── main.py              # 4. 主程序：把上面全部串起来

2. 核心设定（数据与模型）：
    使用模型：LeNet-5（经典卷积神经网络，1998年提出）。
    数据集：MNIST（手写数字识别），使用 flatten=False 加载（保持 4D 结构）。
    数据初始大小（关键）：
        输入形状：(N, 1, 28, 28) （即：100张图，1个灰度通道，28x28像素）。
        标签形状：(N, 10) （独热编码）。
    训练批次（Batch Size）：100
    训练轮次（Epochs）：10 - 20 个左右
3. LeNet-5 标准维度流（完整推导）