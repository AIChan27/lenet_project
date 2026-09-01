import numpy as np
import math
from collections import OrderedDict
from common.layers import *

"""
----------------------------------------------------------------------
层名称	操作	输入维度	输出维度	参数/备注

C1	卷积 + ReLU	(N, 1, 28, 28)	(N, 6, 24, 24)	6个 5x5，步长1，无填充

S2	池化	(N, 6, 24, 24)	(N, 6, 12, 12)	窗口 2x2，步长2

C3	卷积 + ReLU	(N, 6, 12, 12)	(N, 16, 8, 8)	16个 5x5，步长1

S4	池化	(N, 16, 8, 8)	(N, 16, 4, 4)	窗口 2x2，步长2

F5	展平 + Affine	(N, 256)	(N, 120)	全连接

F6	Affine	(N, 120)	(N, 84)	全连接

Out	Affine + Softmax	(N, 84)	(N, 10)	输出层
----------------------------------------------------------------------
"""


class LeNet5:
    def __init__(
        self,
        input_dim=(1, 28, 28),
        conv_param_1={"filter_num": 6, "filter_size": 5, "pad": 0, "stride": 1},
        conv_param_2={"filter_num": 16, "filter_size": 5, "pad": 0, "stride": 1},
        hidden_size_1=120,
        hidden_size_2=84,
        output_size=10,
        weight_init_std=0.01,
    ):
        # 初始化权重
        self.params = {}
        # # C1	卷积 + ReLU	(N, 1, 28, 28)	(N, 6, 24, 24)	6个 5x5，步长1，无填充
        # self.params["W1"] = weight_init_std * np.random.randn(6, input_dim[0], 5, 5)
        # self.params["b1"] = np.zeros(6)

        # C1 卷积层使用 He 初始化（输入通道 * 卷积核大小 = node_num_1）
        node_num_1 = input_dim[0] * 5 * 5  # 第一层：1 * 5 * 5 = 25
        self.params["W1"] = np.random.randn(6, input_dim[0], 5, 5) *math.sqrt(2) / np.sqrt(node_num_1)
        self.params["b1"] = np.zeros(6)

        # # C3	卷积 + ReLU	(N, 6, 12, 12)	(N, 16, 8, 8)	16个 5x5，步长1
        # self.params["W2"] = weight_init_std * np.random.randn(16, 6, 5, 5)
        # self.params["b2"] = np.zeros(16)

        # C3 卷积层使用 He 初始化（输入通道 * 卷积核大小 = node_num_2）
        node_num_2 = 6 * 5 * 5  # 第二层：6 * 5 * 5 = 150
        self.params["W2"] = np.random.randn(16, 6, 5, 5) *math.sqrt(2) / np.sqrt(node_num_2)
        self.params["b2"] = np.zeros(16)

        # # F5	展平+Affine + ReLU	(N, 256)	(N, 120)	全连接
        # # 为什么是 256 ？因为输入是 16 * 4 * 4 = 256
        # self.params["W3"] = weight_init_std * np.random.randn(256, hidden_size_1)
        # self.params["b3"] = np.zeros(hidden_size_1)

        # F5 展平+Affine + ReLU 全连接层使用 He 初始化（node_num = 16 * 4 * 4 = 256）
        self.params["W3"] = np.random.randn(256, hidden_size_1) *math.sqrt(2) / np.sqrt(256)
        self.params["b3"] = np.zeros(hidden_size_1)

        # # F6	Affine + ReLU	(N, 120)	(N, 84)	全连接
        # self.params["W4"] = weight_init_std * np.random.randn(
        #     hidden_size_1, hidden_size_2
        # )
        # self.params["b4"] = np.zeros(hidden_size_2)

        # F6 Affine + ReLU 全连接层（120 -> 84）
        self.params["W4"] = np.random.randn(hidden_size_1, hidden_size_2) *math.sqrt(2) / np.sqrt(hidden_size_1)
        self.params["b4"] = np.zeros(hidden_size_2)

        # # Out	Affine + Softmax	(N, 84)	(N, 10)	输出层
        # self.params["W5"] = weight_init_std * np.random.randn(
        #     hidden_size_2, output_size
        # )
        # self.params["b5"] = np.zeros(output_size)

        # Out Affine + Softmax 全连接层（84 -> 10）
        self.params["W5"] = np.random.randn(hidden_size_2, output_size) *math.sqrt(2) / np.sqrt(hidden_size_2)
        self.params["b5"] = np.zeros(output_size)
        # 生成层：
        # Convolution-->ReLU-->Pooling-->Convolution-->ReLU-->Pooling-->展平+Affine1 (256→120)-->ReLU-->Affine2 (120→84)-->ReLU-->Affine3 (84→10)-->Softmax
        self.layers = OrderedDict()
        self.layers["Convolution_1"] = Convolution(
            self.params["W1"],
            self.params["b1"],
            conv_param_1["stride"],
            conv_param_1["pad"],
        )
        self.layers["ReLU_1"] = ReLU()
        self.layers["Pooling_1"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Convolution_2"] = Convolution(
            self.params["W2"],
            self.params["b2"],
            conv_param_2["stride"],
            conv_param_2["pad"],
        )
        self.layers["ReLU_2"] = ReLU()
        self.layers["Pooling_2"] = Pooling(pool_h=2, pool_w=2, stride=2)
        # 展平+Affine1 (256→120)-->ReLU-->Affine2 (120→84)-->ReLU-->Affine3 (84→10)-->Softmax
        self.layers["Affine_1"] = Affine(self.params["W3"], self.params["b3"])
        self.layers["ReLU_3"] = ReLU()
        self.layers["Affine_2"] = Affine(self.params["W4"], self.params["b4"])
        self.layers["ReLU_4"] = ReLU()
        self.layers["Affine_3"] = Affine(self.params["W5"], self.params["b5"])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    # 计算识别准确率
    def accuracy(self, x, t):
        y = self.predict(x)
        # 找出每一行（每张图）中，概率最大的那个分类的索引。
        y = np.argmax(y, axis=1)
        # 如果 t 是独热编码（(N, 10)），也把它转成数字索引（(N,)）
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        """
        ----------------------------------------------------------------------
        使用一个例子来理解 accuracy = np.sum(y == t) / float(x.shape[0])
        假设：y = [5, 3, 7]、t = [5, 1, 7]、当前这批有100 张图
        ① y == t 是一个布尔数组，根据例子，数组返回[True, False, True]
        ② 在 NumPy 中，True --> 1，False --> 0。np.sum([True, False, True]) 的结果是 2（说明100张图里猜对了2张）。
        ③ float(x.shape[0]) 就是 N（当前这批有多少张图，此时N=100）。
        ④ 最终结果：2 / 100 = 0.02（2% 的准确率）。这个结果会被送到 Trainer 里打印出来，用来判断模型有没有过拟合。
        ----------------------------------------------------------------------
        """
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        grads["W1"] = self.layers["Convolution_1"].dW
        grads["b1"] = self.layers["Convolution_1"].db
        grads["W2"] = self.layers["Convolution_2"].dW
        grads["b2"] = self.layers["Convolution_2"].db
        grads["W3"] = self.layers["Affine_1"].dW
        grads["b3"] = self.layers["Affine_1"].db
        grads["W4"] = self.layers["Affine_2"].dW
        grads["b4"] = self.layers["Affine_2"].db
        grads["W5"] = self.layers["Affine_3"].dW
        grads["b5"] = self.layers["Affine_3"].db
        return grads
