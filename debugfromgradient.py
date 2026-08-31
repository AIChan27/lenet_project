import numpy as np
from dataset.mnist import load_mnist
from lenet5 import LeNet5
from trainer import Trainer
from save_and_load import save_params, load_params
from common.gradient import numerical_gradient


(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, flatten=False, one_hot_label=True
)

# 取极小的一批数据测试
x_batch = x_train[:4]
t_batch = t_train[:4]

network = LeNet5(input_dim=(1, 28, 28))
# 1. 用你的反向传播算梯度
grads_backprop = network.gradient(x_batch, t_batch)

# 2. 用数值微分算梯度（这是一个标准答案）
def loss_W(w):
    return network.loss(x_batch, t_batch)

# 3. 对比两者的差距
for key in network.params:
    grads_num = numerical_gradient(loss_W, network.params[key])
    diff = np.abs(grads_backprop[key] - grads_num).max()
    print(f"{key} 的最大误差: {diff}")


