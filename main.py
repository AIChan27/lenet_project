import numpy as np
from dataset.mnist import load_mnist
from lenet5 import LeNet5
from trainer import Trainer
from save_and_load import save_params, load_params

"""
----------------------------------------------------------------------
1. 加载数据（注意 flatten=False，保持 4D 结构）
2. 实例化网络
3. 实例化训练器并开始训练
4. 训练完毕，保存参数
5. 加载参数，进行最终测试
----------------------------------------------------------------------
"""
# 1. 加载数据（注意 flatten=False，保持 4D 结构）
(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, flatten=False, one_hot_label=True
)
"""
----------------------------------------------------------------------
切分验证集
x_train[:50000] 拿前5万张作为“真正的训练集”，x_train[50000:] 拿剩下的1万张作为验证集（x_val）。
因为你的 Trainer 需要 x_val 和 t_val 来监控过拟合、确定早停。
测试集（x_test, t_test）是绝对不能动的，必须留到最后测试！ 你把训练集切分成了5万+1万，是正确的做法。
----------------------------------------------------------------------
"""
x_train, x_val = x_train[:50000], x_train[50000:]
t_train, t_val = t_train[:50000], t_train[50000:]
# 2. 实例化网络
network = LeNet5(input_dim=(1, 28, 28))
# 3. 实例化训练器并开始训练
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
trainer.train()
# 4. 训练完毕，保存参数
save_params(network, "lenet_params.pkl")
# 5. （可选演示）加载参数：模拟下次开机重新加载模型
# 注意：这一步会覆盖当前网络，但因为我们刚保存的就是最好的，所以没问题
# load_params(network, "lenet_params.pkl")
accuracy=network.accuracy(x_test,t_test)
print(f"最终测试结果为：{accuracy:.4f}")
