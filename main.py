import numpy as np
import os
import matplotlib.pyplot as plt
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
    epochs=100,
    mini_batch_size=256,
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

# ================= 画图（训练结束后做） =================
# 设置中文字体，防止乱码
# 用来正常显示中文标签（Windows 默认用黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
# 创建画布，画两个子图：左边看Loss，右边看准确率
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左边：Loss 曲线
axes[0].plot(trainer.train_loss_list, label='训练损失', color='blue')
axes[0].set_xlabel('迭代次数')
axes[0].set_ylabel('损失值')
axes[0].set_title('训练损失曲线')
axes[0].legend()
axes[0].grid(True)

# 右边：准确率曲线
axes[1].plot(trainer.train_acc_list, label='训练准确率', color='green', marker='o')
axes[1].plot(trainer.val_acc_list, label='验证准确率', color='red', marker='s')
axes[1].set_xlabel('训练轮数 (Epoch)')
axes[1].set_ylabel('准确率')
axes[1].set_title('训练与验证准确率对比')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

if not os.path.exists('assets'):
    os.makedirs('assets')  # 创建一个存放图片的文件夹
plt.savefig('assets/training_curves.png', dpi=300, bbox_inches='tight')

plt.show()


