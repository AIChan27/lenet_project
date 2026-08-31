import numpy as np
import copy
from collections import OrderedDict
from common.optimizer import *

"""
----------------------------------------------------------------------
Trainer类的作用，通俗来讲就是不断地更新 w 和 b，直到模型学得差不多了。
训练完毕后，把 w 和 b 保存成 .pkl 文件（对应你项目里的 save_and_load.py）。
下一次使用时（加载参数）：直接加载 .pkl 文件，用 predict() 方法直接识别新图片。这时候完全不需要 Trainer，也不会更新 w 和 b。
类参数：
    network：深度学习模型类实例
    x_train, t_train：训练数据（考题+答案）
    x_val, t_val：验证集数据（模拟考），在训练中途用来检查有没有“死记硬背（过拟合）”，只观测不改参数。
    epochs：训练轮数。注意：epochs 是外层大循环。比如 epochs=10 意味着“把训练集从头到尾完整看10遍”。
    mini_batch_size：每轮“吃”多少张图。比如 mini_batch_size=100，意味着跑完一个 epoch，需要更新 60000 / 100 = 600 次。
    optimizer：优化器名称的字符串（'SGD' 或 'Adam'）。
    optimizer_param：优化器参数的字典。比如 {'lr': 0.01}。这个字典会传给 _create_optimizer，然后解包传给优化器类。
训练集和验证集的配合（“尽信书则不如无书”）：
    第一阶段：训练阶段：
        模型在训练集上抽取 Mini-batch（比如 100 张）。
        进行前向传播、反向传播，用优化器算出新的梯度，更新 w 和 b。
        这时候验证集在场外默默看着，不插手。
    第二阶段：验证阶段：
        当一个 epoch（把 60000 张训练集全部看了一遍）结束后，训练暂停。
        把验证集（10000 张）喂给模型，只做前向传播（不做反向传播，不更新参数）。
        模型对着验证集输出预测，算出验证集准确率。
    配合的核心目的--防过拟合监控（过拟合：只能拟合训练数据，但不能很好地拟合不包含在训练数据中的其他数据的状态。）：
        如果训练集准确率一直在飙升，但验证集准确率停滞甚至开始下降，这意味着模型发生了过拟合。
        这时Trainer就会喊停：“别练了，你已经不会做新题了！”，这就叫早停（Early Stopping）。
        另外，调节学习率，也是通过观察验证集的表现来决定的。
----------------------------------------------------------------------
"""


class Trainer:
    def __init__(
        self,
        network,
        x_train,
        t_train,
        x_val,
        t_val,
        epochs=10,
        mini_batch_size=100,
        optimizer="SGD",
        optimizer_param={"lr": 0.01},
    ):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_val = x_val
        self.t_val = t_val
        self.epochs = epochs
        self.batch_size = mini_batch_size
        # 创建优化器
        self.optimizer = self._create_optimizer(optimizer, optimizer_param)
        # 计算总训练量
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / self.batch_size, 1)
        # 用于记录训练过程的容器
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_acc_list = []

    # 通过解包字典（**param）来创建优化器，以后加新优化器只需加个elif
    def _create_optimizer(self, name, param):
        if name == "SGD":
            return SGD(lr=param["lr"])
        elif name == "Momentum":
            return Momentum(lr=param["lr"], momentum=param.get("momentum", 0.9))
        elif name == "AdaGrad":
            return AdaGrad(lr=param["lr"])
        elif name == "Adam":
            return Adam(
                lr=param["lr"],
                beta1=param.get("beta1", 0.9),
                beta2=param.get("beta2", 0.999),
            )
        else:
            raise ValueError(f"未知的优化器: {name}")

    """
    ----------------------------------------------------------------------
    Trainer类的核心方法，其基本步骤为：
    --- ① 平时写作业（训练阶段） ---
    --- ② 模拟考（验证阶段） ---
    --- ③ 防过拟合监控（保存最佳 & 早停） ---
    --- ④ 训练结束，恢复最佳状态 ---
    ----------------------------------------------------------------------
    """

    def train(self):
        # 初始化最佳验证集准确率（设为0）和最佳参数保存容器
        best_val_acc = 0.0
        best_params = None
        # 早停耐心值：连续5轮验证集没提升就停下
        patience = 5
        # 循环开始
        for epoch in range(self.epochs):
            # --- ① 平时写作业（训练阶段） ---
            # 抽取 mini-batch 将庞大的训练集进行拆分
            batch_mask = np.random.choice(self.train_size, self.batch_size)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]
            # 前向、反向，更新参数
            grads = self.network.gradient(x_batch, t_batch)
            self.optimizer.update(self.network.params, grads)
            # 记录 Loss
            loss = self.network.loss(x_batch, t_batch)
            self.train_loss_list.append(loss)
            # --- ② 模拟考（验证阶段） ---
            # 一个 epoch 结束，检查训练集和验证集准确率
            # train_acc：训练集 val_acc：验证集
            train_acc=self.network.accuracy(self.x_train,self.t_train)
            val_acc=self.network.accuracy(self.x_val,self.t_val)
            self.train_acc_list.append(train_acc)
            self.val_acc_list.append(val_acc)
            print(f"Epoch {epoch+1}/{self.epochs} | 训练集: {train_acc:.4f} | 验证集: {val_acc:.4f}")
            # --- ③ 防过拟合监控（保存最佳 & 早停） ---
            # 如果验证集准确率创了新高，就保存当前的所有参数
            if val_acc>best_val_acc:
                best_val_acc=val_acc
                # 深拷贝，防止后续更新变味
                best_params=copy.deepcopy(self.network.params)
            """
            ----------------------------------------------------------------------
            早停判断：如果最近 5 轮的验证集成绩都没有超过之前的最佳成绩
            通俗理解：只要验证集准确率一直没超过“历史最高点（best_val_acc）”，我就认为它正在走向过拟合。因为它虽然还在训练，但已经“不会做新题”了。
            拆分if语句：
            前半部分：len(self.val_acc_list) > patience：
                意思是你至少已经跑完 patience+1 个 epoch 了，防止训练刚开始（还没积累足够数据）就误判为早停。
            后半部分：all(self.val_acc_list[-i] <= best_val_acc for i in range(1, patience + 1))：
                range(1, patience + 1) 生成序列 [1, 2, 3, 4, 5]。
                对于每个 i，self.val_acc_list[-i] 代表最近第 i 轮的验证集成绩（即倒数第1轮、倒数第2轮...倒数第5轮）。
                <= best_val_acc：判断这5轮的每一轮成绩，是不是都小于等于历史最高成绩。
                all(...)：只有当这5个条件全部成立时，整体才是 True。
            ----------------------------------------------------------------------
            """
            if len(self.val_acc_list)>patience and all(self.val_acc_list[-i]<=best_val_acc for i in range(1,patience+1)):
                print("验证集连续多轮未提升，触发早停！")
                break
        # --- ④ 训练结束，恢复最佳状态 ---
        if best_params is not None:
            self.network.params=best_params
            print(f"训练结束！已恢复验证集最高准确率（{best_val_acc:.4f}）时的模型参数。")