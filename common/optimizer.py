import numpy as np

# 最优化：找到使损失函数的值尽可能小的参数。


# 随机梯度下降法 SGD
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


# "动量"法 Momentum
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


# 自适应学习率的梯度下降优化算法 ‌AdaGrad
# AdaGrad 为每个参数维护一个累积梯度平方和，梯度大的参数学习率自动变小，梯度小的参数学习率保持较大
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


# Adam全称是 Adaptive Moment Estimation（自适应矩估计）。它相当于“物理学中的动量 + 自适应刹车”的结合体
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        
        for key in params.keys():
            # 1. 更新一阶和二阶矩估计
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # 2. 执行偏差修正（注意这里用的是 self.iter！）
            mt_hat = self.m[key] / (1 - self.beta1**self.iter)
            vt_hat = self.v[key] / (1 - self.beta2**self.iter)
            
            # 3. 更新参数
            params[key] -= self.lr * mt_hat / (np.sqrt(vt_hat) + 1e-7)