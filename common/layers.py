import numpy as np
from common.functions import *
from common.util import *

# Conv-->ReLU-->Pooling（MAX）-->Conv-->ReLU-->Pooling（MAX）-->Conv-->Affine-->Affine-->Softmax


class Convolution:
    """
    ----------------------------------------------------------------------
    （N，C，H，W）*（FN，C，FH，FW）→（N，FN，OH，OW）+（FN，1，1）→（N，FN，OH，OW）
    输入数据*滤波器+偏置→输出数据
    输出大小为（填充为P，步幅为S）：
    OH = ( H + 2P − FH ) / S + 1
    OW = ( W + 2P − FW ) / S + 1
    ----------------------------------------------------------------------
    """

    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        self.x = None
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None

    def forward(self, x):
        # 准备数据并确定输出数据规模
        N, C, H, W = x.shape
        FN, C, FH, FW = self.W.shape
        out_h = int((H + 2 * self.pad - FH) / self.stride + 1)
        out_w = int((W + 2 * self.pad - FW) / self.stride + 1)
        self.x = x
        self.col = im2col(x, FH, FW, self.stride, self.pad)
        self.col_W = self.W.reshape(FN, -1).T
        out = self.col @ self.col_W + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
        self.db = np.sum(dout, axis=0)
        self.dW = self.col.T @ dout
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        dcol = dout @ self.col_W.T
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        # out = np.maximum(0, x)
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int((H - self.pool_h) / self.stride + 1)
        out_w = int((W - self.pool_w) / self.stride + 1)
        self.x = x
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)
        self.arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        """
        ----------------------------------------------------------------------
        为什么这里使用双括号？
        np.zeros 接收的第一个参数本来是一个形状元组（例如 (2,3)）。
        虽然写成 np.zeros(2,3) 在 NumPy 里也能运行（它相当于把两个数字当形状参数），
        但用双括号能明确告诉程序“这是一个元组”，
        可读性更好，也能避免多参数调用带来的潜在混淆。
        ----------------------------------------------------------------------
        """
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        """
        ----------------------------------------------------------------------
        (pool_size,) 的写法：
        这是 Python 中只有一个元素的元组的写法。
        因为 (pool_size) 会被解释为普通的括号运算，
        结果就是个数字（比如 (4) 就是 4）。
        加上逗号变成 (pool_size,)，它就是一个真正的元组 (4,)。
        它的作用是和 dout.shape 进行元组拼接，
        比如 dout.shape 是 (N, out_h, out_w, C)，拼接后变成 (N, out_h, out_w, C, pool_size)，
        完美对应展开后的窗口维度！
        ----------------------------------------------------------------------
        """
        dmax = dmax.reshape(dout.shape + (pool_size,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = x @ self.W + self.b
        return out

    def backward(self, dout):
        dx = dout @ self.W.T
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx
