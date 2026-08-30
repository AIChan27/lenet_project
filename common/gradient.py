import numpy as np


def numerical_gradient_1d(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # 生成和x形状相同的数组
    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 还原值
    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        for idx, x in enumerate(X):
            grad[idx] = numerical_gradient_1d(f, x)

        return grad


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    """
    ----------------------------------------------------------------------
    NumPy 提供的一个“万能多维数组遍历器”
        x：数据
        flags=["multi_index"]：返回一个多维坐标元组
        op_flags=["readwrite"]：允许写
    ----------------------------------------------------------------------
    """
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        # 读取当前坐标（比如 (2, 0, 4, 4)）
        idx = it.multi_index
        # 取数
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        # 计算导数
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        # 还原值
        x[idx] = tmp_val
        # 下一个点
        it.iternext()

    return grad