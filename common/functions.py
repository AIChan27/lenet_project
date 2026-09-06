import numpy as np
from scipy.ndimage import rotate


# 激活函数 sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 激活函数 ReLu
def relu(x):
    return np.maximum(0, x)


# 输出层softmax函数
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    out = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    return out


# 损失函数：均方误差 sum_squared_error
def sum_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# 损失函数：交叉熵误差 cross_entropy_error
def cross_entropy_error(y, t):
    # 即如果传入的是单张图 (10,)，变成 (1, 10)，统一格式
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 如果 t 是 One-hot 格式（矩阵），把它转换成数字标签（例如 [5, 3, 7]）
    if t.size == y.size:
        t = t.argmax(axis=1)
    # 取出批次大小 N
    batch_size = y.shape[0]
    """
    ----------------------------------------------------------------------
    核心公式的等价实现

    原始公式：E = - 1/N * Σn Σk tnk * ln(ynk)
    假设t=[0,0,0,0,0,1,0,0,0,0]；
    如果按照原始公式，此时需要t与y逐元素相乘后相加，但其实t中只有1个元素是有效元素。

    运行t = t.argmax(axis=1)时，将会把[0,0,0,0,0,1,0,0,0,0]变成数字5；
    这样配合y[np.arange(batch_size), t]，np.arange(batch_size)提供行号，t提供列号；
    即若t=[5，3，7]，则y[0,5]、y[1,3]、y[2,7]分别代表取第0张图的第五列，第一张图的第三列，第二张图的第七列...
    等价于公式里的Σk
    ----------------------------------------------------------------------
    """
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# 数据增强---几何变换：平移、翻转、旋转
def data_augmentation(x,max_shift=2,max_rotate = 15):
    batch_size = x.shape[0]
    x_aug = x.copy()
    for i in range(batch_size):
        # 平移
        if np.random.rand() > 0.5:
            dx = np.random.randint(-max_shift, max_shift + 1)
            dy = np.random.randint(-max_shift, max_shift + 1)
            x_aug[i] = np.roll(x_aug[i], shift=(dy, dx), axis=(1, 2))
        # 翻转
        if np.random.rand() > 0.5:
            x_aug[i] = np.fliplr(x_aug[i])
        # 旋转
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-max_rotate, max_rotate)
            x_aug[i] = rotate(x_aug[i], angle, reshape=False, mode="constant", cval=0.0)
    return x_aug


# 数据增强---添加噪声：
def add_noise(x, sigma=0.02):
    batch_size = x.shape[0]
    x_aug = x.copy()
    noise = np.random.randn(*x.shape) * sigma
    x_noisy = x_aug + noise
    return np.clip(x_noisy, 0.0, 1.0)
