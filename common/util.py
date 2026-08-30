import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    ----------------------------------------------------------------------
    input_data：输入数据，通常是一个4维张量 (N, C, H, W)（即：数据批量数，通道数，高度，宽度）。
    filter_h / filter_w：滤波器（卷积核）的高度和宽度（比如 3×3 的核就是 3, 3）。
    stride：步幅。滤波器每次滑动的像素数。默认是 1。
    pad：填充。在图像的四周补0的像素数。默认是 0。设置填充可以控制输出图像的大小。
    ----------------------------------------------------------------------
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    ----------------------------------------------------------------------
    Col（二维数组）：	输入。就是 im2col 生成的矩阵，形状为 (N * OH * OW, C * FH * FW)。
    x_shape（元组 (tuple)）：	原始输入图片的形状 (N, C, H, W)。必须传入，因为函数需要知道最终要还原成多大的图片，以及填充（Pad）之前的原始尺寸。
    filter_h（int）：滤波器（卷积核）的高度。与 im2col 保持一致。
    filter_w	（int）：滤波器（卷积核）的宽度。与 im2col 保持一致。
    Stride（int）：步幅。默认是 1。与 im2col 保持一致。
    pad	（int）：填充。默认是 0。注意：函数内部会先还原出带填充的大图 (H + 2*pad, W + 2*pad)，最后再把四周的填充裁剪掉，只返回原始尺寸 (H, W)。
    ----------------------------------------------------------------------
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(
        0, 3, 4, 5, 1, 2
    )

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad : H + pad, pad : W + pad]
