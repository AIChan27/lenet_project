import numpy as np
from pathlib import Path
from PIL import Image
from lenet5 import LeNet5
from save_and_load import load_params
from common.functions import softmax
from scipy.ndimage import rotate, zoom

def random_augmentation(x, max_shift=2, max_rotate=15, scale_min=0.9, scale_max=1.1, crop_pad=2):
    # 1. 获取输入数据的形状：N（批量大小）、C（通道数）、H（高）、W（宽）
    # LeNet-5 输入是 (N, 1, 28, 28)
    N, C, H, W = x.shape

    # 2. 创建一个和 x 一样形状的全零数组，用来存放增强后的图片
    x_aug = np.zeros_like(x)

    # 3. 遍历批量中的每一张图，逐张处理
    for i in range(N):
        
        # 4. 取出第 i 张图，因为是单通道灰度图，所以取 x[i, 0]，形状变为 (H, W)
        img = x[i, 0]

        # 5. 【翻转】以50%概率水平翻转
        if np.random.rand() > 0.5:
            img = np.fliplr(img)

        # 6. 【平移】以50%概率进行随机平移
        if np.random.rand() > 0.5:
            # 生成-2到2的随机整数
            dx = np.random.randint(-max_shift, max_shift + 1) 
            dy = np.random.randint(-max_shift, max_shift + 1)
            # np.roll 是循环移位。如果把图片向右移，左边空出来的位置会自动从右边“卷”过来（类似环形）。
            # 虽然这模拟了平移，但会在边缘产生“环绕”效果。如果不想要环绕，需结合 np.pad 补0。
            img = np.roll(img, shift=(dy, dx), axis=(0, 1))

        # 7. 【旋转】以50%概率在±15度内旋转
        if np.random.rand() > 0.5:
            # 生成-15到15度之间的随机数
            angle = np.random.uniform(-max_rotate, max_rotate) 
            # 用 scipy.ndimage.rotate 旋转。reshape=False 表示保持原图尺寸不裁剪。
            # mode='constant', cval=0.0 表示旋转造成的空白区域用0（黑色）填充。
            img = rotate(img, angle, reshape=False, mode='constant', cval=0.0)

        # 8. 【缩放】以50%概率进行随机缩放
        if np.random.rand() > 0.5:
            # 生成0.9到1.1之间的随机缩放系数
            scale = np.random.uniform(scale_min, scale_max) 
            # 用 scipy.ndimage.zoom 缩放
            img = zoom(img, scale, order=1)
            h_new, w_new = img.shape
            # 如果缩放后比原图大，就随机裁剪中心区域回原尺寸
            if h_new >= H and w_new >= W:
                y = np.random.randint(0, h_new - H + 1)
                x_ = np.random.randint(0, w_new - W + 1)
                img = img[y:y+H, x_:x_+W]
            # 如果缩放后比原图小，就用0填充回原尺寸
            else:
                pad_x = W - w_new
                pad_y = H - h_new
                img = np.pad(img, ((pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2)), mode='constant')

        # 9. 【随机裁剪】以50%概率，先加一圈0边框，再随机裁剪回原尺寸（模拟物体局部被遮挡）
        if np.random.rand() > 0.5:
            img = np.pad(img, crop_pad, mode='constant')
            y = np.random.randint(0, crop_pad * 2)
            x_ = np.random.randint(0, crop_pad * 2)
            img = img[y:y+H, x_:x_+W]

        # 10. 把处理好的第 i 张图放回输出数组
        x_aug[i, 0] = img

    # 11. 返回增强后的整个批次
    return x_aug

def add_noise(x, sigma=0.02):
    """
    向输入图像添加高斯噪声
    参数:
        x: 形状为 (N, C, H, W) 的输入张量（注意：值在 0~1 之间）
        sigma: 噪声的强度（标准差），一般取 0.01 ~ 0.05
    返回:
        加噪后的图像（限制在 0~1 之间）
    """
    # 1. 生成和 x 形状一样的标准正态分布随机数（均值为0，方差为1），乘以sigma
    noise = np.random.randn(*x.shape) * sigma
    
    # 2. 将噪声加到原图上
    x_noisy = x + noise
    
    # 3. 最关键的一步！像素值不能低于0，也不能超过1。
    # np.clip(x, 0, 1) 会将小于0的数变成0，大于1的数变成1
    return np.clip(x_noisy, 0.0, 1.0)

# import numpy as np

# x = np.array([[1, 100, 50], [3, 200, 100]])

# print(np.mean(x, axis=0))
# print(np.sum(x, axis=0) / x.shape[0])

# 第一步：处理图片
base_dir = Path(__file__).parent
img_dir = base_dir / "inferenceimg"
jpg_files = list(img_dir.glob("*.jpg"))
img_count = len(jpg_files)
print(f"inference_img 文件夹下共有 {img_count} 张 JPG 图片。")

imgnp = np.zeros((img_count, 1, 28, 28), dtype=np.float32)

for count in range(img_count):
    img = Image.open(jpg_files[count]).convert("L")
    img = np.array(img) / 255.0
    
    if img.mean() > 0.5:
        img = 1.0 - img
        
    # 寻找数字边界框
    rows = np.any(img > 0.1, axis=1)
    cols = np.any(img > 0.1, axis=0)
    
    if np.any(rows) and np.any(cols):
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        img_cropped = img[rmin:rmax+1, cmin:cmax+1]
        if img_cropped.shape[0] > 0 and img_cropped.shape[1] > 0:
            img = img_cropped
    
    h, w = img.shape
    if h == 0 or w == 0:
        continue  # 极端情况跳过
    
    # 【修改】让数字占更大比例，便于识别！
    scale = 24.0 / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    if new_h == 0:
        new_h = 1
    if new_w == 0:
        new_w = 1
    
    img_pil = Image.fromarray((img * 255).astype(np.uint8)).resize((new_w, new_h))
    img = np.array(img_pil) / 255.0
    
    canvas = np.zeros((28, 28))
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = img
    
    imgnp[count, 0] = canvas

    Image.fromarray((canvas * 255).astype(np.uint8)).save(f"preprocessed_{count}.png")

# 第二步：加载模型并推理
network = LeNet5(input_dim=(1, 28, 28))
load_params(network, "lenet_params.pkl")

# 极其重要：切换为推理模式！
network.train_flag = False

# 获取原始得分（Logits）
logits = network.predict(imgnp)

# 手动应用 Softmax 得到概率
probs = softmax(logits)

# 第三步：得出结论
predictions = np.argmax(probs, axis=1)
confidences = np.max(probs, axis=1)

print("\n========== 推理结果展示 ==========")
for i in range(img_count):
    print(f"图片 {i+1}: 预测为数字 {predictions[i]}，置信度 {confidences[i]:.4f}")
print("==================================")