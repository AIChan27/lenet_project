```text
lenet_project/
│── dataset/             # dataset 文件夹（包含 mnist.py）
│── common/              # common 文件夹（包含 layers.py, gradient.py, functions.py）
│── lenet5.py            # 1. 这里写 LeNet-5 的网络类
│── trainer.py           # 2. 这里写我自己的 Trainer 类
│── save_and_load.py     # 3. 这里写 pkl 保存和加载的测试
│── main.py              # 4. 主程序：把上面全部串起来