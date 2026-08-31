import pickle
import numpy as np

# 注：save 和 load 是纯函数，它们不需要知道网络内部结构，只需要 network.params 这个字典！

def save_params(network, filename="lenet_params.pkl"):
    """
    将训练好的模型权重（network.params）保存到指定的 .pkl 文件中。
    
    参数:
        network: 你的 LeNet5 模型实例
        filename: 保存的文件名（默认 lenet_params.pkl）
    """
    # 'wb' 表示二进制写入模式（Write Binary）
    with open(filename, 'wb') as f:
        # pickle.dump 将字典序列化并写入文件
        pickle.dump(network.params, f)
    
    print(f"✅ 模型参数已成功保存到 {filename}！")


def load_params(network, filename="lenet_params.pkl"):
    """
    从 .pkl 文件中加载模型权重，并赋值给 network.params。
    
    参数:
        network: 你的 LeNet5 模型实例（必须已经初始化了结构）
        filename: 要读取的文件名（默认 lenet_params.pkl）
    """
    # 'rb' 表示二进制读取模式（Read Binary）
    with open(filename, 'rb') as f:
        # pickle.load 读取字典并赋值给网络
        network.params = pickle.load(f)
        
    print(f"✅ 模型参数已从 {filename} 成功加载！")
    print("现在你可以直接调用 network.predict(x) 进行识别了！")


# 以下是测试代码，如果你直接运行 python save_and_load.py，它会执行：
if __name__ == "__main__":
    from lenet5 import LeNet5
    
    # 1. 先实例化一个空的网络（权重是随机初始化的）
    test_net = LeNet5(input_dim=(1, 28, 28))
    
    # 2. 保存它（虽然还没训练，但可以测试保存功能）
    save_params(test_net, "test_params.pkl")
    
    # 3. 创建一个新的网络，并把参数加载进去
    new_net = LeNet5(input_dim=(1, 28, 28))
    load_params(new_net, "test_params.pkl")
    
    # 4. 验证：看看加载后的参数是不是和保存前的一样（数值上完全一致）
    if np.allclose(test_net.params['W1'], new_net.params['W1']):
        print("🎉 保存和加载功能测试成功！参数完全一致！")
    else:
        print("❌ 测试失败，参数不一致！")