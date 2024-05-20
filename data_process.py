import os
import numpy as np
import random

class DataLoader:
    def __init__(self, data_dir, file_prefix, file_suffix, file_range):
        self.data_dir = data_dir
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.file_range = file_range
        self.train_data = []
        self.test_data = []

    def load_data(self):
        # 加载所有数据文件
        all_data = []
        for a0 in self.file_range:
            file_name = f"{self.file_prefix}{a0:.1f}{self.file_suffix}"
            file_path = os.path.join(self.data_dir, file_name)
            data = np.loadtxt(file_path)
            all_data.append(data)
        
        # 打乱数据
        random.seed(42)  # 设置随机种子以确保结果可重复
        random.shuffle(all_data)
        
        # 分割成训练集和测试集（80%训练集，20%测试集）
        num_train = int(len(all_data) * 0.8)
        self.train_data = all_data[:num_train]
        self.test_data = all_data[num_train:]
        
    def sample_data(self, data_list, num_samples):
        if len(data_list) < num_samples:
            raise ValueError(f"数据集中的数据不足，只剩{len(data_list)}组数据，无法抽取{num_samples}组")
        sampled_data = random.sample(data_list, num_samples)
        return sampled_data

# 使用示例
if __name__ == "__main__":
    data_dir = 'F:\\PINN_code\\Data_base_input\\CombinedData'
    file_prefix = 'combined_data_a0='
    file_suffix = '.txt'
    file_range = np.arange(0.1, 1.1, 0.1)

    data_loader = DataLoader(data_dir, file_prefix, file_suffix, file_range)
    data_loader.load_data()

    try:
        train_samples = data_loader.sample_data(data_loader.train_data, 8)
        test_samples = data_loader.sample_data(data_loader.test_data, 2)
        print("Sampled train data shapes:", [sample.shape for sample in train_samples])
        print("Sampled test data shapes:", [sample.shape for sample in test_samples])
    except ValueError as e:
        print(e)