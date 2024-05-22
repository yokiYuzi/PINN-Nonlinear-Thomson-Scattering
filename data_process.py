import os
import numpy as np
import random

class CustomDataLoader:
    def __init__(self, data_dir, file_prefix, file_suffix, file_range):
        self.data_dir = data_dir
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.file_range = file_range
        self.all_data = []

    def load_data(self):
        # 加载所有数据文件
        for a0 in self.file_range:
            file_name = f"{self.file_prefix}{a0:.1f}{self.file_suffix}"
            file_path = os.path.join(self.data_dir, file_name)
            data = np.loadtxt(file_path)
            self.all_data.append(data)
        print(f"Total files loaded: {len(self.all_data)}")
        random.shuffle(self.all_data)  # 打乱数据

    def get_train_test_split(self, train_ratio=0.8):
        # 根据比例分割数据集
        num_train = int(len(self.all_data) * train_ratio)
        train_data = self.all_data[:num_train]
        test_data = self.all_data[num_train:]
        return train_data, test_data