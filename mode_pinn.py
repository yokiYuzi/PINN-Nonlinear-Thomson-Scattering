#搭建I-PINN的主要训练流程
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from data_process import DataLoader as PINNDataLoader
from Loss_function import calculate_loss




class PINNDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 数据格式应与预处理一致
        data = self.samples[idx]
        inputs = data[:, :-7]  # 假设前面的部分是输入
        #这里传递出激光峰值振幅参数a_0
        #Laser_param = data[0, 15]  # 提取第16列的第一个数据
        #好像也不用传a0，要计算loss的是新的pred_a0啊
        #这里传递出前15列物理信息数据
        #前15列不是和inputs没有区别么（笑了）
        last_seven_first_row = data[0, -7:]  # 提取后七列的第一行
        return torch.tensor(inputs).float(),torch.tensor(last_seven_first_row).float()

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(15 * 89999, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 7)
        )
    
    def forward(self, x):
        # 确保输入x被正确地扁平化
        x = x.view(x.size(0), -1)  # 这会将每个输入样本扁平化为一维向量
        return self.fc(x)

def train_model(train_samples):
    dataset = PINNDataset(train_samples)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        for  inputs, last_seven_first_row in train_loader:
            
            optimizer.zero_grad()
            outputs = model(inputs)
            #要检查loss_function的参数顺序
            # 传递参数到损失函数
            #物理信息数据就是inputs(前15列)
            #多传入一个后7列第一行的参数修整即可
            loss = calculate_loss(outputs,  last_seven_first_row, inputs)
            loss.backward()
            optimizer.step()
            
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # 保存模型
    torch.save(model.state_dict(), 'simple_nn_model.pth')

if __name__ == "__main__":
    # 实例化数据加载器
    data_loader = PINNDataLoader('F:\\PINN_code\\Data_base_input\\CombinedData', 'combined_data_a0=', '.txt', np.arange(0.1, 1.1, 0.1))
    data_loader.load_data()
    
    train_samples = data_loader.sample_data(data_loader.train_data, 8)  # 获取8个训练样本
    train_model(train_samples)