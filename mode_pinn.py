#搭建I-PINN的主要训练流程
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from data_process import CustomDataLoader as PINNDataLoader
import matplotlib.pyplot as plt
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
            nn.Linear(15 * 1999, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
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
    all_epoch_loss = []
    for epoch in range(20):
        epoch_loss = []
        for  inputs, last_seven_first_row in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            
            optimizer.zero_grad()
            outputs = model(inputs)
            #要检查loss_function的参数顺序
            # 传递参数到损失函数
            #物理信息数据就是inputs(前15列)
            #多传入一个后7列第一行的参数修整即可
            loss = calculate_loss(outputs,  last_seven_first_row, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        average_loss = np.mean(epoch_loss)
        all_epoch_loss.append(average_loss)   
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        print(f'Epoch {epoch+1}, 平均损失: {np.mean(epoch_loss)}')
    # 保存模型
    torch.save(model.state_dict(), 'simple_nn_model.pth')
    plt.figure(figsize=(10, 5))
    plt.plot(all_epoch_loss, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def test_model(test_samples, model):
    model.eval()  # Set model to evaluation mode
    test_losses = []
    predictions = []
    actuals = []
    
    for inputs, last_seven_first_row in test_samples:
        with torch.no_grad():
            outputs = model(inputs)
            loss = calculate_loss(outputs, last_seven_first_row, inputs)
            test_losses.append(loss.item())
            predictions.extend(outputs.numpy())
            actuals.extend(last_seven_first_row.numpy())
    
    # Calculate average loss
    avg_loss = np.mean(test_losses)
    print(f'Average Test Loss: {avg_loss}')

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(actuals, label='Actual Data')
    plt.plot(predictions, label='Predicted Data', linestyle='--')
    plt.title('Comparison of Actual and Predicted Data')
    plt.xlabel('Data Index')
    plt.ylabel('Data Value')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data_loader = PINNDataLoader('F:\\PINN_code\\Data_base_input\\CombinedData', 'combined_data_a0=', '.txt', np.arange(0.1, 10.1, 0.1))
    data_loader.load_data()
    train_samples, test_samples = data_loader.get_train_test_split(train_ratio=0.8)

    print(f"Number of train samples: {len(train_samples)}, Number of test samples: {len(test_samples)}")
    train_model(train_samples)
    model = PINN()
    model.load_state_dict(torch.load('simple_nn_model.pth'))
    test_model(test_samples, model)