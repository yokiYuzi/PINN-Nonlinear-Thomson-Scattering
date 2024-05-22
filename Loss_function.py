#这个文件夹用来保存损失函数
#总共两组lossfunction
#第一组：参数上的lossfunction(直接用参数相减做差值)


#第二组：带回方程里做参数检验，和之前的物理信息做差值比较
#第二组PDE传值是传出pred_a0这个值，传到MATLAB中，在Data_base_compare存储（思考到也许不止一个epoch)
#然后需要和原来的Data_base_input做对照

import torch
import numpy as np
import matlab.engine
import os

def calculate_loss(predicted_params, actual_params, data):

    # 第一部分：计算参数上的损失函数
    # 只计算a0的损失，否则太多存在泛化误差 
    # 即使是对称的也取绝对值
    actual_params_a0 = abs(actual_params[0,0])
    param_loss = torch.sum((predicted_params - actual_params_a0) ** 2)
    
    physical_info_param = predicted_params[0, 0].item()
    print('实际物理值为',actual_params_a0)
    print('预期物理值为',physical_info_param)
    # 第二部分：带回方程里做参数检验
    # 启动MATLAB引擎
    eng = matlab.engine.start_matlab()

    # 调用MATLAB函数
    eng.PDE_constraints(float(physical_info_param), nargout=0)


    # 定义基础文件夹路径
    base_folder_name = 'Data_base_compare'
    # 构建文件路径，不再添加子文件夹
    gm_data_path = os.path.join(base_folder_name, f'data_compare_a0={physical_info_param:.1f}.txt')

    # 检查文件是否存在
    if not os.path.exists(gm_data_path):
        raise FileNotFoundError(f'MATLAB output file not found: {gm_data_path}')

    ## 读取MATLAB生成的数据文件
    #folder_name = f'Data_base_compare/a0={physical_info_param:.1f}'
    #gm_data_path = os.path.join(folder_name, f'data_compare_a0={physical_info_param:.1f}.txt')
    #if not os.path.exists(gm_data_path):
    #    raise FileNotFoundError(f'MATLAB output file not found: {gm_data_path}')

    gm_data = np.loadtxt(gm_data_path)
    #需要转化为torch形式
    gm_data_tensor = torch.tensor(gm_data, dtype=torch.float32)
    # 对每列做差值平方求和
    physical_loss = torch.sum((data - gm_data_tensor) ** 2)

    # 总损失
    total_loss = param_loss + physical_loss

    # 关闭MATLAB引擎
    eng.quit()
    
    return total_loss

# 示例用法



