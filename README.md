为了更好的解决传统负载上的计算问题，本研究提出利用神经网络对非线性PDE方程进行训练和模拟，搭建逆物理信息神经网络（I-PINNs）。通过在选定数据范围内的依照同尺度分割的物理信息数据，本研究搭建了泛化性能高，准确性好，能够有效的通过读入物理信息数据（如电子在空间中的运动轨迹数据、电子速度数据等），推断出存在的非线性物理边界条件。在计算完成后能够有效的解决计算负载问题和非线性模拟问题。从而在特定参数的功能上替代传统的网格计算模拟方法，并且极大的降低计算功耗


![image](https://github.com/yokiYuzi/PINN-Nonlinear-Thomson-Scattering/assets/76743561/81391291-a0f4-44b3-9095-e66d8166581c)
I-PINN逆向物理信息数据输入神经网路流程图

![image](https://github.com/yokiYuzi/PINN-Nonlinear-Thomson-Scattering/assets/76743561/33027328-d065-46dc-95d8-617d6c108ba8)


![image](https://github.com/yokiYuzi/PINN-Nonlinear-Thomson-Scattering/assets/76743561/9c6aff07-59cf-45cd-8dc3-ec7095f610f8)
I-PINN中损失函数计算流程以及泛化优化流程

![image](https://github.com/yokiYuzi/PINN-Nonlinear-Thomson-Scattering/assets/76743561/de5459f0-23b3-422f-b749-a0864529ca6c)
I-PINN逆物理信息神经网路epoch中损失函数值

![image](https://github.com/yokiYuzi/PINN-Nonlinear-Thomson-Scattering/assets/76743561/7e0aa8f4-14a0-4b13-8dc4-ecf786fec17d)

![image](https://github.com/yokiYuzi/PINN-Nonlinear-Thomson-Scattering/assets/76743561/9404c7a4-da27-44d8-a6c9-c245cc5c72cc)

![image](https://github.com/yokiYuzi/PINN-Nonlinear-Thomson-Scattering/assets/76743561/f4acc344-db6d-4e91-8ac8-15ea50ecb349)

![image](https://github.com/yokiYuzi/PINN-Nonlinear-Thomson-Scattering/assets/76743561/83a47146-832c-49fb-8014-c95deb272fe9)

在收敛模型中的逆向测试，其中（a）对照为：实际a0=0.5 预计a0=0.75 (b) 对照为：实际a0=5 预计a0=7.54 (c) 对照为：实际a0=4 预计a0=11 .799(d) 对照为：实际a0=10 预计a0=20.085
