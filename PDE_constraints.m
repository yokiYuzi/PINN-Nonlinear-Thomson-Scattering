function PDE_constraints1(a0)
    % 设置全局变量
    global L b0 u0 zf del p0 z0 gm0;

    % 设置初始参数
    del = 1;
    p0 = 0;
    L = 2 * pi * 10;
    b0 = 2 * pi * 5;
    zf = b0^2 / 2;
    z0 = 0;
    gm0 = 1;
    u0 = -(1 - gm0^-2)^0.5;

    % 调用仿真函数并接收返回值
    [Gm, xx, uu, du, dt, dudt, t] = fig1b(a0);

    % 删除Gm数据的第一行
    Gm(1, :) = [];

    % 创建和检查文件夹
    folder_name = sprintf('Data_base_compare/a0=%.1f', a0);
    if ~exist(folder_name, 'dir')
        mkdir(folder_name);
    end

    % 保存数据，文件名包含当前的a0值
    save(fullfile(folder_name, 'Gm.txt'), 'Gm', '-ascii');
    save(fullfile(folder_name, 'x.txt'), 'xx', '-ascii');
    save(fullfile(folder_name, 'u.txt'), 'uu', '-ascii');
    save(fullfile(folder_name, 'du.txt'), 'du', '-ascii');
    save(fullfile(folder_name, 'dt.txt'), 'dt', '-ascii');
    save(fullfile(folder_name, 'dudt.txt'), 'dudt', '-ascii');
    save(fullfile(folder_name, 't.txt'), 't', '-ascii');

      % 保存单独数据
    file_names = {'t.txt', 'x.txt', 'u.txt', 'dt.txt', 'du.txt', 'dudt.txt', 'Gm.txt'};
    data_vars = {t, xx, uu, dt, du, dudt, Gm};
    for i = 1:length(file_names)
    current_data = data_vars{i};
    save(fullfile(folder_name, file_names{i}), 'current_data', '-ascii');
    end

    % 读取数据并合并
    combined_data = [];
    for i = 1:length(file_names)
        temp_data = load(fullfile(folder_name, file_names{i}));
        combined_data = [combined_data, temp_data];  % 横向连接矩阵
    end
    % 保存合并后的数据到上一级文件夹
    combined_filename = fullfile('Data_base_compare', sprintf('data_compare_a0=%.1f.txt', a0));
    save(combined_filename, 'combined_data', '-ascii');

    % 绘图并保存图片
    figure; hold on;
    plot3(xx(:, 3) / (2 * pi), xx(:, 1) / (2 * pi), xx(:, 2) / (2 * pi), 'k', 'LineWidth', 2);
    xlabel('z/\lambda_0'); ylabel('y/\lambda_0'); zlabel('x/\lambda_0');
    set(gca, 'LineWidth', 1, 'FontSize', 18);
    view(-45, 10);
    image_name = fullfile(folder_name, sprintf('a0=%.1f.fig', a0));
    savefig(image_name); % 使用 savefig 来保存 .fig 文件
    close(gcf);
end

function [Gm, xx, uu, du, dt, dudt, t] = fig1b(a0)
    % 仿真函数，计算电子的轨迹和能量增益
    global a0 L b0 u0 zf del p0 z0 gm0
    yyi = [0 0 z0 0 0 u0 gm0];

    num_points = 2000; % 固定步数
    tspan = linspace(0, 210 * L, num_points); % 计时的时间 从z0开始计时

    options = odeset('RelTol', 1e-13, 'AbsTol', repmat(1e-13, 1, 7));
    [t, yy] = ode45(@lac, tspan, yyi, options);

    % 提取结果
    x1 = yy(:, 1); x2 = yy(:, 2); x3 = yy(:, 3);
    u1 = yy(:, 4); u2 = yy(:, 5); u3 = yy(:, 6);
    Gm = yy(:, 7) - 1;
    xx = [x1, x2, x3];
    uu = [u1, u2, u3];
    du = diff(uu);
    dt = diff(t);
    dudt = [du(:, 1) ./ dt, du(:, 2) ./ dt, du(:, 3) ./ dt];
    xx(end, :) = [];
    uu(end, :) = [];
    t(end) = [];
end

function dy=lac(x,y)
global a0 L b0 u0 zf del p0
dy=zeros(7,1);
eta=y(3)-x+5*L;
pz=atan(y(3)/zf);
b=b0*(1+y(3)^2/zf^2)^.5;
rz=y(3)/(zf^2+y(3)^2);
phi=y(3)-x-pz-p0+(y(1)^2+y(2)^2)/2*rz;
pth=y(3)-x-2*pz-p0+(y(1)^2+y(2)^2)/2*rz+pi;
al=a0*exp(-(y(1)^2+y(2)^2)/b0^2/(1+y(3)^2/zf^2)-eta.^2/L^2)/sqrt(1+y(3)^2/zf^2);
ax=al*cos(phi);ay=del*al*sin(phi);
x1=2*y(1)/(b0*b);x2=2*y(2)/(b0*b);
az=-al*sin(pth)*x1-del*al*cos(pth)*x2;

ptal=al*2*eta/L^2;
pxal=al*(-2*y(1)/b^2);pyal=al*(-2*y(2)/b^2);
pzb=b0*y(3)/zf^2*(1+y(3)^2/zf^2)^-.5;
pzal=-al*pzb/b+al*2*(y(1)^2+y(2)^2)/b^3*pzb+al*(-2*eta/L^2);
ptphi=-1;pxphi=y(1)*rz;pyphi=y(2)*rz;
pzphi=1-zf/(y(3)^2+zf^2)-(y(1)^2+y(2)^2)/2*(y(3)^2-zf^2)/(y(3)^2+zf^2)^2;

ptax=cos(phi)*ptal-sin(phi)*al*ptphi;
ptay=-del*(sin(phi)*ptal+al*cos(phi)*ptphi);
pyax=cos(phi)*pyal-sin(phi)*al*pyphi;
pxay=-del*(sin(phi)*pxal+cos(phi)*al*pxphi);
pzax=cos(phi)*pzal-sin(phi)*al*pzphi;
pzay=-del*(sin(phi)*pzal+al*cos(phi)*pzphi);
pxaz=-(2/(b0*b)*al*sin(pth)+x1*pxal*sin(pth)+x1*al*cos(pth)*pxphi...
    +del*(x2*pxal*cos(pth)+x2*al*(-sin(pth))*pxphi));
pyaz=-(x1*pyal*sin(pth)+x1*al*cos(pth)*pyphi...
    +del*(2/(b0*b)*al*cos(pth)+x2*pyal*cos(pth)+x2*al*(-sin(pth))*pyphi));
ptaz=-(x1*al*cos(pth)*ptphi+del*x2*al*(-sin(pth))*ptphi...
    +x1*ptal*sin(pth)+del*x2*ptal*cos(pth));

dy(1)=y(4);
dy(2)=y(5);
dy(3)=y(6);
dy(4)=((1-y(4)^2)*ptax+y(5)*(pyax-pxay)+y(6)*(pzax-pxaz)-y(4)*y(5)*ptay-y(4)*y(6)*ptaz)/y(7);
dy(5)=((1-y(5)^2)*ptay+y(4)*(pxay-pyax)+y(6)*(pzay-pyaz)-y(4)*y(5)*ptax-y(5)*y(6)*ptaz)/y(7);
dy(6)=((1-y(6)^2)*ptaz+y(4)*(pxaz-pzax)+y(5)*(pyaz-pzay)-y(4)*y(6)*ptax-y(5)*y(6)*ptay)/y(7);
dy(7)=y(4)*ptax+y(5)*ptay+y(6)*ptaz;
end

