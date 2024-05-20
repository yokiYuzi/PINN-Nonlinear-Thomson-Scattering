 function fig1b
% This is for calculateing electron's trajectory and energy gain.
global  a0 L b0 u0 zf del p0
del=1;%偏振参数，激光的偏振态，p=0线偏振，p=1圆偏振，p=0~1椭圆偏振
p0=0;  %激光初相位
a0=-7;%归一化的激光振幅，控制光强（<1为非相对论领域，>1为相对论领域）
L=2*pi*10; %激光脉宽 2*pi=3.3fs
b0=2*pi*5;%z=0时的激光束腰半径

zf=b0^2/2*1;%对应的瑞利长度
z0=0;  %可以改变对撞中心（电子初始的z轴坐标）
gm0=1;%电子初始能量
u0=-(1-gm0^-2)^.5; %光速归一化的电子速度
yyi=[0 0 z0 0 0 u0 gm0];
tspan=[0 210*L];%计时的时间 从z0开始计时
options = odeset('RelTol',1e-13,'AbsTol',[1e-13 1e-13 1e-13 1e-13 1e-13 1e-13 1e-13]);
[t,yy]=ode45(@lac,tspan,yyi,options);
x1=yy(:,1); x2=yy(:,2);x3=yy(:,3);
u1=yy(:,4);u2=yy(:,5); u3=yy(:,6);
Gm=yy(:,7)-1;
Eta=x3-t;
aL=a0*exp(-(x1.^2+x2.^2)/b0^2./(1+x3.^2/zf^2)-Eta.^2/L^2)./sqrt(1+x3.^2/zf^2);
r=(x1.^2+x2.^2).^0.5;
v=(u1.^2+u2.^2).^0.5;

xx=[x1,x2,x3];
uu=[u1,u2,u3];
du=diff(uu);
dt=diff(t);
dudt=[du(:,1)./dt,du(:,2)./dt,du(:,3)./dt];
xx(end,:)=[];
uu(end,:)=[];
t(end)=[];

save Gm.txt Gm -ASCII;
save x.txt xx -ASCII;
save u.txt uu -ASCII;
save du.txt du -ASCII;
save dt.txt dt -ASCII;
save dudt.txt dudt -ASCII;
save t.txt t -ASCII;

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