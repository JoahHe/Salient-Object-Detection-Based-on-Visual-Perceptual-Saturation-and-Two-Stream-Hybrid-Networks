function Ipn=linyu(I,t)
%input:I--输入图像,t--像素位置
%output: Ipn---输出一个邻域分量图像
[r,c,d]=size(I);
    if t<=d
        Ipn(:,:)=I(:,:,t);%像素点的颜色信息
    elseif t>d&&t<=2*d 
        Ipn=I(:,:,t-d); %采样像素赋值为原图，以便于大部分像素的位置改变后，没有改变的像素点已被填满
        Ipn(1:r-1,1:c-1)=I(2:r,2:c,t-d);%颜色信息%(i+1,j+1)点% 逆时针从i+1，j+1开始，考虑邻域像素颜色
    elseif t>2*d&&t<=3*d
        Ipn=I(:,:,t-2*d); %
        Ipn(1:r,1:c-1)=I(1:r,2:c,t-2*d);%颜色信息%(i,j+1)点
    elseif t>3*d&&t<=4*d
        Ipn=I(:,:,t-3*d); %        
        Ipn(2:r,1:c-1)=I(1:r-1,2:c,t-3*d);%颜色信息%(i-1,j+1)点
    elseif t>4*d&&t<=5*d
        Ipn=I(:,:,t-4*d); %        
        Ipn(2:r,1:c)=I(1:r-1,1:c,t-4*d);%颜色信息%(i-1,j)点
    elseif t>5*d&&t<=6*d
        Ipn=I(:,:,t-5*d); %        
        Ipn(2:r,2:c)=I(1:r-1,1:c-1,t-5*d);%颜色信息%(i-1,j-1)点
    elseif t>6*d&&t<=7*d
        Ipn=I(:,:,t-6*d); %        
        Ipn(1:r,2:c)=I(1:r,1:c-1,t-6*d);%颜色信息%(i,j-1)点
    elseif t>7*d&&t<=8*d
        Ipn=I(:,:,t-7*d); %        
        Ipn(1:r-1,2:c)=I(2:r,1:c-1,t-7*d);%颜色信息%(i+1,j-1)点
    elseif t>8*d&&t<=9*d
        Ipn=I(:,:,t-8*d); %        
        Ipn(1:r-1,1:c)=I(2:r,1:c,t-8*d);%颜色信息%(i+1,j)点
    end