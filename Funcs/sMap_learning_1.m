function [smap,BW]=sMap_learning_1(I,BW,smp,times)
%基于学习的显著性检测,spimg
%首先利用fft获得全局显著性，索引显著度，获得最大显著度的前几个注视点，分别以这些注视点为中心构建熵最大窗口，
%获得正负样本训练得到分类模型，利用分类模型分类像素获得目标（局部感知）
%叠加相关结果获得新的显著度图（结合了全局和局部感知）
%Input:I-彩色图像,nn--显著点数量
%Output:smap-显著度图

[r,c,d]=size(I);
sum=zeros(r,c);%存储单分类器分割结果相加的和

for k=1:times  %微跳视次数
 
    %BW = elm_segment(I,BW,smp,spimg,80,'sig','elm',800); %0.87113
    BW =elm_segment_27(I,BW,smp,40,'sig','elm',1000);%,spimg
    BW=select_max_region(BW,5);%
    %BW=imfill(BW,'holes'); %if delete it: F_score:0.87286
    sum=sum+BW;
end
smap=sum;%

function BW=elm_segment(I,Ip,smp,spimg,NumberofHiddenNeurons,ActivationFunction,type,s_num)

%input:彩色原图I，Ip-正样本二值图，Map-显著图 
%NumberofHiddenNeurons,ActivationFunction——elm隐节点数，激活函数
%output:BW--分割结果的二值图

[r,c,d]=size(I);

kbn=~Ip;%%负样本候选区
kbp=logical(Ip);%F_score:0.83156,R_score:0.71578,P_score:0.9009

Np=sum(kbp(:)==1);%计算框中可选像素数目
Nn=sum(kbn(:)==1);%计算框外可选像素数目

kd=d;
num_p=s_num;%1000;%框内随机选100个正样本
num_n=s_num;%1001;%框外随机选择110个负样本
if num_p>Np
    num_p=Np;%防止抽样数超过实际像素数，导致出错
end
if num_n>Nn
    num_n=Nn;%防止抽样数超过实际像素数，导致出错
end
pp=randperm(Np);
pn=randperm(Nn);

for t=1:kd+4
    if t<=kd
        Ip=I(:,:,t);%颜色信息
        In=I(:,:,t);
    elseif t==kd+1      
        Ip=smp;%显著图信息
        In=smp;%
    elseif t>=kd+2 %超像素颜色
        Ip=spimg(:,:,t-(kd+2)+1);
        In=spimg(:,:,t-(kd+2)+1);%
    %elseif t==kd+3       %梯度1
    %    Ip=Ig;
    %    In=Ig;%梯度 
    %elseif t==kd+4       %显著度信息
    %    Ip=Map;
    %    In=Map;
    elseif t>=kd+2       %hsv
        %Ip=Ihsv(:,:,t-4);
        %In=Ihsv(:,:,t-4);%
        Ip=Ilab(:,:,t-4);
        In=Ilab(:,:,t-4);%
        %Ip=Iycbcr(:,:,t-4);
        %In=Iycbcr(:,:,t-4);%
    %elseif t>=kd+3      
    %    Ip=Igabor(:,:,t-kd-2);%gabor纹理信息
    %    In=Igabor(:,:,t-kd-2);    
    end
    sample_pt(:,t)=Ip(kbp);%();%取注视区域的像素做正样本
    sample_nt(:,t)=In(kbn);%外围区域，随机像素做负样本
end

sample_p=sample_pt(pp(1:num_p),:);%随机选择 num_p 个正样本
sample_n=sample_nt(pn(1:num_n),:);%随机选择 num_n 个负样本
train_label=cat(1,1*ones(num_p,1),2*ones(num_n,1));
train_data=[sample_p;sample_n];
for k=1:kd+4
    if k<=kd
        I_=I(:,:,k);
        test_data(:,k)=I_(:);
    elseif k==kd+1
        test_data(:,k)=smp(:);
    elseif k>=kd+2
        spimg_gray = spimg(:,:,k-(kd+2)+1);
        test_data(:,k)=spimg_gray(:);%7
    %elseif k==kd+3
    %    test_data(:,k)=Ig(:);%6
    %elseif k==kd+4
    %    test_data(:,k)=Map(:);
    elseif k>=kd+2
        %temp=Ihsv(:,:,k-4);
        temp=Ilab(:,:,k-4);
        %temp=Iycbcr(:,:,k-4);
        test_data(:,k)=temp(:);%8-10
    %elseif k>=kd+3
    %    I_gabor=Igabor(:,:,k-kd-2);%gabor纹理信息
    %    test_data(:,k)=I_gabor(:); 
    end
end
if type=='elm'
   [InputWeight,OutputWeight,BiasofHiddenNeurons,TrainingTime] = elm_train(train_label,train_data,NumberofHiddenNeurons,ActivationFunction);
   [predict_label, TestingTime] = elm_predict(test_data, InputWeight,OutputWeight,NumberofHiddenNeurons, BiasofHiddenNeurons,ActivationFunction);
else %type=='svm'
   [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = PolySVC(train_data', train_label',1);
   %gamma=10; C=100;%3d,5d原来的参数值   gamma=0.005; C=100;
   %[AlphaY, SVs, Bias, Parameters, nSV, nLabel] = RbfSVC(train_data', train_label',gamma, C);
   [predict_label, DecisionValue]= SVMClass(test_data', AlphaY, SVs, Bias, Parameters, nSV, nLabel);
end
BWout=predict_label<2;%如果矩阵元素<2,就标记为1，否则为0. 得到目标像素的类别向量
BW=reshape(BWout,[r,c]); %变为一个目标位置为1,其余位置为0的逻辑矩阵

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

function BW=elm_segment_27(I,Ip,smp,NumberofHiddenNeurons,ActivationFunction,type,s_num)
%input:彩色原图I，Ip-正样本二值图，Map-显著图 ,spimg
%NumberofHiddenNeurons,ActivationFunction——elm隐节点数，激活函数
%output:BW--分割结果的二值图
[r,c,d]=size(I);
if d>1
    I1=rgb2gray(I);
else
    I1=I;
end
[xx,yy]= gradient(I1);%edge(I,'log');%edge(I,'canny');%求梯度图
Ig = sqrt(xx.^2+yy.^2);

%th=1/(r*c)*sum(abs(Ig(:)));%设定一个梯度阈值，以便object采样限制在高梯度区域，减少干扰;背景采样仍是均匀随机
%Ig_bw=im2bw(Ig,th);%>平均梯度的高梯度二值区域

kbn=~Ip;%%负样本候选区
kbp=logical(Ip);%正样本候选区%

Np=sum(kbp(:)==1);%计算框中可选像素数目
Nn=sum(kbn(:)==1);%计算框外可选像素数目

kd=d*9;
num_p=s_num;%1000;%框内随机选100个正样本
num_n=s_num;%1001;%框外随机选择110个负样本
if num_p>Np
    num_p=Np;%防止抽样数超过实际像素数，导致出错
end
if num_n>Nn
    num_n=Nn;%防止抽样数超过实际像素数，导致出错
end
pp=randperm(Np);
pn=randperm(Nn);

for t=1:kd+2%5%3%+60
    if t<=kd
        Ip=linyu(I,t);%I(:,:,t);%颜色信息
        In=linyu(I,t);%I(:,:,t);
    elseif t==kd+1      
        Ip=Ig;%smp;  %梯度信息
        In=Ig;%smp;     
    elseif t==kd+2      
        Ip=smp;    %显著度信息，
        In=smp;
    elseif t>=kd+3 %
        Ip=spimg(:,:,t-(kd+3)+1);%超像素信息
        In=spimg(:,:,t-(kd+3)+1);%
    %elseif t>=kd+3      
    %     Ip=Igabor(:,:,t-kd-2);%gabor纹理信息
    %    In=Igabor(:,:,t-kd-2);   
    elseif t>=kd+6%1       %hsv
        %Ip=Ihsv(:,:,t-4);
        %In=Ihsv(:,:,t-4);%
        Ip=Ilab(:,:,t-4);%-28
        In=Ilab(:,:,t-4);%-28
    end
    sample_pt(:,t)=Ip(kbp);%();%取注视区域的像素做正样本
    sample_nt(:,t)=In(kbn);%外围区域，随机像素做负样本
end

sample_p=sample_pt(pp(1:num_p),:);%随机选择 num_p 个正样本
sample_n=sample_nt(pn(1:num_n),:);%随机选择 num_n 个负样本
train_label=cat(1,1*ones(num_p,1),2*ones(num_n,1));
train_data=[sample_p;sample_n];
for k=1:kd+2%5%1%3%+6
    if k<=kd
        I_=linyu(I,k);%I_=I(:,:,k);
        test_data(:,k)=I_(:);
    elseif k==kd+1
        test_data(:,k)=Ig(:);%smp(:);
    elseif k==kd+2
        test_data(:,k)=smp(:);
    elseif k>=kd+3
        spimg_gray = spimg(:,:,k-(kd+3)+1);
        test_data(:,k)=spimg_gray(:);%7
    %elseif k==kd+2
    %    test_data(:,k)=Map(:);
    elseif k>=kd+5%1
        %temp=Ihsv(:,:,k-4);
        temp=Ilab(:,:,k-4);%-28
        %temp=Iycbcr(:,:,k-4);
        test_data(:,k)=temp(:);%8-10
    %elseif k>=kd+3
    %    I_gabor=Igabor(:,:,k-kd-2);%gabor纹理信息
    %    test_data(:,k)=I_gabor(:);    
    end
end

switch type
    case 'elm'
            [InputWeight,OutputWeight,BiasofHiddenNeurons,TrainingTime] = elm_train(train_label,train_data,NumberofHiddenNeurons,ActivationFunction);
            [predict_label, TestingTime] = elm_predict(test_data, InputWeight,OutputWeight,NumberofHiddenNeurons, BiasofHiddenNeurons,ActivationFunction);
    case 'svm'
            [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = PolySVC(train_data', train_label',1);
            %   %gamma=1; C=100;%3d,5d原来的参数值   gamma=0.005; C=100;
            %   %[AlphaY, SVs, Bias, Parameters, nSV, nLabel] = RbfSVC(train_data', train_label',gamma, C);
            [predict_label, DecisionValue]= SVMClass(test_data', AlphaY, SVs, Bias, Parameters, nSV, nLabel);
    case 'cs'
        cs=1;
        repeat=1;
        type='general';%
        % Train.X - training samples
        % Train.y - training labels
        % Test.X - testing samples
        % Test.y - testing labels
        Train_data.X=train_data';
        Train_data.y=train_label';
        Test_data.X=test_data';
        Test_data.y=[];%
        [predict_label] = NSC(Train_data, Test_data, cs, repeat,type);%        
end
BWout=predict_label<2;%如果矩阵元素<2,就标记为1，否则为0. 得到目标像素的类别向量
BW=reshape(BWout,[r,c]); %变为一个目标位置为1,其余位置为0的逻辑矩阵
