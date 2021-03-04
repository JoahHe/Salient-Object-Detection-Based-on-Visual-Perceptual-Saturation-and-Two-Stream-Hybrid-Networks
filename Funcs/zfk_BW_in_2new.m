function [BWout, smp] = zfk_BW_in_2new(srcImg,smp1,smp2,alf)
%% 双通道zfk  感知饱和的显著目标检测，输入初始双显著图，输出目标二值图和新显著图
%% 双通道zfk融合，初始kk=2时比较bw1与bw2,随后等同于单通道基于注视区的zfk,直至饱和(选择两通道中最可靠通道结果做正反馈)
%选合适的图像
    sp_img = srcImg;
    [r2, c2, ~]=size(sp_img);%原大图sp_img
    
    smp10 = imresize(smp1,[r2 c2]);%尺寸与BW一致,准备做正反馈
    smp20 = imresize(smp2,[r2 c2]);    
    bw1= imbinarize(smp10);%二值化，得到目标区域             
    bw1 = select_max_region(bw1,5);%二值化图中，选最大面积的2个连通域做感兴趣的目标
    bw2= imbinarize(smp20);%二值化，得到目标区域             
    bw2 = select_max_region(bw2,5);   
    
    %alf=0.1;%0.5;%two-supervised%unsupervised+supervised加权系数
    sMap = mat2gray(alf*smp10 + (1-alf)*smp20);%
    smp1= imresize(sMap,[r2 c2]);
    
    if (all(bw1(:)==1)) %排除bottom-up通道全部都是1,无背景像素的情况            
           bw3=bw2;
           bw4=bw1;
           bw5=~bw2;
    elseif (all(bw2(:)==0)) %排除top-down通道全部都是0,无目标像素情况
           bw3=bw1;
           bw4=~bw2;
           bw5=~bw1;
    else  
           bw3=and(bw1,bw2);%bw1与bw2交集,1正样本的感知饱和区域mask，0-其他
           bw4=xor(bw1,bw2);%待处理区域mask，1为mask,0为饱和区
           bw5=~or(bw1,bw2);%负样本的饱和区mask，1为负样本饱和区mask,0-其他
    end 
    BW=main_elm_segment_271(sp_img,bw3,bw4,bw5,smp10,smp20,40,'sig','elm',1000);%20201213改动了,smp1,smp2直接进入PELM学习建模，不用alf加权
    BW=or(and(bw4,BW),bw3);%饱和区mask+待处理区域mask => 生成更新的目标区
    BW=select_max_region(BW,5);
    %kk=1;
    %BW_t(:,:,kk)=BW;%BW_t时间脉冲序列   
    %smp1=mat2gray(smp1+BW);%加上效果差
    [BWout, smp] = zfk_BW_in(sp_img,BW,smp1);%单通道,20201213改动，不加这个zfk效果差。加上效果最好。
        %{ 
        %figure;    
        subplot(161);imshow(bw1,[]);title('RBD');
        subplot(162);imshow(bw2,[]);title('PicaNet');
        subplot(163);imshow(bw3,[]);title('Positive Mask');
        subplot(164);imshow(bw4,[]);title('Region need be refined');
        subplot(165);imshow(bw5,[]);title('Negative Mask');
        subplot(166);imshow(BW,[]);title('New salient object');
        %pause
        %}
end

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
end


function BW=main_elm_segment_271(I,bw3,bw4,bw5,smp1,smp2,NumberofHiddenNeurons,ActivationFunction,type,s_num)
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

kbn=bw5;%~Ip;%%负样本候选区
kbp=bw3;%logical(Ip);%正样本候选区%

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

for t=1:kd+3%5%3%+60
    if t<=kd
        Ip=linyu(I,t);%I(:,:,t);%颜色信息
        In=linyu(I,t);%I(:,:,t);
    elseif t==kd+1      
        Ip=Ig;%smp;  %梯度信息
        In=Ig;%smp;     
    elseif t==kd+2      
        Ip=smp1;    %显著度信息，
        In=smp1;
    elseif t>=kd+3 %
        Ip=smp2;%spimg(:,:,t-(kd+3)+1);%超像素信息
        In=smp2;%spimg(:,:,t-(kd+3)+1);%
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
for k=1:kd+3%5%1%3%+6
    if k<=kd
        I_=linyu(I,k);%I_=I(:,:,k);
        test_data(:,k)=I_(:);
    elseif k==kd+1
        test_data(:,k)=Ig(:);%smp(:);
    elseif k==kd+2
        test_data(:,k)=smp1(:);
    elseif k>=kd+3
        %spimg_gray = spimg(:,:,k-(kd+3)+1);
        test_data(:,k)=smp2(:);%spimg_gray(:);%7
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
end

