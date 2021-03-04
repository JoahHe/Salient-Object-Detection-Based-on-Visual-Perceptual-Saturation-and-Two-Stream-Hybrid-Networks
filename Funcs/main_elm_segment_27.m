function BW=main_elm_segment_27(I,Ip,NumberofHiddenNeurons,ActivationFunction,type,s_num)

%input:彩色原图I，Ip-正样本二值图，Map-显著图 
%NumberofHiddenNeurons,ActivationFunction——elm隐节点数，激活函数
%output:BW--分割结果的二值图

[r,c,d]=size(I);

kbn=~Ip;%%负样本候选区
kbp=Ip;%logical(Ig_bw.*Ip);%Ig_bw.*Ip;%正样本候选区

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

for t=1:kd%
    Ipn=linyu(I,t);
    sample_pt(:,t)=Ipn(kbp);%();%取注视区域的像素做正样本
    sample_nt(:,t)=Ipn(kbn);%外围区域，随机像素做负样本
end

sample_p=sample_pt(pp(1:num_p),:);%随机选择 num_p 个正样本
sample_n=sample_nt(pn(1:num_n),:);%随机选择 num_n 个负样本

train_label=cat(1,1*ones(num_p,1),2*ones(num_n,1));%
train_data=[sample_p;sample_n];

for t=1:kd%
    I_=linyu(I,t);
    test_data(:,t)=I_(:);
end

   [InputWeight,OutputWeight,BiasofHiddenNeurons,TrainingTime] = elm_train(train_label,train_data,NumberofHiddenNeurons,ActivationFunction);
   [predict_label, TestingTime] = elm_predict(test_data, InputWeight,OutputWeight,NumberofHiddenNeurons, BiasofHiddenNeurons,ActivationFunction);

BWout=predict_label<2;%如果矩阵元素<2,就标记为1，否则为0. 得到目标像素的类别向量
BW=reshape(BWout,[r,c]); %变为一个目标位置为1,其余位置为0的逻辑矩阵
end
 