function BW=main_elm_segment_27(I,Ip,NumberofHiddenNeurons,ActivationFunction,type,s_num)

%input:��ɫԭͼI��Ip-��������ֵͼ��Map-����ͼ 
%NumberofHiddenNeurons,ActivationFunction����elm���ڵ����������
%output:BW--�ָ����Ķ�ֵͼ

[r,c,d]=size(I);

kbn=~Ip;%%��������ѡ��
kbp=Ip;%logical(Ig_bw.*Ip);%Ig_bw.*Ip;%��������ѡ��

Np=sum(kbp(:)==1);%������п�ѡ������Ŀ
Nn=sum(kbn(:)==1);%��������ѡ������Ŀ

kd=d*9;
num_p=s_num;%1000;%�������ѡ100��������
num_n=s_num;%1001;%�������ѡ��110��������
if num_p>Np
    num_p=Np;%��ֹ����������ʵ�������������³���
end
if num_n>Nn
    num_n=Nn;%��ֹ����������ʵ�������������³���
end
pp=randperm(Np);
pn=randperm(Nn);

for t=1:kd%
    Ipn=linyu(I,t);
    sample_pt(:,t)=Ipn(kbp);%();%ȡע�������������������
    sample_nt(:,t)=Ipn(kbn);%��Χ�������������������
end

sample_p=sample_pt(pp(1:num_p),:);%���ѡ�� num_p ��������
sample_n=sample_nt(pn(1:num_n),:);%���ѡ�� num_n ��������

train_label=cat(1,1*ones(num_p,1),2*ones(num_n,1));%
train_data=[sample_p;sample_n];

for t=1:kd%
    I_=linyu(I,t);
    test_data(:,t)=I_(:);
end

   [InputWeight,OutputWeight,BiasofHiddenNeurons,TrainingTime] = elm_train(train_label,train_data,NumberofHiddenNeurons,ActivationFunction);
   [predict_label, TestingTime] = elm_predict(test_data, InputWeight,OutputWeight,NumberofHiddenNeurons, BiasofHiddenNeurons,ActivationFunction);

BWout=predict_label<2;%�������Ԫ��<2,�ͱ��Ϊ1������Ϊ0. �õ�Ŀ�����ص��������
BW=reshape(BWout,[r,c]); %��Ϊһ��Ŀ��λ��Ϊ1,����λ��Ϊ0���߼�����
end
 