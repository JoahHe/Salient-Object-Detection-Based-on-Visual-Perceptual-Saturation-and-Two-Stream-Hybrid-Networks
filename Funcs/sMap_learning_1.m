function [smap,BW]=sMap_learning_1(I,BW,smp,times)
%����ѧϰ�������Լ��,spimg
%��������fft���ȫ�������ԣ����������ȣ������������ȵ�ǰ����ע�ӵ㣬�ֱ�����Щע�ӵ�Ϊ���Ĺ�������󴰿ڣ�
%�����������ѵ���õ�����ģ�ͣ����÷���ģ�ͷ������ػ��Ŀ�꣨�ֲ���֪��
%������ؽ������µ�������ͼ�������ȫ�ֺ;ֲ���֪��
%Input:I-��ɫͼ��,nn--����������
%Output:smap-������ͼ

[r,c,d]=size(I);
sum=zeros(r,c);%�洢���������ָ�����ӵĺ�

for k=1:times  %΢���Ӵ���
 
    %BW = elm_segment(I,BW,smp,spimg,80,'sig','elm',800); %0.87113
    BW =elm_segment_27(I,BW,smp,40,'sig','elm',1000);%,spimg
    BW=select_max_region(BW,5);%
    %BW=imfill(BW,'holes'); %if delete it: F_score:0.87286
    sum=sum+BW;
end
smap=sum;%

function BW=elm_segment(I,Ip,smp,spimg,NumberofHiddenNeurons,ActivationFunction,type,s_num)

%input:��ɫԭͼI��Ip-��������ֵͼ��Map-����ͼ 
%NumberofHiddenNeurons,ActivationFunction����elm���ڵ����������
%output:BW--�ָ����Ķ�ֵͼ

[r,c,d]=size(I);

kbn=~Ip;%%��������ѡ��
kbp=logical(Ip);%F_score:0.83156,R_score:0.71578,P_score:0.9009

Np=sum(kbp(:)==1);%������п�ѡ������Ŀ
Nn=sum(kbn(:)==1);%��������ѡ������Ŀ

kd=d;
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

for t=1:kd+4
    if t<=kd
        Ip=I(:,:,t);%��ɫ��Ϣ
        In=I(:,:,t);
    elseif t==kd+1      
        Ip=smp;%����ͼ��Ϣ
        In=smp;%
    elseif t>=kd+2 %��������ɫ
        Ip=spimg(:,:,t-(kd+2)+1);
        In=spimg(:,:,t-(kd+2)+1);%
    %elseif t==kd+3       %�ݶ�1
    %    Ip=Ig;
    %    In=Ig;%�ݶ� 
    %elseif t==kd+4       %��������Ϣ
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
    %    Ip=Igabor(:,:,t-kd-2);%gabor������Ϣ
    %    In=Igabor(:,:,t-kd-2);    
    end
    sample_pt(:,t)=Ip(kbp);%();%ȡע�������������������
    sample_nt(:,t)=In(kbn);%��Χ�������������������
end

sample_p=sample_pt(pp(1:num_p),:);%���ѡ�� num_p ��������
sample_n=sample_nt(pn(1:num_n),:);%���ѡ�� num_n ��������
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
    %    I_gabor=Igabor(:,:,k-kd-2);%gabor������Ϣ
    %    test_data(:,k)=I_gabor(:); 
    end
end
if type=='elm'
   [InputWeight,OutputWeight,BiasofHiddenNeurons,TrainingTime] = elm_train(train_label,train_data,NumberofHiddenNeurons,ActivationFunction);
   [predict_label, TestingTime] = elm_predict(test_data, InputWeight,OutputWeight,NumberofHiddenNeurons, BiasofHiddenNeurons,ActivationFunction);
else %type=='svm'
   [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = PolySVC(train_data', train_label',1);
   %gamma=10; C=100;%3d,5dԭ���Ĳ���ֵ   gamma=0.005; C=100;
   %[AlphaY, SVs, Bias, Parameters, nSV, nLabel] = RbfSVC(train_data', train_label',gamma, C);
   [predict_label, DecisionValue]= SVMClass(test_data', AlphaY, SVs, Bias, Parameters, nSV, nLabel);
end
BWout=predict_label<2;%�������Ԫ��<2,�ͱ��Ϊ1������Ϊ0. �õ�Ŀ�����ص��������
BW=reshape(BWout,[r,c]); %��Ϊһ��Ŀ��λ��Ϊ1,����λ��Ϊ0���߼�����

function Ipn=linyu(I,t)
%input:I--����ͼ��,t--����λ��
%output: Ipn---���һ���������ͼ��
[r,c,d]=size(I);
    if t<=d
        Ipn(:,:)=I(:,:,t);%���ص����ɫ��Ϣ
    elseif t>d&&t<=2*d 
        Ipn=I(:,:,t-d); %�������ظ�ֵΪԭͼ���Ա��ڴ󲿷����ص�λ�øı��û�иı�����ص��ѱ�����
        Ipn(1:r-1,1:c-1)=I(2:r,2:c,t-d);%��ɫ��Ϣ%(i+1,j+1)��% ��ʱ���i+1��j+1��ʼ����������������ɫ
    elseif t>2*d&&t<=3*d
        Ipn=I(:,:,t-2*d); %
        Ipn(1:r,1:c-1)=I(1:r,2:c,t-2*d);%��ɫ��Ϣ%(i,j+1)��
    elseif t>3*d&&t<=4*d
        Ipn=I(:,:,t-3*d); %        
        Ipn(2:r,1:c-1)=I(1:r-1,2:c,t-3*d);%��ɫ��Ϣ%(i-1,j+1)��
    elseif t>4*d&&t<=5*d
        Ipn=I(:,:,t-4*d); %        
        Ipn(2:r,1:c)=I(1:r-1,1:c,t-4*d);%��ɫ��Ϣ%(i-1,j)��
    elseif t>5*d&&t<=6*d
        Ipn=I(:,:,t-5*d); %        
        Ipn(2:r,2:c)=I(1:r-1,1:c-1,t-5*d);%��ɫ��Ϣ%(i-1,j-1)��
    elseif t>6*d&&t<=7*d
        Ipn=I(:,:,t-6*d); %        
        Ipn(1:r,2:c)=I(1:r,1:c-1,t-6*d);%��ɫ��Ϣ%(i,j-1)��
    elseif t>7*d&&t<=8*d
        Ipn=I(:,:,t-7*d); %        
        Ipn(1:r-1,2:c)=I(2:r,1:c-1,t-7*d);%��ɫ��Ϣ%(i+1,j-1)��
    elseif t>8*d&&t<=9*d
        Ipn=I(:,:,t-8*d); %        
        Ipn(1:r-1,1:c)=I(2:r,1:c,t-8*d);%��ɫ��Ϣ%(i+1,j)��
    end

function BW=elm_segment_27(I,Ip,smp,NumberofHiddenNeurons,ActivationFunction,type,s_num)
%input:��ɫԭͼI��Ip-��������ֵͼ��Map-����ͼ ,spimg
%NumberofHiddenNeurons,ActivationFunction����elm���ڵ����������
%output:BW--�ָ����Ķ�ֵͼ
[r,c,d]=size(I);
if d>1
    I1=rgb2gray(I);
else
    I1=I;
end
[xx,yy]= gradient(I1);%edge(I,'log');%edge(I,'canny');%���ݶ�ͼ
Ig = sqrt(xx.^2+yy.^2);

%th=1/(r*c)*sum(abs(Ig(:)));%�趨һ���ݶ���ֵ���Ա�object���������ڸ��ݶ����򣬼��ٸ���;�����������Ǿ������
%Ig_bw=im2bw(Ig,th);%>ƽ���ݶȵĸ��ݶȶ�ֵ����

kbn=~Ip;%%��������ѡ��
kbp=logical(Ip);%��������ѡ��%

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

for t=1:kd+2%5%3%+60
    if t<=kd
        Ip=linyu(I,t);%I(:,:,t);%��ɫ��Ϣ
        In=linyu(I,t);%I(:,:,t);
    elseif t==kd+1      
        Ip=Ig;%smp;  %�ݶ���Ϣ
        In=Ig;%smp;     
    elseif t==kd+2      
        Ip=smp;    %��������Ϣ��
        In=smp;
    elseif t>=kd+3 %
        Ip=spimg(:,:,t-(kd+3)+1);%��������Ϣ
        In=spimg(:,:,t-(kd+3)+1);%
    %elseif t>=kd+3      
    %     Ip=Igabor(:,:,t-kd-2);%gabor������Ϣ
    %    In=Igabor(:,:,t-kd-2);   
    elseif t>=kd+6%1       %hsv
        %Ip=Ihsv(:,:,t-4);
        %In=Ihsv(:,:,t-4);%
        Ip=Ilab(:,:,t-4);%-28
        In=Ilab(:,:,t-4);%-28
    end
    sample_pt(:,t)=Ip(kbp);%();%ȡע�������������������
    sample_nt(:,t)=In(kbn);%��Χ�������������������
end

sample_p=sample_pt(pp(1:num_p),:);%���ѡ�� num_p ��������
sample_n=sample_nt(pn(1:num_n),:);%���ѡ�� num_n ��������
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
    %    I_gabor=Igabor(:,:,k-kd-2);%gabor������Ϣ
    %    test_data(:,k)=I_gabor(:);    
    end
end

switch type
    case 'elm'
            [InputWeight,OutputWeight,BiasofHiddenNeurons,TrainingTime] = elm_train(train_label,train_data,NumberofHiddenNeurons,ActivationFunction);
            [predict_label, TestingTime] = elm_predict(test_data, InputWeight,OutputWeight,NumberofHiddenNeurons, BiasofHiddenNeurons,ActivationFunction);
    case 'svm'
            [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = PolySVC(train_data', train_label',1);
            %   %gamma=1; C=100;%3d,5dԭ���Ĳ���ֵ   gamma=0.005; C=100;
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
BWout=predict_label<2;%�������Ԫ��<2,�ͱ��Ϊ1������Ϊ0. �õ�Ŀ�����ص��������
BW=reshape(BWout,[r,c]); %��Ϊһ��Ŀ��λ��Ϊ1,����λ��Ϊ0���߼�����
