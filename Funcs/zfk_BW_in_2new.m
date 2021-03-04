function [BWout, smp] = zfk_BW_in_2new(srcImg,smp1,smp2,alf)
%% ˫ͨ��zfk  ��֪���͵�����Ŀ���⣬�����ʼ˫����ͼ�����Ŀ���ֵͼ��������ͼ
%% ˫ͨ��zfk�ںϣ���ʼkk=2ʱ�Ƚ�bw1��bw2,����ͬ�ڵ�ͨ������ע������zfk,ֱ������(ѡ����ͨ������ɿ�ͨ�������������)
%ѡ���ʵ�ͼ��
    sp_img = srcImg;
    [r2, c2, ~]=size(sp_img);%ԭ��ͼsp_img
    
    smp10 = imresize(smp1,[r2 c2]);%�ߴ���BWһ��,׼����������
    smp20 = imresize(smp2,[r2 c2]);    
    bw1= imbinarize(smp10);%��ֵ�����õ�Ŀ������             
    bw1 = select_max_region(bw1,5);%��ֵ��ͼ�У�ѡ��������2����ͨ��������Ȥ��Ŀ��
    bw2= imbinarize(smp20);%��ֵ�����õ�Ŀ������             
    bw2 = select_max_region(bw2,5);   
    
    %alf=0.1;%0.5;%two-supervised%unsupervised+supervised��Ȩϵ��
    sMap = mat2gray(alf*smp10 + (1-alf)*smp20);%
    smp1= imresize(sMap,[r2 c2]);
    
    if (all(bw1(:)==1)) %�ų�bottom-upͨ��ȫ������1,�ޱ������ص����            
           bw3=bw2;
           bw4=bw1;
           bw5=~bw2;
    elseif (all(bw2(:)==0)) %�ų�top-downͨ��ȫ������0,��Ŀ���������
           bw3=bw1;
           bw4=~bw2;
           bw5=~bw1;
    else  
           bw3=and(bw1,bw2);%bw1��bw2����,1�������ĸ�֪��������mask��0-����
           bw4=xor(bw1,bw2);%����������mask��1Ϊmask,0Ϊ������
           bw5=~or(bw1,bw2);%�������ı�����mask��1Ϊ������������mask,0-����
    end 
    BW=main_elm_segment_271(sp_img,bw3,bw4,bw5,smp10,smp20,40,'sig','elm',1000);%20201213�Ķ���,smp1,smp2ֱ�ӽ���PELMѧϰ��ģ������alf��Ȩ
    BW=or(and(bw4,BW),bw3);%������mask+����������mask => ���ɸ��µ�Ŀ����
    BW=select_max_region(BW,5);
    %kk=1;
    %BW_t(:,:,kk)=BW;%BW_tʱ����������   
    %smp1=mat2gray(smp1+BW);%����Ч����
    [BWout, smp] = zfk_BW_in(sp_img,BW,smp1);%��ͨ��,20201213�Ķ����������zfkЧ�������Ч����á�
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
end


function BW=main_elm_segment_271(I,bw3,bw4,bw5,smp1,smp2,NumberofHiddenNeurons,ActivationFunction,type,s_num)
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

kbn=bw5;%~Ip;%%��������ѡ��
kbp=bw3;%logical(Ip);%��������ѡ��%

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

for t=1:kd+3%5%3%+60
    if t<=kd
        Ip=linyu(I,t);%I(:,:,t);%��ɫ��Ϣ
        In=linyu(I,t);%I(:,:,t);
    elseif t==kd+1      
        Ip=Ig;%smp;  %�ݶ���Ϣ
        In=Ig;%smp;     
    elseif t==kd+2      
        Ip=smp1;    %��������Ϣ��
        In=smp1;
    elseif t>=kd+3 %
        Ip=smp2;%spimg(:,:,t-(kd+3)+1);%��������Ϣ
        In=smp2;%spimg(:,:,t-(kd+3)+1);%
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
end

