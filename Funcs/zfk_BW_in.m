function [BW, smp] = zfk_BW_in(srcImg,BW,sMap)
%% bottom-up ��������������Ŀ���⣬����ͼ�����Ŀ���ֵͼ������ͼ
%rand('state',370);
%ѡ���ʵ�ͼ��
    %scale_n = 200;    sp_img =im2double(imresize(srcImg,scale_n/size(srcImg,2)));%im2double(srcImg);%
    sp_img = srcImg;
    [r2, c2, d2]=size(sp_img);%��Сͼsp_img
    
    BW1 = imresize(BW,[r2 c2]);%��СBW,׼����������
    smp1= imresize(sMap,[r2 c2]);
    %spimg1 = imresize(spimg,[r2 c2]);
    
    [smp, BW_t] = elm_zfk_segment(sp_img,smp1);%��ʾ�������γɵ�ʱ�����ж�ֵͼBW_t,spimg1

 %% Segmentation by learning
    [r0,c0,d0]=size(srcImg);
 %%{
    BW=a_threshold(smp,'ostu');%
    BW=imresize(BW,[r0,c0]);
    %BW=select_max_region(BW,1);%
    
    %BW =main_elm_segment_27(srcImg,BW,40,'sig','elm',1000);%0.8659��rbd+zfnet+fp��PR�����ж˲�ƽ�����ж˵���rbd+afnet����
    %%BW =main_elm_segment_28(srcImg,BW,sMap,40,'sig','elm',1000);%0.8621��rbd+zfnet+fp��PR����ƽ������rbd+afnet���߻����غ�
    
    BW=select_max_region(BW,5);%
    %BWout=imfill(BW,'holes');%output BW of object
 %}
    smp=imresize(smp,[r0,c0]);%output smp ԭ�������ͼ
    
end

function Iout = rbd(srcImg,pixNumInSP)

%% 1. Parameter Settings
doFrameRemoving = true;
doNormalize = true;
useSP = true;           %You can set useSP = false to use regular grid for speed consideration

%% 2. Saliency Map Calculation
    if doFrameRemoving
        [noFrameImg, frameRecord] = removeframe(srcImg, 'sobel');
        [h, w, chn] = size(noFrameImg);
    else
        noFrameImg = srcImg;
        [h, w, chn] = size(noFrameImg);
    end       
    frameRecord = [h, w, 1, h, 1, w];

    
    %% Segment input rgb image into patches (SP/Grid)
    %pixNumInSP = 550;                           %pixels in each superpixel
    spnumber = round( h * w / pixNumInSP );     %super-pixel number for current image
    
    if useSP
        [idxImg, adjcMatrix, pixelList] = SLIC_Split(noFrameImg, spnumber);
    else
        [idxImg, adjcMatrix, pixelList] = Grid_Split(noFrameImg, spnumber);        
    end
    %% Get super-pixel properties
    spNum = size(adjcMatrix, 1);
    meanRgbCol = GetMeanColor(noFrameImg, pixelList);
    meanLabCol = colorspace('Lab<-', double(meanRgbCol)/255);
    meanPos = GetNormedMeanPos(pixelList, h, w);
    bdIds = GetBndPatchIds(idxImg);
    colDistM = GetDistanceMatrix(meanLabCol);
    posDistM = GetDistanceMatrix(meanPos);
    [clipVal, geoSigma, neiSigma] = EstimateDynamicParas(adjcMatrix, colDistM);
    
    %% Saliency Optimization
    [bgProb, bdCon, bgWeight] = EstimateBgProb(colDistM, adjcMatrix, bdIds, clipVal, geoSigma);
    wCtr = CalWeightedContrast(colDistM, posDistM, bgProb);
    optwCtr = SaliencyOptimization(adjcMatrix, bdIds, colDistM, neiSigma, bgWeight, wCtr);
    
    %smapName=fullfile(RES, strcat(noSuffixName, '_wCtr_Optimized.png'));
    Iout = SaveSaliencyMap(optwCtr, pixelList, frameRecord, meanPos,doNormalize);
end

function [num,scale_nn] = scale_etp(Ii)
%Input: Ii--color image
%Output:nn--��Ӧ��С�صĳ߶�
ss=[];
n=[];
%kk=[32,40,50,64];%0.8736
kk=[32,40,50,64,80];%0.8736
%kk=[30,40,50,60];%0.85802

for i=1:length(kk)
    Img = imresize(Ii,kk(i)/size(Ii,2));%��߶�
    [n(i),ss(i)]=fix_p_etp(Img);  %ĳ���߶��µ�ע�ӵ�ľ�����ֵ  
end
[cmin, ind]=sort(ss);%��С��ɢ�ȣ���Ӧ�ĳ߶�
scale_nn=kk(ind(1));%�߶Ȳ���kk
num=n(ind(1));%ע�ӵ�����
%figure;plot(ss);title('��Сͼ��-��С��');     
end

function [num,etp] = fix_p_etp(inImg)
%% ������ͼ����Phase�õ� 
     [r,c,d]=size(inImg);
     saliencyMap=rgb2gray(phase_fft(inImg));%�Աຯ�� 0.8634
     %saliencyMap=(phase_fft(inImg));%�Աຯ�� 0.8564
     %saliencyMap = saliencyMap1(inImg);%0.8606

     hr=round(0.1*r);%ȥ���߽����
     hc=round(0.1*c);%
     saliencyMap(1:hr,:)=0;
     saliencyMap(r-hr:r,:)=0;
     saliencyMap(:,1:hc)=0;
     saliencyMap(:,c-hc:c)=0;

     B=saliencyMap(:);   %�����ֵλ��
     [cmax, ind]=sort(B,'descend');%��������������ֵ���Ա��ҵ�ǰn�����ֵ������λ��
        
     M1=mean(B);%��ֵ
     L_m1=B(find(B>M1));%���ھ�ֵ�ĵ�
     M2=mean(L_m1);
     L_m2=L_m1(find(L_m1>M2));%����3/4ֵ�ĵ�
     %M3=mean(L_m2);
     %L_m3=L_m2(find(L_m2>M3));

     n=length(L_m2);%length(L_m2)%length(L_m3);%11;%
     %if n<100                                    %+0.8738  %-0.0.8738
     %   n=100;
     %end
     num=n;%�������num--��ע�ӵ�����
        
     lmax=sqrt(r*r+c*c);         
     ind=ind(1:n);
     [x,y]=ind2sub(size(saliencyMap),ind);%ע�ӵ��λ����Ϣ(2ά)
     cent=[round(r/2),round(c/2)];%0.868
     %cent(:,1)=round(mean(x));     cent(:,2)=round(mean(y));%0.8675
     
     d=pdist2([x,y],cent);%�㵽ͼ�����ĵľ���
     
     d_n=d/lmax;%�����һ��
     
     step=0.01;%1/num;%��Χ����,
     rang=0:step:1;
     L=round(1/step);%num;%
     H_p=0;%���ۼ�
     p_prop=zeros(L);%zeros(num);
     for i=1:L
         d_n_1= d_n>=rang(i);
         d_n_2= d_n<rang(i+1);
         p_num= d_n_1 & d_n_2;%������뷶Χ�����ĵ�����      
         p_prop(i)=sum(p_num)/n;%�����������ʾΪ����_
         if p_prop(i)~=0         %ȥ������Ϊ0�ĵ�
             H_p=-p_prop(i)*log2(p_prop(i))+H_p;   %����ֵ�Ĺ�ʽ
         end
     end
     %figure;plot(p_prop);title('����ͼ');
     etp=H_p;
end

function saliencyMap=phase_fft(inImg)  
%Input:inImg->�Ҷ�ͼ��
%Output:saliencyMap-������ͼ

myFFT = fft2(inImg); 
myPhase = angle(myFFT);
saliencyMap = abs(ifft2(exp(i*myPhase))).^2; 
end

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

%(1)PELM
   [InputWeight,OutputWeight,BiasofHiddenNeurons,TrainingTime] = elm_train(train_label,train_data,NumberofHiddenNeurons,ActivationFunction);
   [predict_label, TestingTime] = elm_predict(test_data, InputWeight,OutputWeight,NumberofHiddenNeurons, BiasofHiddenNeurons,ActivationFunction);
%{   
%(2)RVFL���ɭ��
   %s = RandStream('mcg16807','Seed',0);
   %RandStream.setGlobalStream(s);
   %option.ensemble = 20;
   %predict_label = hybrid_model(train_data,train_label,test_data,option);
%(3)RVFL   
   option.N =40;%number of hidden neurons
   % option.bias:    whether to have bias in the output neurons
   % option.link:    whether to have the direct link.
   option.ActivationFunction = 'sig';%Activation Functions used   
   % option.seed:    Random Seeds
   % option.mode     1: regularized least square, 2: Moore-Penrose pseudoinverse
   % option.RandomType: different randomnization methods. Currently only support Gaussian and uniform.
   % option.Scale    Linearly scale the random features before feedinto the
   % nonlinear activation function.  
   %                 In this implementation, we consider the threshold which lead to 0.99 of the maximum/minimum value of the activation function as the saturating threshold.
   %                 Option.Scale=0.9 means all the random features will be linearly scaled
   %                 into 0.9* [lower_saturating_threshold,upper_saturating_threshold].
   %option.Scalemode Scalemode=1 will scale the features for all neurons.
   %                 Scalemode=2  will scale the features for each hidden
   %                 neuron separately.
   %                 Scalemode=3 will scale the range of the randomization for
   %                 uniform diatribution.
   test_label = ones(size(test_data,1),1);
   predict_label = RVFL_train_online(train_data,train_label,test_data,test_label,option);
%}
BWout=predict_label<1.5;%�������Ԫ��<2,�ͱ��Ϊ1������Ϊ0. �õ�Ŀ�����ص��������
BW=reshape(BWout,[r,c]); %��Ϊһ��Ŀ��λ��Ϊ1,����λ��Ϊ0���߼�����
end
 
function [bgProb, bdCon, bgWeight] = EstimateBgProb(colDistM, adjcMatrix, bdIds, clipVal, geoSigma)
% Estimate background probability using boundary connectivity

% Code Author: Wangjiang Zhu
% Email: wangjiang88119@gmail.com
% Date: 3/24/2014

bdCon = BoundaryConnectivity(adjcMatrix, colDistM, bdIds, clipVal, geoSigma, true);

bdConSigma = 1; %sigma for converting bdCon value to background probability
fgProb = exp(-bdCon.^2 / (2 * bdConSigma * bdConSigma)); %Estimate bg probability
bgProb = 1 - fgProb;

bgWeight = bgProb;
% Give a very large weight for very confident bg sps can get slightly
% better saliency maps, you can turn it off.
fixHighBdConSP = true;
highThresh = 3;
if fixHighBdConSP
    bgWeight(bdCon > highThresh) = 1000;
end

end

function optwCtr = SaliencyOptimization(adjcMatrix, bdIds, colDistM, neiSigma, bgWeight, fgWeight)
% Solve the least-square problem in Equa(9) in our paper

% Code Author: Wangjiang Zhu
% Email: wangjiang88119@gmail.com
% Date: 3/24/2014

adjcMatrix_nn = LinkNNAndBoundary(adjcMatrix, bdIds);
colDistM(adjcMatrix_nn == 0) = Inf;
Wn = Dist2WeightMatrix(colDistM, neiSigma);      %smoothness term
mu = 0.1;                                                   %small coefficients for regularization term
W = Wn + adjcMatrix * mu;                                   %add regularization term
D = diag(sum(W));

bgLambda = 5;   %global weight for background term, bgLambda > 1 means we rely more on bg cue than fg cue.
E_bg = diag(bgWeight * bgLambda);       %background term
E_fg = diag(fgWeight);          %foreground term

spNum = length(bgWeight);
optwCtr =(D - W + E_bg + E_fg) \ (E_fg * ones(spNum, 1));
end

function sMap=SaveSaliencyMap(feaVec, pixelList, frameRecord, doNormalize, fill_value)
% Fill back super-pixel values to image pixels and save into .png images

% Code Author: Wangjiang Zhu
% Email: wangjiang88119@gmail.com
% Date: 3/24/2014

if (~iscell(pixelList))
    error('pixelList should be a cell');
end

if (nargin < 5)
    doNormalize = true;
end

if (nargin < 6)
    fill_value = 0;
end

h = frameRecord(1);
w = frameRecord(2);

top = frameRecord(3);
bot = frameRecord(4);
left = frameRecord(5);
right = frameRecord(6);

partialH = bot - top + 1;
partialW = right - left + 1;
partialImg = CreateImageFromSPs(feaVec, pixelList, partialH, partialW, doNormalize);

if partialH ~= h || partialW ~= w
    feaImg = ones(h, w) * fill_value;
    feaImg(top:bot, left:right) = partialImg;
    %imwrite(feaImg, imgName);
    sMap=feaImg;
else
    sMap=partialImg;
    %imwrite(partialImg, imgName);
end
end

function [img, spValues] = CreateImageFromSPs(spValues, pixelList, height, width, doNormalize)
% create an image from its superpixels' values
% spValues is all superpixel's values, e.g., saliency
% pixelList is a cell (with the same size as spValues) of pixel index arrays

% Code Author: Wangjiang Zhu
% Email: wangjiang88119@gmail.com
% Date: 3/24/2014

if (~iscell(pixelList))
    error('pixelList should be a cell');
end

if (length(pixelList) ~= length(spValues))
    error('different sizes in spValues and pixelList');
end

if (nargin < 5)
    doNormalize = true;
end

minVal = min(spValues);
maxVal = max(spValues);
if doNormalize
    spValues = (spValues - minVal) / (maxVal - minVal + eps);
else
    if minVal < -1e-6 || maxVal > 1 + 1e-6
        error('feature values do not range from 0 to 1');
    end
end

img = zeros(height, width);
for i=1:length(pixelList)
    img(pixelList{i}) = spValues(i);
end
end

function [sMap_new, BW, kk]=elm_learning_segment_1(sp_img)
%srcImg:ԭͼ��sp_img:��������Сͼ
%sMap;%learning���,����ͼ����ֵͼ
%scale:ͼ�����ų߶�
%BW:output��ֵͼ��

[r,c,~]=size(sp_img);
sMap=saliencyMap1(sp_img);%��ȡ��ɫ����ͼ

hy=round(0.05*c);%��ʩ��ͼ����ص���������0������ͼ��Ե����(),best 0.05
hx=round(0.05*r);
sMap(1:hx,:)=0;
sMap(:,1:hy)=0;
sMap(r-hx:r,:)=0;
sMap(:,c-hy:c)=0;
BW=a_threshold(sMap,'ostu');%����ͼ��ֵ�����Զ�ȷ��Ŀ����ֵ

BW=select_max_region(BW,3);
BW=imfill(BW,'holes'); 

kk=0;
Ip=BW;
sum=mat2gray(sMap);
while (1)
    times=1;
    sum=sum+sMap_learning(sp_img,Ip,sMap,times);%elm27 Ч����
    BW=a_threshold(sum,'ostu');
    BW=select_max_region(BW,3);
    Fmax=ComputeFMeasure_1(Ip,BW);
    kk=kk+1;
    if Fmax>=0.95 || kk>10   
        break;
    else
        Ip=BW;
        %sMap=mat2gray(sum);%new+ %elm28����
    end
end  
sMap_new=mat2gray(sum);   
end

function [sMap_new, BW_t] = elm_zfk_segment(sp_img,sMap)
%srcImg:ԭͼ��sp_img:��������Сͼ,spimg
%sMap;%learning���,����ͼ����ֵͼ
%scale:ͼ�����ų߶�
%BW_t:output��ֵ����ͼ��
if islogical(sMap)
    BW=sMap;
else
    BW=a_threshold(sMap,'ostu');%����ͼ��ֵ�����Զ�ȷ��Ŀ����ֵ
end
BW=select_max_region(BW,5);%3,5
%BW=imfill(BW,'holes'); 
BW_t(:,:,1)=BW;

kk=1;
Ip=BW;
sum=mat2gray(sMap);%����ԭ����ͼ%zeros(size(sMap));
while (1)
    times=1;%1;
    %sum=sum+sMap_learning(sp_img,Ip,sMap,times);%����rgb  F=0.8620
    beta=1/(kk+1);%2/(kk+1);%���������ĵ�����Ϊ��Ȩϵ������������Խ������������Ӱ��ԽС��
    sum=sum+beta*sMap_learning_1(sp_img,Ip,sMap,times);%9�㣬rgb*9+smp F=0.8625spimg,
    BW=a_threshold(sum,'ostu');
    BW=select_max_region(BW,5);%3,5
    Fmax=ComputeFMeasure_1(Ip,BW);
    kk=kk+1;
    BW_t(:,:,kk)=BW;
    if Fmax>=0.85 || kk>10   
        break;
    else
        Ip=BW;
        sMap = mat2gray(sum);%��ǰû�У�20191128����
    end
end  
sMap_new = mat2gray(sum);
end

function BW=main_elm_segment_28(I,Ip,smp,NumberofHiddenNeurons,ActivationFunction,type,s_num)

%input:��ɫԭͼI��Ip-��������ֵͼ��Map-����ͼ 
%NumberofHiddenNeurons,ActivationFunction����elm���ڵ����������
%output:BW--�ָ����Ķ�ֵͼ

[r,c,d]=size(I);

if d>1
    I1=rgb2gray(I);
end
[xx,yy]= gradient(I1);%edge(I,'log');%edge(I,'canny');%���ݶ�ͼ
Ig = sqrt(xx.^2+yy.^2);

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

for t=1:kd+2
    if t<=kd
        Ip=I(:,:,t);%��ɫ��Ϣ
        In=I(:,:,t);
    elseif t==kd+1      
        Ip=smp;%����ͼ��Ϣ
        In=smp;%
    elseif t==kd+2 %����1
        Ip=Ig;
        In=Ig;%
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
for k=1:kd+1
    if k<=kd
        I_=I(:,:,k);
        test_data(:,k)=I_(:);
    elseif k==kd+1
        test_data(:,k)=smp(:);
    elseif k==kd+2
        test_data(:,k)=Ig(:);
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

end