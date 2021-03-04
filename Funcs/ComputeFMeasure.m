function [Results]=ComputeFMeasure(RES,resSuffix, GT, gtSuffix) %%��ʾÿ��ͼ��Fscore result
%function ComputeFMeasure(DBpath,SegResultsSubPath,SysType)    %����ʾÿ��ͼ��Fscore result
%Compute the F-score for a single segment
%Syntax:
%       [Results]=ComputeFMeasure(DBpath,SegResultsSubPath,SysType)
%Input:
%       DBpath - The directory of the entire evaluation Database
%       SegResultsSubPath - The name of the sub-directory  in which the results of
%                           the algorithm to be evaluated  are placed.
%       SysType - The type of system in use, this determines the path
%       separation char.There are two optional values 'win' or 'unix' if no value is
%                 specified the default is set to 'win'.
%Output:
%       Results - An 100X3 matrix where Results(i,1) holds the best F-score for a single segment.
%                 Results(i,2) and Results(i,3) holds the corresponding Recall and Precision scores.
%       Example:
%                 [Results]=ComputeFMeasure('c:\Evaluation_DB','MyRes','pc');
%
%The evaluation function is given as is without any warranty. The Weizmann
%institute of science is not liable for any damage, lawsuits, 
%or other loss resulting from the use of the evaluation functions.
%Written by Sharon Alpert Department of Computer Science and Applied Mathematics
%The Weizmann Institute of Science 2007

%�������þ�����
%GT;%�ֶ��ָ��ֵͼ�����GT�� ���Ա�Ա�seg�н����������ܲ���
%RES;%�Զ��ָ�õ��Ķ�ֵͼ��ŵ�seg���ļ��У�Ŀ¼���¡�
%����Ӳ���ͼ��ע����Ӳ���ͼ��ʱ����Ӧ��segĿ¼��ҲҪ���Զ��ָ���ļ����Ա�Աȣ�������ܲ���
l=dir(strcat(GT,'\*',gtSuffix));

Results=zeros(length(l),3);
for i=1:length(l)%100%
    im=im2double(imread(strcat(GT,'\',l(i).name)));
    Hmask=((im(:,:,1)>0.5));%  0.6           Get the Human binary segmentation                        %
    %fprintf('Working on image:%s\n',l(i).name);

    noSuffixName = l(i).name(1:end-length(gtSuffix));
    smapName=fullfile(RES, strcat(noSuffixName, resSuffix));
    Segmap=(imread(smapName));
    
    Hmask=imresize(Hmask,size(Segmap));%??????????????
    
    [Pmax Rmax Fmax]=CalcCandScore(Segmap,Hmask);
    Results(i,1)=Fmax;
    Results(i,2)=Rmax;
    Results(i,3)=Pmax;
    %figure;
    %subplot(121);imshow(Hmask);title(l(i).name);
    %subplot(122);imshow(smapName);title(num2str(Fmax));
end;
fprintf('F_score:%s\n',num2str(mean(Results(:,1)))); %��ʾƽ����F_score
fprintf('R_score:%s\n',num2str(mean(Results(:,2)))); %��ʾƽ���Ļص���
fprintf('P_score:%s\n',num2str(mean(Results(:,3)))); %��ʾƽ������ȷ��
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             Calcuate the F-score                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [p r f]=CalcPRPixel(GT,mask)
    if (sum(GT(:)&mask(:))==0)
        p=0;r=0;f=0;
        return;
    end;
    r=sum(GT(:)&mask(:))./sum(GT(:));
    c=sum(mask(:))-sum(GT(:)&mask(:));
    p=sum(GT(:)&mask(:))./(sum(GT(:)&mask(:))+c);
    %f=(r*p)/(0.5*(r+p));%ԭ��׼
    beta2=0.3;
    f=((1+beta2)*r*p)/(r+beta2*p);%2015cheng minmin
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             Calcuate the F-score of the evaluated method             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Pmax Rmax Fmax]=CalcCandScore(Segmap,Humanmap)

Fmax=0;
Pmax=0;
Rmax=0;

   Segmap=(Segmap(:,:,1)==1);%�Զ��ָ��Ŀ��=1������=0(�˴��ñ�������Ŀ��Segmap(:,:,1)==0��Fֵ�������䣬��Ϊ��������Ŀ���ֵ=1 or =0������)
   NumOfSegs=unique(Segmap(:)); %find out how many segments
   
   for j=1:length(NumOfSegs)
             t=(Segmap==NumOfSegs(j));
             if sum(t(:))<=5 continue;end; %skip small segments
             [p r f]=CalcPRPixel(t,Humanmap);
             if (f>Fmax)
                 Fmax=f;
                 Pmax=p;
                 Rmax=r;
             end;
             
   end;%Go over all segments in the image
end%Go over all segmentations in the Dir


