function [Results]=ComputeFMeasure(RES,resSuffix, GT, gtSuffix) %%显示每个图的Fscore result
%function ComputeFMeasure(DBpath,SegResultsSubPath,SysType)    %不显示每个图的Fscore result
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

%参数设置举例：
%GT;%手动分割二值图像放在GT中 ，以便对比seg中结果，算出性能参数
%RES;%自动分割得到的二值图像放到seg子文件夹（目录）下。
%可添加测试图像，注意添加测试图像时，对应当seg目录中也要有自动分割的文件，以便对比，算出性能参数
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
fprintf('F_score:%s\n',num2str(mean(Results(:,1)))); %显示平均的F_score
fprintf('R_score:%s\n',num2str(mean(Results(:,2)))); %显示平均的回调率
fprintf('P_score:%s\n',num2str(mean(Results(:,3)))); %显示平均的正确率
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
    %f=(r*p)/(0.5*(r+p));%原标准
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

   Segmap=(Segmap(:,:,1)==1);%自动分割的目标=1，背景=0(此处用背景当作目标Segmap(:,:,1)==0，F值基本不变，因为下面语句对目标的值=1 or =0不敏感)
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


