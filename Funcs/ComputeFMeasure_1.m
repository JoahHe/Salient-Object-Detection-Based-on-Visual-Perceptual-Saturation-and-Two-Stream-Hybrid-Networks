function Fmax=ComputeFMeasure_1(Hmask,im) %%比较Hmask,im的Fscore

    Hmask=((Hmask(:,:,1)==1));%Get the Human binary segmentation 目标=1，背景=0(用背景当作目标(im(:,:,1)==0，F值会提高至0.9，因为背景面积大，像素多)                   %
    im=im(:,:,1);
    
    [Pmax Rmax Fmax]=CalcCandScore(im,Hmask);   
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
function [Pmax Rmax Fmax]=CalcCandScore(Segmap,HumanSegmap)

Fmax=0;
Pmax=0;
Rmax=0;

   Segmap=(Segmap(:,:,1)==1);%自动分割的目标=1，背景=0(此处用背景当作目标Segmap(:,:,1)==0，F值基本不变，因为下面语句对目标的值=1 or =0不敏感)
   NumOfSegs=unique(Segmap(:)); %find out how many segments
   
   for j=1:length(NumOfSegs)
             t=(Segmap==NumOfSegs(j));
             if sum(t(:))<=5 continue;end; %skip small segments
             [p r f]=CalcPRPixel(t,HumanSegmap);
             if (f>Fmax)
                 Fmax=f;
                 Pmax=p;
                 Rmax=r;
             end;
             
        end;%Go over all segments in the image      
end%Go over all segmentations in the Dir


