function Fmax=ComputeFMeasure_1(Hmask,im) %%�Ƚ�Hmask,im��Fscore

    Hmask=((Hmask(:,:,1)==1));%Get the Human binary segmentation Ŀ��=1������=0(�ñ�������Ŀ��(im(:,:,1)==0��Fֵ�������0.9����Ϊ������������ض�)                   %
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
    %f=(r*p)/(0.5*(r+p));%ԭ��׼
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

   Segmap=(Segmap(:,:,1)==1);%�Զ��ָ��Ŀ��=1������=0(�˴��ñ�������Ŀ��Segmap(:,:,1)==0��Fֵ�������䣬��Ϊ��������Ŀ���ֵ=1 or =0������)
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


