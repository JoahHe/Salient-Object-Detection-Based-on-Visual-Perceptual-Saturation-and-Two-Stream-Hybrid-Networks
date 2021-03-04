function Ip=a_threshold(sMap,type)
%type:'ostu' or 'adaptive'
[r c]=size(sMap);
I3=sMap;%mat2gray(sMap);%转为图像后，fscore稍高些
%I3=sMap;%直接用sMap数值，fscore低
if strcmp(type,'ostu')%==1
   level=graythresh(I3);%大津法自动确定目标阈值
else
    level=2*sum(I3(:))/(r*c);%自适应阈值法
    if level<=0 || level>=1
        level=graythresh(I3);%level=0.1;%
    end
end

Ip=im2bw(I3,level);
%Ip=select_max_region(Ip,3);%
%Ip=imfill(Ip,'holes');

end