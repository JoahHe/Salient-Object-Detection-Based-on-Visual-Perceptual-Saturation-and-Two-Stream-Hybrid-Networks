function Ip=a_threshold(sMap,type)
%type:'ostu' or 'adaptive'
[r c]=size(sMap);
I3=sMap;%mat2gray(sMap);%תΪͼ���fscore�Ը�Щ
%I3=sMap;%ֱ����sMap��ֵ��fscore��
if strcmp(type,'ostu')%==1
   level=graythresh(I3);%����Զ�ȷ��Ŀ����ֵ
else
    level=2*sum(I3(:))/(r*c);%����Ӧ��ֵ��
    if level<=0 || level>=1
        level=graythresh(I3);%level=0.1;%
    end
end

Ip=im2bw(I3,level);
%Ip=select_max_region(Ip,3);%
%Ip=imfill(Ip,'holes');

end