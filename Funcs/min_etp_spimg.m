function num_sp = min_etp_spimg(srcImg)
%Input: srcImg--color image �����ɫͼ��
%Output:num_sp--�����Ӧ��С�صĳ�������������
[r,c,d]=size(srcImg); 
scale_n = scale_etp(srcImg);
%���ճ߶���Сͼ��    
sp_img =im2double(imresize(srcImg,scale_n/size(srcImg,2)));
[r1,c1,d1]=size(sp_img); 

num_sp=floor((r*c)/(r1*c1));%pixNumInSP
%num_sp                                   %��ʾ���ŵ� ÿ�������ر�ʾ����������  
end

function scale_nn = scale_etp(Ii)
%Input: Ii--color image
%Output:scale_nn--��Ӧ��С�ص���Сͼ���height

%kk=[32,40,50];%
kk=[25,30,35,40,45,50];%
Img = imresize(Ii,kk(1)/size(Ii,2));%��߶�
max_e = fix_p_etp(Img);  %���߶��µ�ע�ӵ�ľ�����ֵ

flag=0;%�����־
for i=2:length(kk)
    Img = imresize(Ii,kk(i)/size(Ii,2));%��߶�
    ss = fix_p_etp(Img);  %ĳ���߶��µ�ע�ӵ�ľ�����ֵ
    if (max_e <= ss && flag==0 && i<length(kk))
        max_e = ss; %ĳ���߶��µ�ע�ӵ�ľ�����ֵ 
    elseif (max_e > ss && flag==0)
        scale_nn = kk(i);
        break;
    elseif (max_e <= ss && flag>0)
        scale_nn = kk(i-1);
        break;
    elseif (max_e > ss && flag>0)
        max_e = ss;
        flag=flag+1;%�����־ 
    else
        scale_nn = kk(2);%
    end
end
%disp(scale_nn);
end

function etp = fix_p_etp(inImg)
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
        
     n=50;%length(L_m1);%length(L_m2);%length(L_m3);%11;%20
     if length(B)<50                                    %20 +0.8738  %-0.0.8738
        n=length(B);%
     end
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


%{
function scale_nn = scale_etp(Ii)
%Input: Ii--color image
%Output:nn--��Ӧ��С�صĳ߶�

%kk=[24,32,40,50,64];%pr no so good
%kk=[32,36,40,45,50];%good
kk=[25,30,35,40,45,50];%
ss=zeros(length(kk));%[];
for i=1:length(kk)
    Img = imresize(Ii,kk(i)/size(Ii,2));%��߶�
    ss(i)=fix_p_etp(Img);  %ĳ���߶��µ�ע�ӵ�ľ�����ֵ  
end
[cmin, ind]=sort(ss);%��С��ɢ�ȣ���Ӧ�ĳ߶�
scale_nn=kk(ind(1));%�߶Ȳ���kk
%figure;plot(ss);title('��Сͼ��-��С��');     
end

function etp = fix_p_etp(inImg)
%% ������ͼ����Phase�õ� 
     [r,c,d]=size(inImg);
     saliencyMap=rgb2gray(phase_fft(inImg));%�Աຯ�� 0.8634
     
     hr=round(0.1*r);%ȥ���߽����
     hc=round(0.1*c);%
     saliencyMap(1:hr,:)=0;
     saliencyMap(r-hr:r,:)=0;
     saliencyMap(:,1:hc)=0;
     saliencyMap(:,c-hc:c)=0;

     B=saliencyMap(:);   %�����ֵλ��
     [cmax, ind]=sort(B,'descend');%��������������ֵ���Ա��ҵ�ǰn�����ֵ������λ��
          
     n=20;%length(L_m1);%length(L_m2);%length(L_m3);%11;%
     if length(B)<n                                    %+0.8738  %-0.0.8738
        n=length(B);%
     end
     %num=n;%�������num--��ע�ӵ�����    
     lmax=sqrt(r*r+c*c);         
     ind=ind(1:n);
     [x,y]=ind2sub(size(saliencyMap),ind);%ע�ӵ��λ����Ϣ(2ά)
     %plot(x,y,'*');
     xi=[x,y];
     mat_dist = squareform(pdist(xi));%���������
     mean_dist =  (sum(mat_dist(:))/(n*(n-1)*0.25))/lmax;%ƽ�������һ��
     %mean_dist = (sum(mat_dist(:))/(n*(n-1)*0.25))/max(mat_dist(:));%ƽ�������һ��
     %figure;plot(p_prop);title('����ͼ');
     etp=mean_dist; 
   
%{   
     M1=mean(B);%��ֵ
     L_m1=B(find(B>M1));%���ھ�ֵ�ĵ�
     M2=mean(L_m1);
     L_m2=L_m1(find(L_m1>M2));%����3/4ֵ�ĵ�

     n=length(L_m2);%����Ҫ��ĵ������
     %if n<100      %�����趨һ������ֵ
     %   n=100;
     %end
     num=n;%�������num--��ע�ӵ�����
        
     lmax=sqrt(r*r+c*c);%ͼ��Խ��߳��ȣ��������һ����         
     ind=ind(1:n);
     [x,y]=ind2sub(size(saliencyMap),ind);%ע�ӵ��λ����Ϣ(2ά)
     cent=[round(r/2),round(c/2)];%ͼ������λ��
    
     d=pdist2([x,y],cent);%����ע�ӵ㵽ͼ�����ĵľ���
     
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
%}
end
%}

function saliencyMap=phase_fft(inImg)  
%�������������Ҷ�ͼ��inImg������ͼ
%Input:inImg->�Ҷ�ͼ��
%Output:saliencyMap-������ͼ

myFFT = fft2(inImg); 
myPhase = angle(myFFT);
saliencyMap = abs(ifft2(exp(i*myPhase))).^2; 

end

