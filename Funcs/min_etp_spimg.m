function num_sp = min_etp_spimg(srcImg)
%Input: srcImg--color image 输入彩色图像
%Output:num_sp--输出对应最小熵的超像素数量参数
[r,c,d]=size(srcImg); 
scale_n = scale_etp(srcImg);
%按照尺度缩小图像    
sp_img =im2double(imresize(srcImg,scale_n/size(srcImg,2)));
[r1,c1,d1]=size(sp_img); 

num_sp=floor((r*c)/(r1*c1));%pixNumInSP
%num_sp                                   %显示最优的 每个超像素表示的像素数量  
end

function scale_nn = scale_etp(Ii)
%Input: Ii--color image
%Output:scale_nn--对应最小熵的缩小图像的height

%kk=[32,40,50];%
kk=[25,30,35,40,45,50];%
Img = imresize(Ii,kk(1)/size(Ii,2));%变尺度
max_e = fix_p_etp(Img);  %最大尺度下的注视点的距离熵值

flag=0;%升序标志
for i=2:length(kk)
    Img = imresize(Ii,kk(i)/size(Ii,2));%变尺度
    ss = fix_p_etp(Img);  %某个尺度下的注视点的距离熵值
    if (max_e <= ss && flag==0 && i<length(kk))
        max_e = ss; %某个尺度下的注视点的距离熵值 
    elseif (max_e > ss && flag==0)
        scale_nn = kk(i);
        break;
    elseif (max_e <= ss && flag>0)
        scale_nn = kk(i-1);
        break;
    elseif (max_e > ss && flag>0)
        max_e = ss;
        flag=flag+1;%降序标志 
    else
        scale_nn = kk(2);%
    end
end
%disp(scale_nn);
end

function etp = fix_p_etp(inImg)
%% 显著度图，由Phase得到 
     [r,c,d]=size(inImg);
     saliencyMap=rgb2gray(phase_fft(inImg));%自编函数 0.8634
     %saliencyMap=(phase_fft(inImg));%自编函数 0.8564
     %saliencyMap = saliencyMap1(inImg);%0.8606

     hr=round(0.1*r);%去除边界干扰
     hc=round(0.1*c);%
     saliencyMap(1:hr,:)=0;
     saliencyMap(r-hr:r,:)=0;
     saliencyMap(:,1:hc)=0;
     saliencyMap(:,c-hc:c)=0;

     B=saliencyMap(:);   %求最大值位置
     [cmax, ind]=sort(B,'descend');%降序索引显著性值，以便找到前n个最大值点所在位置
        
     n=50;%length(L_m1);%length(L_m2);%length(L_m3);%11;%20
     if length(B)<50                                    %20 +0.8738  %-0.0.8738
        n=length(B);%
     end
     num=n;%输出参数num--》注视点数量
        
     lmax=sqrt(r*r+c*c);         
     ind=ind(1:n);
     [x,y]=ind2sub(size(saliencyMap),ind);%注视点的位置信息(2维)
     cent=[round(r/2),round(c/2)];%0.868
     %cent(:,1)=round(mean(x));     cent(:,2)=round(mean(y));%0.8675
     
     d=pdist2([x,y],cent);%点到图像中心的距离
     
     d_n=d/lmax;%距离归一化
     
     step=0.01;%1/num;%范围步长,
     rang=0:step:1;
     L=round(1/step);%num;%
     H_p=0;%熵累计
     p_prop=zeros(L);%zeros(num);
     for i=1:L
         d_n_1= d_n>=rang(i);
         d_n_2= d_n<rang(i+1);
         p_num= d_n_1 & d_n_2;%满足距离范围条件的点数量      
         p_prop(i)=sum(p_num)/n;%将点的数量表示为概率_
         if p_prop(i)~=0         %去掉概率为0的点
             H_p=-p_prop(i)*log2(p_prop(i))+H_p;   %求熵值的公式
         end
     end
     %figure;plot(p_prop);title('概率图');
     etp=H_p;
end


%{
function scale_nn = scale_etp(Ii)
%Input: Ii--color image
%Output:nn--对应最小熵的尺度

%kk=[24,32,40,50,64];%pr no so good
%kk=[32,36,40,45,50];%good
kk=[25,30,35,40,45,50];%
ss=zeros(length(kk));%[];
for i=1:length(kk)
    Img = imresize(Ii,kk(i)/size(Ii,2));%变尺度
    ss(i)=fix_p_etp(Img);  %某个尺度下的注视点的距离熵值  
end
[cmin, ind]=sort(ss);%最小的散度，对应的尺度
scale_nn=kk(ind(1));%尺度参数kk
%figure;plot(ss);title('缩小图像-最小熵');     
end

function etp = fix_p_etp(inImg)
%% 显著度图，由Phase得到 
     [r,c,d]=size(inImg);
     saliencyMap=rgb2gray(phase_fft(inImg));%自编函数 0.8634
     
     hr=round(0.1*r);%去除边界干扰
     hc=round(0.1*c);%
     saliencyMap(1:hr,:)=0;
     saliencyMap(r-hr:r,:)=0;
     saliencyMap(:,1:hc)=0;
     saliencyMap(:,c-hc:c)=0;

     B=saliencyMap(:);   %求最大值位置
     [cmax, ind]=sort(B,'descend');%降序索引显著性值，以便找到前n个最大值点所在位置
          
     n=20;%length(L_m1);%length(L_m2);%length(L_m3);%11;%
     if length(B)<n                                    %+0.8738  %-0.0.8738
        n=length(B);%
     end
     %num=n;%输出参数num--》注视点数量    
     lmax=sqrt(r*r+c*c);         
     ind=ind(1:n);
     [x,y]=ind2sub(size(saliencyMap),ind);%注视点的位置信息(2维)
     %plot(x,y,'*');
     xi=[x,y];
     mat_dist = squareform(pdist(xi));%计算点点距离
     mean_dist =  (sum(mat_dist(:))/(n*(n-1)*0.25))/lmax;%平均距离归一化
     %mean_dist = (sum(mat_dist(:))/(n*(n-1)*0.25))/max(mat_dist(:));%平均距离归一化
     %figure;plot(p_prop);title('概率图');
     etp=mean_dist; 
   
%{   
     M1=mean(B);%均值
     L_m1=B(find(B>M1));%大于均值的点
     M2=mean(L_m1);
     L_m2=L_m1(find(L_m1>M2));%大于3/4值的点

     n=length(L_m2);%符合要求的点的数量
     %if n<100      %可以设定一个底限值
     %   n=100;
     %end
     num=n;%输出参数num--》注视点数量
        
     lmax=sqrt(r*r+c*c);%图像对角线长度，做距离归一化用         
     ind=ind(1:n);
     [x,y]=ind2sub(size(saliencyMap),ind);%注视点的位置信息(2维)
     cent=[round(r/2),round(c/2)];%图像中心位置
    
     d=pdist2([x,y],cent);%计算注视点到图像中心的距离
     
     d_n=d/lmax;%距离归一化
     
     step=0.01;%1/num;%范围步长,
     rang=0:step:1;
     L=round(1/step);%num;%
     H_p=0;%熵累计
     p_prop=zeros(L);%zeros(num);
     for i=1:L
         d_n_1= d_n>=rang(i);
         d_n_2= d_n<rang(i+1);
         p_num= d_n_1 & d_n_2;%满足距离范围条件的点数量      
         p_prop(i)=sum(p_num)/n;%将点的数量表示为概率_
         if p_prop(i)~=0         %去掉概率为0的点
             H_p=-p_prop(i)*log2(p_prop(i))+H_p;   %求熵值的公式
         end
     end
     %figure;plot(p_prop);title('概率图');
     etp=H_p;
%}
end
%}

function saliencyMap=phase_fft(inImg)  
%用这个函数求出灰度图像inImg的显著图
%Input:inImg->灰度图像
%Output:saliencyMap-〉显著图

myFFT = fft2(inImg); 
myPhase = angle(myFFT);
saliencyMap = abs(ifft2(exp(i*myPhase))).^2; 

end

