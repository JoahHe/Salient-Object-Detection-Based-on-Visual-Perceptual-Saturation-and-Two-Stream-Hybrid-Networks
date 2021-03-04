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