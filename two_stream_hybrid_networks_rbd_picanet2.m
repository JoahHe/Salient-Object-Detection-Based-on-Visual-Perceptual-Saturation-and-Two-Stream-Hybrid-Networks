clc;
close all;

addpath('.\Funcs\RBD\Funcs\SLIC');%RBD算法用到超像素处理\SLIC'
addpath('.\Funcs');

%% 1. Parameter Settings
RES = 'Result_save';%
if ~exist(RES, 'dir')
    mkdir(RES);
end
DIRS = '.\data';%image and gt
dirs = dir(fullfile(DIRS));
alg_path = '.\RES_other_model';
alg_name2 = 'PiCANet2';%'AFNet';%
alg_name3 = 'PiCANet';

for dir_k = 3%:length(dirs)
        dir_name = dirs(dir_k).name;%image name
        dir_name
        Other_alg_dir = fullfile(alg_path,dir_name);
        
        SRC = fullfile(DIRS,dir_name,'image');
        srcSuffix = '.jpg';%suffix for your input image
        files = dir(fullfile(SRC, strcat('*', srcSuffix)));
tic
        for k=1:length(files)
            disp(k);
            srcName = files(k).name;
            noSuffixName = srcName(1:end-length(srcSuffix));

            srcImg = imread(fullfile(SRC, srcName));        %% read Image
            [r,c,d]=size(srcImg); 
            if d<3
              srcImg(:,:,1)=srcImg(:,:,1);
              srcImg(:,:,2)=srcImg(:,:,1);
              srcImg(:,:,3)=srcImg(:,:,1);
            end         
            if r>512  
                srcImg = imresize(srcImg,512/r);
            elseif c>512
                srcImg = imresize(srcImg,512/c);   
            end
            [r,c,d]=size(srcImg); 
           
         %% Saliency map by RBD 
            smp1 = imgseg_rbd(srcImg);%rbd modified
            update_num = 1;
            RES3_2 = fullfile(RES,strcat(dir_name),num2str(update_num));  %smp1
            if ~exist(RES3_2, 'dir')
                mkdir(RES3_2);
            end
            imgName=fullfile(RES3_2, strcat(noSuffixName, '.png'));    
            imwrite(smp1, imgName);
                        
            smp2 = im2double(imread(fullfile(Other_alg_dir, alg_name2, strcat(noSuffixName,'.png'))));%picanet产生的smp，每个像素值在[0,1]
            if (size(smp2,3)>1)
                smp2 = smp2(:,:,1);
            end      
            update_num = 2;
            RES3_2 = fullfile(RES,strcat(dir_name),num2str(update_num));  %smp2  picanet
            if ~exist(RES3_2, 'dir')
                mkdir(RES3_2);
            end
            imgName=fullfile(RES3_2, strcat(noSuffixName, '.png'));    
            imwrite(smp2, imgName);            
       
            smp3 = im2double(imread(fullfile(Other_alg_dir, alg_name3, strcat(noSuffixName,'.png'))));            
            if (size(smp3,3)>1)
                smp3 = smp3(:,:,1);
            end
            smp3= imresize(smp3,size(smp1));
            update_num = 3;
            RES3_2 = fullfile(RES,strcat(dir_name),num2str(update_num));  %smp3 f3net
            if ~exist(RES3_2, 'dir')
                mkdir(RES3_2);
            end
            imgName=fullfile(RES3_2, strcat(noSuffixName, '.png'));    
            imwrite(smp3, imgName);%存放smp3,以便做性能比较  
          
      %%  two-branch:smp1 and smp2, two-branch:smp1 and smp3, fusion and then positive feedback
            [~,smp4] = zfk_BW_in_2new(im2double(srcImg),smp1,smp2,0.1);%rbd + picanet
            [~,smp5] = zfk_BW_in_2new(im2double(srcImg),smp1,smp3,0.1);%rbd + f3net
            
            update_num = 4;
            RES3_2 = fullfile(RES,strcat(dir_name),num2str(update_num)); 
            if ~exist(RES3_2, 'dir')
                mkdir(RES3_2);
            end
            imgName=fullfile(RES3_2, strcat(noSuffixName, '.png'));    
            imwrite(smp4, imgName);
            
            update_num = 5;
            RES3_2 = fullfile(RES,strcat(dir_name),num2str(update_num)); 
            if ~exist(RES3_2, 'dir')
                mkdir(RES3_2);
            end
            imgName=fullfile(RES3_2, strcat(noSuffixName, '.png'));    
            imwrite(smp5, imgName);% 
         
          %%  two-stream: smp4 and smp5 fusion
            smp6=mat2gray(smp4+smp5);%added %two-stream fusion
 
            update_num = 6;
            RES3_2 = fullfile(RES,strcat(dir_name),num2str(update_num));  
            if ~exist(RES3_2, 'dir')
                mkdir(RES3_2);
            end
            imgName=fullfile(RES3_2, strcat(noSuffixName, '.png'));    
            imwrite(smp6, imgName);%smp6  
           
            BW3 = imbinarize(smp6);%sMap_final二值化，得到目标区域BW3             
            BW3 = select_max_region(BW3,5);
            update_num = 7;
            RES3_2 = fullfile(RES,strcat(dir_name),num2str(update_num));  
            if ~exist(RES3_2, 'dir')
                mkdir(RES3_2);
            end
            imgName=fullfile(RES3_2, strcat(noSuffixName, '.png'));    
            imwrite(BW3, imgName);%BW of smp6 
    %% two-stream: new annotation by comparison of binary masks of smp4 and smp5
     RES5=fullfile(RES,strcat(dir_name),'new_samples');
     if ~exist(RES5, 'dir')
         mkdir(RES5);
     end
     num=0;
     
     BW1 = imbinarize(smp4);%smp5二值化，得到目标区域BW1             
     BW1 = select_max_region(BW1,5);%二值化图中，选最大面积的2个连通域做感兴趣的目标
     BW2 = imbinarize(smp5);%smp6二值化，得到目标区域BW2             
     BW2 = select_max_region(BW2,5);
     BW3 = imbinarize(smp6);%sMap_final二值化，得到目标区域BW3             
     BW3 = select_max_region(BW3,5);
     
     Fcomp = ComputeFMeasure_1(BW1,BW2);
     if Fcomp>=0.85%0.90
              bwName = fullfile(RES5, strcat(noSuffixName,'.png'));%bw labels
              imwrite(BW3, bwName);%
              imgName = fullfile(RES5,strcat(noSuffixName,'.jpg'));%color images
              imwrite(srcImg, imgName);%
              num=num+1;
              update_num = 8;
              RES3_2 = fullfile(RES,strcat(dir_name),num2str(update_num));  
              if ~exist(RES3_2, 'dir')
                  mkdir(RES3_2);
              end
              imgName=fullfile(RES3_2, strcat(noSuffixName, '.png'));    
              imwrite(smp6, imgName);%smp6   
     end
 toc  
 end
end