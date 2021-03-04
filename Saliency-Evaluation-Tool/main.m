clear all; close all; %clc;

%% set your dataset path and saliency map result path.
CACHE = 'cache_Result_save/';%


if ~exist(CACHE, 'dir')
    mkdir(CACHE);
end
%DS = {'DUTOMRON'}; 
%DS = {'DUTS-TE'};
%DS = {'ECSSD'};
%DS = {'HKU-IS'};
%DS = {'PASCAL'};%
DS = {'SOD'};
%DS = { 'DUTS-TE' ,'ECSSD', 'HKU-IS', 'PASCAL', 'DUTOMRON','SOD'}; %, 'SOCTE'
%% 
 MD_ALL = {'1','2','3','4','5','6'};%model's number
 postfix={'.png','.png','.png','.png','.png','.png','.png','.png'};%};%,'.png'

%%
targetIsFg = true; 
targetIsHigh = true;

for midx=1:length(MD_ALL) %方法数量
    method = MD_ALL{midx};
    if isequal(method, 'GT')||isequal(method, 'PiCANet')...
            ||isequal(method, 'PiCANet-C')||isequal(method, 'PiCANet-R')
        continue;
    end
    for didx=1:length(DS)  %图库数量
        dataset = DS{didx};%指定图库
        
        if exist([CACHE, sprintf('%s_%s.mat',method, dataset)], 'file') %CACHE文件夹中存放性能文件，method_dataset.mat，载入该文件
            load([CACHE, sprintf('%s_%s.mat',method, dataset)]);
        else     %否则，指明GT路径和显著图路径，计算相关性能            
            gtPath = ['E:/20191130_rbd+pf_spimg/data/' dataset '/mask/'];% path of ground truth maps            
            salPath = ['../Result_save/' dataset '/' method '/'];%
            
            if ~exist(salPath, 'dir')
                fprintf('%s %s not exist.\n', dataset, method);
                continue;
            end
            
            %% calculate F-max
            %[~, ~, ~, F_curve, ~] = DrawPRCurve(salPath, postfix{midx}, gtPath, '.png', targetIsFg, targetIsHigh);
            [rec, prec, T, F_curve, iou]=DrawPRCurve(salPath, postfix{midx}, gtPath, '.png', targetIsFg, targetIsHigh);
            %% obtain the total number of image (ground-truth)
            imgFiles = dir(gtPath);
            imgNUM = length(imgFiles)-2;
            if imgNUM<0
                continue;
            end

            %% evaluation score initilization.
            Smeasure=zeros(1,imgNUM)-1;
            Emeasure=zeros(1,imgNUM)-1;
            MAE=zeros(1,imgNUM)-1;

            %% calculate MAE and Smeasure
%            tic;
            for i = 1:imgNUM

                %fprintf('Evaluating: %d/%d\n',i,imgNUM);

                gt_name =  imgFiles(i+2).name;
                sal_name =  replace(imgFiles(i+2).name,'.png', postfix{midx});
                if ~exist([salPath sal_name], 'file')
                    continue;
                end

                %load gt
                gt = imread([gtPath gt_name]);

                if numel(size(gt))>2
                    gt = rgb2gray(gt);
                end
                
                if ~islogical(gt)
                    gt = gt(:,:,1) > 128;
                end
                
                %load salency
                sal  = imread([salPath sal_name]);
                MAE(i) = CalMAE(sal, gt);
                
                %check size
                if size(sal, 1) ~= size(gt, 1) || size(sal, 2) ~= size(gt, 2)
                    sal = imresize(sal,size(gt));
                    imwrite(sal,[salPath sal_name]);
                    % fprintf('Error occurs in the path: %s!!!\n', [salPath sal_name]);
                end
                
                %--------------------
                sal = im2double(sal(:,:,1));

                %normalize sal to [0, 1]
                sal = reshape(mapminmax(sal(:)',0,1),size(sal));

                Smeasure(i) = StructureMeasure(sal,logical(gt));               

                %You can change the method of binarization method. As an example, here just use adaptive threshold.
                 threshold =  graythresh(sal);%我们的用法ostu
                 %{
                 threshold = 2* mean(sal(:));%习惯用法
                 if ( threshold > 1 )
                     threshold = 1;
                 end
                 %}
                 Bi_sal = zeros(size(sal));
                 Bi_sal(sal>threshold)=1;
                 Emeasure(i) = Enhancedmeasure(Bi_sal,gt);

            end

 %           toc;

            %%
            Smeasure(Smeasure==-1) = [];
            Emeasure(Emeasure==-1) = [];
            MAE(MAE==-1) = [];
            
            %%
            Sm = mean2(Smeasure);
            Fm = max(F_curve);
            Em = mean2(Emeasure);
            mae = mean2(MAE);

            Sm_std = std2(Smeasure);
            Em_std = std2(Emeasure);
            mae_std = std2(MAE);
            
            %%
            if (~isnan(Fm)||~isnan(mae)||~isnan(Sm)||~isnan(Em))                
                save([CACHE, sprintf('%s_%s.mat',method, dataset)], ...
                    'Fm', 'Sm', 'Sm_std', 'Em', 'Em_std', 'mae', 'mae_std');
                save([CACHE, sprintf('%s_%s_all.mat',method, dataset)], ...
                    'F_curve', 'Smeasure', 'Em', 'MAE');
            end
            
        end
        fprintf('(%s %s ) MAE: %.3f+%.3f;  Smeasure: %.3f+%.3f; Fm: %.3f; Em: %.3f+%.3f\n', ...
            method, dataset, mae, mae_std, Sm, Sm_std, Fm, Em, Em_std);
%         fprintf('(%s %s Dataset) Fmeasure: %.3f; Smeasure: %.3f; MAE: %.3f\n', ...
%             method, dataset, Fm, Sm, mae);
    end
end
