clear, clc, 
close all
addpath(genpath('Funcs'));

%% 1. Parameter Settings
doFrameRemoving = true;
useSP = true;           %You can set useSP = false to use regular grid for speed consideration
doMAEEval = true;       %Evaluate MAE measure after saliency map calculation
doPRCEval = true;       %Evaluate PR Curves after saliency map calculation

SRC = 'Data\SED2_ImgGt';       %Path of input images
RES = 'Data\SED2_Res';       %Path for saving saliency maps
srcSuffix = '.jpg';     %suffix for your input image

if ~exist(RES, 'dir')
    mkdir(RES);
end
%% 2. Saliency Map Calculation
files = dir(fullfile(SRC, strcat('*', srcSuffix)));
for k=1%:length(files)
    disp(k);
    srcName = files(k).name;
    noSuffixName = srcName(1:end-length(srcSuffix));
    %% Pre-Processing: Remove Image Frames
    srcImg = imread(fullfile(SRC, srcName));
    if doFrameRemoving
        [noFrameImg, frameRecord] = removeframe(srcImg, 'sobel');
        [h, w, chn] = size(noFrameImg);
    else
        noFrameImg = srcImg;
        [h, w, chn] = size(noFrameImg);
        frameRecord = [h, w, 1, h, 1, w];
    end
    
    %% Segment input rgb image into patches (SP/Grid)
    pixNumInSP = 550;                           %pixels in each superpixel
    spnumber = round( h * w / pixNumInSP );     %super-pixel number for current image
    
    if useSP
        [idxImg, adjcMatrix, pixelList] = SLIC_Split(noFrameImg, spnumber);
    else
        [idxImg, adjcMatrix, pixelList] = Grid_Split(noFrameImg, spnumber);        
    end
    %% Get super-pixel properties
    spNum = size(adjcMatrix, 1);
    meanRgbCol = GetMeanColor(noFrameImg, pixelList);
    meanLabCol = colorspace('Lab<-', double(meanRgbCol)/255);
    meanPos = GetNormedMeanPos(pixelList, h, w);
    bdIds = GetBndPatchIds(idxImg);
    colDistM = GetDistanceMatrix(meanLabCol);
    posDistM = GetDistanceMatrix(meanPos);
    [clipVal, geoSigma, neiSigma] = EstimateDynamicParas(adjcMatrix, colDistM);
    
    %% Saliency Optimization
    [bgProb, bdCon, bgWeight] = EstimateBgProb(colDistM, adjcMatrix, bdIds, clipVal, geoSigma);
    wCtr = CalWeightedContrast(colDistM, posDistM, bgProb);
    optwCtr = SaliencyOptimization(adjcMatrix, bdIds, colDistM, neiSigma, bgWeight, wCtr);
    
    smapName=fullfile(RES, strcat(noSuffixName, '_wCtr_Optimized.png'));
    Iout = SaveSaliencyMap(optwCtr, pixelList, frameRecord, smapName, true);
%{
    %% Saliency Filter
    [cmbVal, contrast, distribution] = SaliencyFilter(colDistM, posDistM, meanPos);
    
    smapName=fullfile(RES, strcat(noSuffixName, '_SF.png'));
    Iout2 = SaveSaliencyMap(cmbVal, pixelList, frameRecord, smapName, true);    
  
    %% Geodesic Saliency
    geoDist = GeodesicSaliency(adjcMatrix, bdIds, colDistM, posDistM, clipVal);
    
    smapName=fullfile(RES, strcat(noSuffixName, '_GS.png'));
    Iout3 = SaveSaliencyMap(geoDist, pixelList, frameRecord, smapName, true);
    
    %% Manifold Ranking
    [stage2, stage1, bsalt, bsalb, bsall, bsalr] = ManifoldRanking(adjcMatrix, idxImg, bdIds, colDistM);
    
    smapName=fullfile(RES, strcat(noSuffixName, '_MR_stage2.png'));
    Iout4 = SaveSaliencyMap(stage2, pixelList, frameRecord, smapName, true);

   Iout= mat2gray(Iout1+Iout2+Iout3+Iout4);
   smapName=fullfile(RES, strcat(noSuffixName, '_Mix.png'));
   imwrite(Iout, smapName);
 %}  
end

%% 3. Evaluate MAE
if doMAEEval
    GT = SRC;
    gtSuffix = '.png';
    CalMeanMAE(RES, '_wCtr_Optimized.png', GT, gtSuffix);
    CalMeanMAE(RES, '_SF.png', GT, gtSuffix);
    CalMeanMAE(RES, '_GS.png', GT, gtSuffix);
    CalMeanMAE(RES, '_MR_stage2.png', GT, gtSuffix);
    %CalMeanMAE(RES, '_Mix.png', GT, gtSuffix);
end

%% 4. Evaluate PR Curve
if doPRCEval
    GT = SRC;
    gtSuffix = '.png';
    figure, hold on;
    DrawPRCurve(RES, '_wCtr_Optimized.png', GT, gtSuffix, true, true, 'm');
    DrawPRCurve(RES, '_SF.png', GT, gtSuffix, true, true, 'g');
    DrawPRCurve(RES, '_GS.png', GT, gtSuffix, true, true, 'b');
    DrawPRCurve(RES, '_MR_stage2.png', GT, gtSuffix, true, true, 'k');
    %DrawPRCurve(RES, '_Mix.png', GT, gtSuffix, true, true, 'r');
    hold off;
    grid on;
    lg = legend({'wCtr\_opt'; 'SF'; 'GS'; 'MR';});
    set(lg, 'location', 'southwest');
end