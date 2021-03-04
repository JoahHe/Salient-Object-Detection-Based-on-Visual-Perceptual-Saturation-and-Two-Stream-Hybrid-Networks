function Iout = rbd(srcImg,pixNumInSP)

%% 1. Parameter Settings
doFrameRemoving = true;
doNormalize = true;
useSP = true;           %You can set useSP = false to use regular grid for speed consideration

%% 2. Saliency Map Calculation
    if doFrameRemoving
        [noFrameImg, frameRecord] = removeframe(srcImg, 'sobel');
        [h, w, chn] = size(noFrameImg);
    else
        noFrameImg = srcImg;
        [h, w, chn] = size(noFrameImg);
    end       
    frameRecord = [h, w, 1, h, 1, w];

    
    %% Segment input rgb image into patches (SP/Grid)
    %pixNumInSP = 550;                           %pixels in each superpixel
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
    
    %smapName=fullfile(RES, strcat(noSuffixName, '_wCtr_Optimized.png'));
    Iout = SaveSaliencyMap(optwCtr, pixelList, frameRecord, meanPos,doNormalize);
end

function Iout=SaveSaliencyMap(feaVec, pixelList, frameRecord, meanPos,doNormalize)
% Fill back super-pixel values to image pixels and save into .png images

% Code Author: Wangjiang Zhu
% Email: wangjiang88119@gmail.com
% Date: 3/24/2014

if (~iscell(pixelList))
    error('pixelList should be a cell');
end

if (nargin < 5)
    doNormalize = true;
end

if (nargin < 6)
    fill_value = 0;
end

h = frameRecord(1);
w = frameRecord(2);

top = frameRecord(3);
bot = frameRecord(4);
left = frameRecord(5);
right = frameRecord(6);

partialH = bot - top + 1;
partialW = right - left + 1;
%partialImg = CreateImageFromSPs_1(feaVec, pixelList, partialH, partialW, meanPos,doNormalize);
partialImg = CreateImageFromSPs(feaVec, pixelList, partialH, partialW, doNormalize);

if partialH ~= h || partialW ~= w
    feaImg = ones(h, w) * fill_value;
    feaImg(top:bot, left:right) = partialImg;
    Iout=feaImg;
else
    Iout=partialImg;
end
end
