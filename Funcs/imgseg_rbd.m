function smp1 = imgseg_rbd(srcImg)

    [r,c,~]= size(srcImg);
    % RBD
    pixNumInSP = min_etp_spimg(srcImg);%
    %pixNumInSP = 500;   %
    [smp_rbd,~] = rbd(srcImg,pixNumInSP);
    smp1=imresize(smp_rbd,[r,c]);%output smp 
end

function [Iout,spimg] = rbd(srcImg,pixNumInSP)

%% 1. Parameter Settings
doFrameRemoving = false;%true;
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
    %pixNumInSP = 500;                           %pixels in each superpixel
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
   %% Super pixels map
    spimg(:,:,1) = CreateImageFromSPs(meanRgbCol(:,1), pixelList, h, w, doNormalize);
    spimg(:,:,2) = CreateImageFromSPs(meanRgbCol(:,2), pixelList, h, w, doNormalize);
    spimg(:,:,3) = CreateImageFromSPs(meanRgbCol(:,3), pixelList, h, w, doNormalize);
    %figure;imshow(spimg);title('Superpixel image');
        
    %% Saliency Optimization
    [bgProb, bdCon, bgWeight] = EstimateBgProb(colDistM, adjcMatrix, bdIds, clipVal, geoSigma);
    wCtr = CalWeightedContrast(colDistM, posDistM, bgProb);
    optwCtr = SaliencyOptimization(adjcMatrix, bdIds, colDistM, neiSigma, bgWeight, wCtr);
    
    %smapName=fullfile(RES, strcat(noSuffixName, '_wCtr_Optimized.png'));
    Iout = SaveSaliencyMap(optwCtr, pixelList, frameRecord, meanPos,doNormalize);
end

function [bgProb, bdCon, bgWeight] = EstimateBgProb(colDistM, adjcMatrix, bdIds, clipVal, geoSigma)
% Estimate background probability using boundary connectivity

% Code Author: Wangjiang Zhu
% Email: wangjiang88119@gmail.com
% Date: 3/24/2014

bdCon = BoundaryConnectivity(adjcMatrix, colDistM, bdIds, clipVal, geoSigma, true);

bdConSigma = 1; %sigma for converting bdCon value to background probability
fgProb = exp(-bdCon.^2 / (2 * bdConSigma * bdConSigma)); %Estimate bg probability
bgProb = 1 - fgProb;

bgWeight = bgProb;
% Give a very large weight for very confident bg sps can get slightly
% better saliency maps, you can turn it off.
fixHighBdConSP = true;
highThresh = 3;
if fixHighBdConSP
    bgWeight(bdCon > highThresh) = 1000;
end

end

function optwCtr = SaliencyOptimization(adjcMatrix, bdIds, colDistM, neiSigma, bgWeight, fgWeight)
% Solve the least-square problem in Equa(9) in our paper

% Code Author: Wangjiang Zhu
% Email: wangjiang88119@gmail.com
% Date: 3/24/2014

adjcMatrix_nn = LinkNNAndBoundary(adjcMatrix, bdIds);
colDistM(adjcMatrix_nn == 0) = Inf;
Wn = Dist2WeightMatrix(colDistM, neiSigma);      %smoothness term
mu = 0.1;                                                   %small coefficients for regularization term
W = Wn + adjcMatrix * mu;                                   %add regularization term
D = diag(sum(W));

bgLambda = 5;   %global weight for background term, bgLambda > 1 means we rely more on bg cue than fg cue.
E_bg = diag(bgWeight * bgLambda);       %background term
E_fg = diag(fgWeight);          %foreground term

spNum = length(bgWeight);
optwCtr =(D - W + E_bg + E_fg) \ (E_fg * ones(spNum, 1));
end

function sMap=SaveSaliencyMap(feaVec, pixelList, frameRecord, doNormalize, fill_value)
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
partialImg = CreateImageFromSPs(feaVec, pixelList, partialH, partialW, doNormalize);

if partialH ~= h || partialW ~= w
    feaImg = ones(h, w) * fill_value;
    feaImg(top:bot, left:right) = partialImg;
    %imwrite(feaImg, imgName);
    sMap=feaImg;
else
    sMap=partialImg;
    %imwrite(partialImg, imgName);
end
end

function [img, spValues] = CreateImageFromSPs(spValues, pixelList, height, width, doNormalize)
% create an image from its superpixels' values
% spValues is all superpixel's values, e.g., saliency
% pixelList is a cell (with the same size as spValues) of pixel index arrays

% Code Author: Wangjiang Zhu
% Email: wangjiang88119@gmail.com
% Date: 3/24/2014

if (~iscell(pixelList))
    error('pixelList should be a cell');
end

if (length(pixelList) ~= length(spValues))
    error('different sizes in spValues and pixelList');
end

if (nargin < 5)
    doNormalize = true;
end

minVal = min(spValues);
maxVal = max(spValues);
if doNormalize
    spValues = (spValues - minVal) / (maxVal - minVal + eps);
else
    if minVal < -1e-6 || maxVal > 1 + 1e-6
        error('feature values do not range from 0 to 1');
    end
end

img = zeros(height, width);
for i=1:length(pixelList)
    img(pixelList{i}) = spValues(i);
end
end


