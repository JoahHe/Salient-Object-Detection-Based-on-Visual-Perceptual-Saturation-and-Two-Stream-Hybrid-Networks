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
