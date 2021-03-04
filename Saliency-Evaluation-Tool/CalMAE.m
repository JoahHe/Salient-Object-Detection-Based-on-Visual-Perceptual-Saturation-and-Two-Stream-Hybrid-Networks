function mae = CalMAE_0207(smap, gtImg)
% Code Author: Wangjiang Zhu
% Email: wangjiang88119@gmail.com
% Date: 3/24/2014
if size(smap, 1) ~= size(gtImg, 1) || size(smap, 2) ~= size(gtImg, 2)
    %error('Saliency map and gt Image have different sizes!\n');
    smap = imresize(smap,size(gtImg));
    %gtImg = imresize(gtImg,size(smap));
end
% smap = imresize(smap,[112,112]);
% gtImg = imresize(gtImg,[112,112]);
if ~islogical(gtImg)
    gtImg = gtImg(:,:,1) > 128;
end
smap = im2double(smap(:,:,1));
% smap = 1./(1+exp(-20*(smap-0.5)));
% smap(smap<0.3)=0;
% smap(smap>0.5)=1;
fgPixels = smap(gtImg);
fgErrSum = length(fgPixels) - sum(fgPixels);
bgErrSum = sum(smap(~gtImg));
mae = (fgErrSum + bgErrSum) / numel(gtImg);