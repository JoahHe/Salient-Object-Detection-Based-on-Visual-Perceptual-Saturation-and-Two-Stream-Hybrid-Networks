function saliencyMap=phase_fft(inImg)  
%Input:inImg->»Ò¶ÈÍ¼Ïñ
%Output:saliencyMap-¡µÏÔÖøÍ¼

myFFT = fft2(inImg); 
myPhase = angle(myFFT);
saliencyMap = abs(ifft2(exp(i*myPhase))).^2; 
%saliencyMap = abs(ifft2(i*myPhase));