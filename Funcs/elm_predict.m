function [out_label, TestingTime] = elm_predict(testing_data, InputWeight,OutputWeight,NumberofHiddenNeurons, BiasofHiddenNeurons, ActivationFunction)
% Usage: elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
%
% Input:
% TestingData_File      - Filename of testing data set
% Elm_Type              - classification
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
% ActivationFunction    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                           'hardlim' for Hardlim function
%                           'tribas' for Triangular basis function
%                           'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
%
% Output: 
% TestingTime           - Time (seconds) spent on predicting ALL testing data
%
% MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
% FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
% neurons; neuron 5 has the highest output means input belongs to 5-th class
%% Sample2 classification: elm('diabetes_train', 'diabetes_test', 20, 'sig')

%%%%%%%%%%% Load testing dataset
%test_data=load(TestingData_File);
%TV.P=test_data(:,1:size(test_data,2))';%S';%S=TV.P';
%clear test_data;                                    %   Release raw testing data array
TV.P=testing_data(:,1:size(testing_data,2))';

NumberofTestingData=size(TV.P,2);
%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;
tempH_test=InputWeight*TV.P;
%clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);        
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);        
        %%%%%%%% More activation functions can be added here        
end
H_test=cat(1,H_test,TV.P,ind);%构造调和极速学习机的H矩阵
clear TV.P;  

TY=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data
end_time_test=cputime;
TestingTime=end_time_test-start_time_test;           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data

for i = 1 : NumberofTestingData
    [x, label_index_actual]=max(TY(:,i));%取输出端口中最大值的输出端的标记为类别标记
    out_label(i)=label_index_actual;
end
%temp = fopen('d:\\elm_result.trn','w');%保存为文本文件
%for i=1:NumberofTestingData
%    fprintf(temp,'%d\n',out_label(i));
%end
%fclose(temp);    
    