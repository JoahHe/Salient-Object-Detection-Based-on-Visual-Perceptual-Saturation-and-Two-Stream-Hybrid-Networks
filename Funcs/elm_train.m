function [InputWeight,OutputWeight,BiasofHiddenNeurons,TrainingTime] = elm_train(training_label, train_data, NumberofHiddenNeurons, ActivationFunction)
% Usage: elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] =
% elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
%
% Input:
% TrainingData_File     - Filename of training data set
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
% TrainingTime          - Time (seconds) spent on training ELM
% MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
% FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
% neurons; neuron 5 has the highest output means input belongs to 5-th class
%
% Sample1 regression: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm('sinc_train', 'sinc_test', 0, 20, 'sig')
% Sample2 classification: elm('diabetes_train', 'diabetes_test', 1, 20, 'sig')
%
%%%%%%%%%%% Load training dataset
%train_data=load(TrainingData_File);
%T=train_data(:,1)';
T=training_label';
P=train_data(:,1:size(train_data,2))';
%clear train_data;                                   %   Release raw training data array

NumberofTrainingData=size(P,2);
NumberofInputNeurons=size(P,1);

    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(T);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%% Processing the targets of training   预处理输出标记: 将原数据的类别标记，用输出端神经元的组合来进行（如-1，1）编码表示
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;
    
%%%%%%%%%%% Calculate weights & biases
start_time_train=cputime;
%??(1)??????%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
%rand('state',0);
rand(1,2);
na=10;
InputWeight=na*(rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1);
%InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;%MAE for_SL_bw.png: 0.083714,F_score:0.85358,R_score:0.79043,P_score:0.90148
%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
%???2????????????????????????????????????????????????????????????????????????????
%Win=na*(rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1);
%Aorth=orth(Win);
%if NumberofHiddenNeurons<NumberofInputNeurons %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%???
%    [Wpca]=pca(P);
%    InputWeight=Aorth*Wpca(1:NumberofHiddenNeurons,:);
%else
%    InputWeight=Aorth;
%end
%???3??????????????????????????????????????????????????????????????????????????
%if NumberofHiddenNeurons>NumberofInputNeurons %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%???
%    InputWeight=orth(2*rand(NumberofInputNeurons+1,NumberofHiddenNeurons)-1);
%%else
%    Aorth=orth(2*rand(NumberofInputNeurons+1,NumberofHiddenNeurons)-1);    
%    if size(P,2)>size(P,1)
%        [~,Wpca]=pca(P');
%    else
%        [~,Wpca]=pca(P);
%    end
%    InputWeight=Aorth*Wpca(1:NumberofHiddenNeurons,:);
%end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
                                        %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;
%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
       H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        %%%%%%%% More activation functions can be added here                
end
clear tempH;                                        %   Release the temporary array for calculation of hidden neuron output matrix H
H=cat(1,H,P,ind);%构造调和极速学习机的H矩阵
clear P;
%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
OutputWeight=pinv(H') * T';                        % slower implementation
%OutputWeight=inv(H * H') * H * T';% faster implementation %有问题？此句分类结果有问题
end_time_train=cputime;
TrainingTime=end_time_train-start_time_train;        %   Calculate CPU time (seconds) spent for training ELM

