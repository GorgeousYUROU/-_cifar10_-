%% CS294A/CS294W Self-taught Learning Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  self-taught learning. You will need to complete code in feedForwardAutoencoder.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises.
%
%% ======================================================================
%  STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.
function stlExercise()

tic
inputSize  = 32 * 32*3;
numLabels  = 10;
hiddenSize = 200;
sparsityParam = 0.1; % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
                     %  in the lecture notes). 
lambda = 3e-3;       % weight decay parameter       
beta = 3;            % weight of sparsity penalty term   
maxIter = 200;

%% ======================================================================
%  STEP 1: Load data from the MNIST database
%
%  This loads our training and test data from the MNIST database files.
%  We have sorted the data for you in this so that you will not have to
%  change it.

% Load cifar10 database files
load('has_label_train_batch_1_3.mat');
trainData=data';
trainLabels=labels+1;
clear data labels ;
load('no_label_train_batch_4_5.mat');
unlabeledData = data';
clear data;
load('self_test_batch.mat');
testData  = data';
testLabels =labels+ 1;   % Shift Labels to the Range 1-5
clear data testdata testlabels;

%% ======================================================================
%  STEP 2: Train the sparse autoencoder
%  This trains the sparse autoencoder on the unlabeled training
%  images. 

%  按均匀分布随机初始化theta参数   Randomly initialize the parameters
theta = initializeParameters(hiddenSize, inputSize);

%% ----------------- YOUR CODE HERE ----------------------
%  Find opttheta by running the sparse autoencoder on
%  unlabeledTrainingImages
%  利用L-BFGS算法，用无标签数据集来训练稀疏自动编码器

opttheta = theta; 

addpath minFunc/
options.Method = 'lbfgs';
options.maxIter = 400;
options.display = 'on';
[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
      inputSize, hiddenSize, ...
      lambda, sparsityParam, ...
      beta, unlabeledData), ...
      theta, options);


%% -----------------------------------------------------
                          
% Visualize weights
W1 = reshape(opttheta(1:hiddenSize * inputSize), hiddenSize, inputSize);
displayColorNetwork(W1');

%%======================================================================
%% STEP 3: 从有标签数据集中提取特征 Extract Features from the Supervised Dataset
%  
%  You need to complete the code in feedForwardAutoencoder.m so that the 
%  following command will extract features from the data.

trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       trainData);

testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       testData);

%%======================================================================
%% STEP 4: Train the softmax classifier

softmaxModel = struct;  
%% ----------------- YOUR CODE HERE ----------------------
%  Use softmaxTrain.m from the previous exercise to train a multi-class
%  classifier. 
%  利用L-BFGS算法，用从有标签训练数据集中提取的特征及其标签，训练softmax回归模型，

%  Use lambda = 1e-4 for the weight regularization for softmax
lambda = 1e-4;
inputSize = hiddenSize;
numClasses = 10; %unique为找出向量中的非重复元素并进行排序
% You need to compute softmaxModel using softmaxTrain on trainFeatures and
% trainLabels

options.maxIter = 100; %最大迭代次数
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            trainFeatures, trainLabels, options);





%% -----------------------------------------------------


%%======================================================================
%% STEP 5: Testing 

%% ----------------- YOUR CODE HERE ----------------------
% Compute Predictions on the test set (testFeatures) using softmaxPredict
% and softmaxModel

[pred] = softmaxPredict(softmaxModel, testFeatures);



%% -----------------------------------------------------

% Classification Score
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));
toc
% (note that we shift the labels by 1, so that digit 0 now corresponds to
%  label 1)
%
% Accuracy is the proportion of correctly classified images
% The results for our implementation was:
%
% Accuracy: 98.3%
%
end
%