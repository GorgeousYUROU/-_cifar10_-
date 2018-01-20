
%% STEP 0: Initialization
%  Here we initialize some parameters used for the exercise.

imageChannels = 3;     % number of channels (rgb, so 3)

patchDim   = 8;          % patch dimension
numPatches = 60000;   % number of patches

visibleSize = patchDim * patchDim * imageChannels;  % number of input units 
outputSize  = visibleSize;   % number of output units
hiddenSize  = 400;           % number of hidden units 

sparsityParam = 0.035; % desired average activation of the hidden units.
lambda = 3e-3;         % weight decay parameter       
beta = 5;              % weight of sparsity penalty term       

epsilon = 0.1;           % epsilon for ZCA whitening
%% STEP 2a: Load patches
%  In this step, we load 100k patches sampled from the STL10 dataset and
%  visualize them. Note that these patches have been scaled to [0,1]
% load('data_batch_1.mat');
% data=double(data)/255;
% patches=sampleIMAGES(data,patchDim,numPatches);
patches=sampleIMAGES(patchDim,numPatches);
displayColorNetwork(patches(:, 1:100));

%% STEP 2b: 预处理 Apply preprocessing
%  In this sub-step, we preprocess the sampled patches, in particular, 
%  ZCA whitening them. 
% 
%  In a later exercise on convolution and pooling, you will need to replicate 
%  exactly the preprocessing steps you apply to these patches before 
%  using the autoencoder to learn features on them. Hence, we will save the
%  ZCA whitening and mean image matrices together with the learned features
%  later on.

% Subtract mean patch (hence zeroing the mean of the patches)
meanPatch = mean(patches, 2);  %为什么是对每行求平均，以前是对每列即每个样本求平均呀？因为以前是灰度图，现在是彩色图，如果现在对每列平均就是对三个通道求平均，这肯定不行
patches = bsxfun(@minus, patches, meanPatch);

% Apply ZCA whitening
sigma = patches * patches' / numPatches; %协方差矩阵
[u, s, v] = svd(sigma);
ZCAWhite = u * diag(1 ./ sqrt(diag(s) + epsilon)) * u';
patches = ZCAWhite * patches;

figure;
displayColorNetwork(patches(:, 1:100));

%% STEP 2c: Learn features
%  You will now use your sparse autoencoder (with linear decoder) to learn
%  features on the preprocessed patches. This should take around 45 minutes.

theta = initializeParameters(hiddenSize, visibleSize);

% Use minFunc to minimize the function
addpath minFunc/

options = struct;
options.Method = 'lbfgs'; 
options.maxIter = 200;
options.display = 'on';

[optTheta, cost] = minFunc( @(p) sparseAutoencoderLinearCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta, options);

% Save the learned features and the preprocessing matrices for use in 
% the later exercise on convolution and pooling
fprintf('Saving learned features and preprocessing matrices...\n');                          
save('STL10Features.mat', 'optTheta', 'ZCAWhite', 'meanPatch');
fprintf('Saved\n');

%% STEP 2d: Visualize learned features

W = reshape(optTheta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
b = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
figure;
displayColorNetwork( (W*ZCAWhite)');