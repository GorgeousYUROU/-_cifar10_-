function convolvedFeatures = cnnConvolve(patchDim, numFeatures, images, W, b, ZCAWhite, meanPatch)
%卷积特征提取：把每一个特征都与每一张大尺寸图片images相卷积，并返回卷积结果
%Returns the convolution of the features given by W and b with the given images
% Parameters:
%  patchDim - patch (feature) dimension
%  numFeatures - number of features
%  images - large images to convolve with, matrix in the form
%           images(r, c, channel, image number)
%  W, b - W, b for features from the sparse autoencoder
%  ZCAWhite, meanPatch - ZCAWhitening and meanPatch matrices used for
%                        preprocessing
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
%                      表示第个featureNum特征与第imageNum张图片相卷的结果保存在矩阵
%                      convolvedFeatures(featureNum, imageNum, ：, ：)的第imageRow行第imageCol列,
%                      而每行和列的大小都为imageDim - patchDim + 1

numImages = size(images, 4);     % 图片数量
imageDim = size(images, 1);      % 每幅图片行数
imageChannels = size(images, 3); % 每幅图片通道数

patchSize = patchDim*patchDim;
assert(numFeatures == size(W,1), 'W should have numFeatures rows');
assert(patchSize*imageChannels == size(W,2), 'W should have patchSize*imageChannels cols');


% Instructions:
%   Convolve every feature with every large image here to produce the 
%   numFeatures x numImages x (imageDim - patchDim + 1) x (imageDim - patchDim + 1) 
%   matrix convolvedFeatures, such that 
%   convolvedFeatures(featureNum, imageNum, imageRow, imageCol) is the
%   value of the convolved featureNum feature for the imageNum image over
%   the region (imageRow, imageCol) to (imageRow + patchDim - 1, imageCol + patchDim - 1)



WT = W*ZCAWhite;           % 等效的网络权值
b_mean = b - WT*meanPatch; % 等效偏置项



convolvedFeatures = zeros(numFeatures, numImages, imageDim - patchDim + 1, imageDim - patchDim + 1);
for imageNum = 1:numImages
  for featureNum = 1:numFeatures

    % convolution of image with feature matrix for each channel
    convolvedImage = zeros(imageDim - patchDim + 1, imageDim - patchDim + 1);
    for channel = 1:3

      % Obtain the feature (patchDim x patchDim) needed during the convolution
      % ---- YOUR CODE HERE ----
      offset = (channel-1)*patchSize;
      feature = reshape(WT(featureNum,offset+1:offset+patchSize), patchDim, patchDim);%取一个权值图像块出来
      im  = images(:,:,channel,imageNum);
      
      
      % Flip the feature matrix because of the definition of convolution, as explained later
      feature = rot90(squeeze(feature),2);
      
      % Obtain the image
      im = squeeze(images(:, :, channel, imageNum));

      % Convolve "feature" with "im", adding the result to convolvedImage
      % be sure to do a 'valid' convolution
      % ---- YOUR CODE HERE ----

      convolvedoneChannel = conv2(im, feature, 'valid');    % 单个特征分别与所有图片相卷
      convolvedImage = convolvedImage + convolvedoneChannel;% 直接把3通道的值加起来，理由：3通道相当于有3个feature-map，类似于cnn第2层以后的输入。
            
      % ------------------------

    end
    
    % Subtract the bias unit (correcting for the mean subtraction as well)
    % Then, apply the sigmoid function to get the hidden activation
    % ---- YOUR CODE HERE ----

     convolvedImage = sigmoid(convolvedImage+b_mean(featureNum));
%     convolvedImage = relu(convolvedImage+b_mean(featureNum));
    
    % -----------------------
    % The convolved feature is the sum of the convolved values for all channels
    convolvedFeatures(featureNum, imageNum, :, :) = convolvedImage;
  end
end


end
function sigm = sigmoid(x)
    sigm = 1./(1+exp(-x));
end
%relu函数
function re = relu(x) 
re = max(x,0); %和0比较，比0大选x
end 
