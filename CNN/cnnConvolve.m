function convolvedFeatures = cnnConvolve(patchDim, numFeatures, images, W, b, ZCAWhite, meanPatch)
%���������ȡ����ÿһ����������ÿһ�Ŵ�ߴ�ͼƬimages�����������ؾ�����
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
%                      ��ʾ�ڸ�featureNum�������imageNum��ͼƬ���Ľ�������ھ���
%                      convolvedFeatures(featureNum, imageNum, ��, ��)�ĵ�imageRow�е�imageCol��,
%                      ��ÿ�к��еĴ�С��ΪimageDim - patchDim + 1

numImages = size(images, 4);     % ͼƬ����
imageDim = size(images, 1);      % ÿ��ͼƬ����
imageChannels = size(images, 3); % ÿ��ͼƬͨ����

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



WT = W*ZCAWhite;           % ��Ч������Ȩֵ
b_mean = b - WT*meanPatch; % ��Чƫ����



convolvedFeatures = zeros(numFeatures, numImages, imageDim - patchDim + 1, imageDim - patchDim + 1);
for imageNum = 1:numImages
  for featureNum = 1:numFeatures

    % convolution of image with feature matrix for each channel
    convolvedImage = zeros(imageDim - patchDim + 1, imageDim - patchDim + 1);
    for channel = 1:3

      % Obtain the feature (patchDim x patchDim) needed during the convolution
      % ---- YOUR CODE HERE ----
      offset = (channel-1)*patchSize;
      feature = reshape(WT(featureNum,offset+1:offset+patchSize), patchDim, patchDim);%ȡһ��Ȩֵͼ������
      im  = images(:,:,channel,imageNum);
      
      
      % Flip the feature matrix because of the definition of convolution, as explained later
      feature = rot90(squeeze(feature),2);
      
      % Obtain the image
      im = squeeze(images(:, :, channel, imageNum));

      % Convolve "feature" with "im", adding the result to convolvedImage
      % be sure to do a 'valid' convolution
      % ---- YOUR CODE HERE ----

      convolvedoneChannel = conv2(im, feature, 'valid');    % ���������ֱ�������ͼƬ���
      convolvedImage = convolvedImage + convolvedoneChannel;% ֱ�Ӱ�3ͨ����ֵ�����������ɣ�3ͨ���൱����3��feature-map��������cnn��2���Ժ�����롣
            
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
%relu����
function re = relu(x) 
re = max(x,0); %��0�Ƚϣ���0��ѡx
end 
