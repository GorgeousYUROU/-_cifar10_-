function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(featureNum, imageNum, poolRow, poolCol)
%                      表示第个featureNum特征与第imageNum张图片的卷积特征池化后的结果保存在矩阵
%                      pooledFeatures(featureNum, imageNum, ：, ：)的第poolRow行第poolCol列
%     

numImages = size(convolvedFeatures, 2);   % 图片数量
numFeatures = size(convolvedFeatures, 1); % 卷积特征数量
convolvedDim = size(convolvedFeatures, 3);% 卷积特征维数

pooledFeatures = zeros(numFeatures, numImages, floor(convolvedDim / poolDim), floor(convolvedDim / poolDim));

% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   numFeatures x numImages x (convolvedDim/poolDim) x (convolvedDim/poolDim) 
%   matrix pooledFeatures, such that
%   pooledFeatures(featureNum, imageNum, poolRow, poolCol) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region 
%   (see http://ufldl/wiki/index.php/Pooling )
%   
%   Use mean pooling here.
% -------------------- YOUR CODE HERE --------------------

resultDim  = floor(convolvedDim / poolDim);
for imageNum = 1:numImages   % 第imageNum张图片
    for featureNum = 1:numFeatures  % 第featureNum个特征
        for poolRow = 1:resultDim
            offsetRow = 1+(poolRow-1)*poolDim;
            for poolCol = 1:resultDim
                offsetCol = 1+(poolCol-1)*poolDim;
                patch = convolvedFeatures(featureNum,imageNum,offsetRow:offsetRow+poolDim-1,...
                    offsetCol:offsetCol+poolDim-1);%取出一个patch
                pooledFeatures(featureNum,imageNum,poolRow,poolCol) = mean(patch(:));
            end
        end
    end
end


end