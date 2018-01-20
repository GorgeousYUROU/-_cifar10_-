function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - ������� the number of classes 
% inputSize - �������ݴ�С the size N of the input vector
% lambda - Ȩ��˥��ϵ�� weight decay parameter
% data -�������ݼ� the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - �������ݵı�ǩ an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);%�������ݼ�������

groundTruth = full(sparse(labels, 1:numCases, 1));%����һ��100*100�ľ������ĵ�labels(i)�е�i�е�Ԫ��ֵΪ1������ȫΪ0������iΪ1��numCases������1��100
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

M = bsxfun(@minus,theta*data,max(theta*data, [], 1));% max(theta*data, [], 1)����theta*dataÿһ�е����ֵ������ֵΪ������
                                                     % theta*data��ÿ��Ԫ��ֵ����ȥ���Ӧ�е����ֵ��������ÿһ�е����ֵ����Ϊ0��
                                                     % ��һ����Ŀ���Ƿ�ֹ��һ������ָ������ʱ���
M = exp(M);
p = bsxfun(@rdivide, M, sum(M));
cost = -1/numCases * groundTruth(:)' * log(p(:)) + lambda/2 * sum(theta(:) .^ 2);
%daoshu gradient
thetagrad = -1/numCases * (groundTruth - p) * data' + lambda * theta; 



% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end