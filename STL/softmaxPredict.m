function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
%����ѵ�����theta���ø�theta���ɴﵽ���10��
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.
theta_x=theta * data;  
[m, pred] = max(theta_x);  
%ֻ��Ҫ�Ƚ�������ʴ��ɹ�ʽ3��֪��ֻ�ü���theta * data���ּ���  







% ---------------------------------------------------------------------

end

