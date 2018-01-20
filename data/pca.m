function m = pca(dataset,vk)
%%================================================================
%% Step 0a: Load data
%  Here we provide the code to load natural image data into x.
%  x will be a 144 * 10000 matrix, where the kth column x(:, k) corresponds to
%  the raw image data from the kth 12x12 image patch sampled.
%  You do not need to change the code below.
name=dataset;
dataset=[dataset,'.mat'];
load(dataset);%������תΪdouble������
close all;%������б���
data = double(data);
labels=double(labels);
x = data';
y=labels;
clear data labels;
% data: [10000x3072 double]
% labels: [10000x3072 double]


%%================================================================
%% Step 0b: Zero-mean the data (by row)
%  You can make use of the mean and repmat/bsxfun functions.

% -------------------- YOUR CODE HERE -------------------- 
avg=mean(x,1);          %�˴������ÿһ�������ľ�ֵ������ÿһ�еľ�ֵ��һ������10000����ֵ
x=x-repmat(avg,size(x,1),1); %��ȥ��ֵ���Ա�֤����pca���ɷݷ����ľ�ֵΪ0

%%================================================================
%% Step 1a: Implement PCA to obtain xRot
%  Implement PCA to obtain xRot, the matrix in which the data is expressed
%  with respect to the eigenbasis of sigma, which is the matrix U.


% -------------------- YOUR CODE HERE -------------------- 
xRot = zeros(size(x)); % You need to compute this  
sigma = x * x' / size(x, 2); %����Э��������ֵ 
[U,S,V]=svd(sigma);  
xRot=U'*x;  %������ת�������


%%================================================================
%% Step 2: Find k, the number of components to retain
%  Write code to determine k, the number of components to retain in order
%  to retain at least 99% of the variance.������99��ά�ȣ�

% -------------------- YOUR CODE HERE -------------------- 
k = 0; % Set k accordingly  
diag_sum=trace(S);  
sum=0;  
rate=0.99;  
while((sum/diag_sum)<rate)  
    k=k+1;  
    sum=sum+S(k,k);  
end 
if(vk~=0) 
k=vk;
end

%%================================================================
%% Step 3: Implement PCA with dimension reduction
%  Now that you have found k, you can reduce the dimension of the data by
%  discarding the remaining dimensions. In this way, you can represent the
%  data in k dimensions instead of the original 144, which will save you
%  computational time when running learning algorithms on the reduced
%  representation.
% 
%  Following the dimension reduction, invert the PCA transformation to produce 
%  the matrix xHat, the dimension-reduced data with respect to the original basis.
%  Visualise the data and compare it to the raw data. You will observe that
%  there is little loss due to throwing away the principal components that
%  correspond to dimensions with low variation.

% -------------------- YOUR CODE HERE -------------------- 
xHat = zeros(size(x));  % You need to compute this  
xTilde = U(:,1:k)' * x;  %���ݽ�ά֮��Ľ��
xHat(1:k,:)=xTilde;  
xHat=U*xHat; %��ԭ����ά����ʣ��άΪȫ0��


%%================================================================
% %% Step 4a: Implement PCA with whitening and regularisation
% %  Implement PCA with whitening and regularisation to produce the matrix
% %  xPCAWhite. 
% 
% epsilon = 0.1;
% xPCAWhite = zeros(size(x));
% 
% % -------------------- YOUR CODE HERE -------------------- 
% xPCAWhite = diag(1./sqrt(diag(S) + epsilon)) * U' * x; %�����ݽ��а׻�
% data = xPCAWhite';
data=xHat';
labels=y;
name=[name,'_pca.mat'];
save(name,'data','labels');
m=1;
end
%%================================================================


