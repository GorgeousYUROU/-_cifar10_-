% function patches = sampleIMAGES(data,patchsize,numpatches)
function patches = sampleIMAGES(patchsize,numpatches)
% sampleIMAGES
% Returns 36000 patches for training
 

patches = zeros(patchsize*patchsize*3, numpatches);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Fill in the variable called "patches" using data
%  from IMAGES. 
% 
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
%  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
%  it. (The contrast on these images look a bit off because they have
%  been preprocessed using using "whitening."  See the lecture notes for
%  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
%  patch corresponding to the pixels in the block (21,21) to (30,30) of
%  Image 1
tic

% load('data_batch_1_pca.mat');
load('data_batch_1.mat');
data=double(data)/255;
data=data';
data=reshape(data,32,32,3,10000);
numpatchestemp=10000;
image_size=size(data(:,:,:,1));
i=randi(image_size(1)-patchsize+1,1,numpatchestemp);%生成元素值随机为大于0且小于image_size(1)-patchsize+1的1行numpatches矩阵
j=randi(image_size(2)-patchsize+1,1,numpatchestemp);
k=randi(image_size(3),1,numpatches);
for num=1:10000
        patches(1:patchsize*patchsize,num)=reshape(data(i(num):i(num)+patchsize-1,j(num):j(num)+patchsize-1,1,k(num)),1,patchsize*patchsize);
        patches(patchsize*patchsize+1:patchsize*patchsize*2,num)=reshape(data(i(num):i(num)+patchsize-1,j(num):j(num)+patchsize-1,2,k(num)),1,patchsize*patchsize);
        patches(patchsize*patchsize*2+1:patchsize*patchsize*3,num)=reshape(data(i(num):i(num)+patchsize-1,j(num):j(num)+patchsize-1,3,k(num)),1,patchsize*patchsize);
end
clear data;


% load('data_batch_2_pca.mat');
load('data_batch_2.mat');
data=double(data)/255;
data=data';
data=reshape(data,32,32,3,10000);
for num=1:10000
        patches(1:patchsize*patchsize,num+10000)=reshape(data(i(num):i(num)+patchsize-1,j(num):j(num)+patchsize-1,1,k(num)),1,patchsize*patchsize);
        patches(patchsize*patchsize+1:patchsize*patchsize*2,num+10000)=reshape(data(i(num):i(num)+patchsize-1,j(num):j(num)+patchsize-1,2,k(num)),1,patchsize*patchsize);
        patches(patchsize*patchsize*2+1:patchsize*patchsize*3,num+10000)=reshape(data(i(num):i(num)+patchsize-1,j(num):j(num)+patchsize-1,3,k(num)),1,patchsize*patchsize);
end
clear data;


% load('data_batch_3_pca.mat');
load('data_batch_3.mat');
data=double(data)/255;
data=data';
data=reshape(data,32,32,3,10000);
for num=1:10000
        patches(1:patchsize*patchsize,num+20000)=reshape(data(i(num):i(num)+patchsize-1,j(num):j(num)+patchsize-1,1,k(num)),1,patchsize*patchsize);
        patches(patchsize*patchsize+1:patchsize*patchsize*2,num+20000)=reshape(data(i(num):i(num)+patchsize-1,j(num):j(num)+patchsize-1,2,k(num)),1,patchsize*patchsize);
        patches(patchsize*patchsize*2+1:patchsize*patchsize*3,num+20000)=reshape(data(i(num):i(num)+patchsize-1,j(num):j(num)+patchsize-1,3,k(num)),1,patchsize*patchsize);
end
clear data;


% load('data_batch_4_pca.mat');
load('data_batch_4.mat');
data=double(data)/255;
data=data';
data=reshape(data,32,32,3,10000);
for num=1:10000
        patches(1:patchsize*patchsize,num+30000)=reshape(data(i(num):i(num)+patchsize-1,j(num):j(num)+patchsize-1,1,k(num)),1,patchsize*patchsize);
        patches(patchsize*patchsize+1:patchsize*patchsize*2,num+30000)=reshape(data(i(num):i(num)+patchsize-1,j(num):j(num)+patchsize-1,2,k(num)),1,patchsize*patchsize);
        patches(patchsize*patchsize*2+1:patchsize*patchsize*3,num+30000)=reshape(data(i(num):i(num)+patchsize-1,j(num):j(num)+patchsize-1,3,k(num)),1,patchsize*patchsize);
end
clear data;


% load('data_batch_5_pca.mat');
load('data_batch_5.mat');
data=double(data)/255;
data=data';
data=reshape(data,32,32,3,10000);
for num=1:10000
        patches(1:patchsize*patchsize,num+40000)=reshape(data(i(num):i(num)+patchsize-1,j(num):j(num)+patchsize-1,1,k(num)),1,patchsize*patchsize);
        patches(patchsize*patchsize+1:patchsize*patchsize*2,num+40000)=reshape(data(i(num):i(num)+patchsize-1,j(num):j(num)+patchsize-1,2,k(num)),1,patchsize*patchsize);
        patches(patchsize*patchsize*2+1:patchsize*patchsize*3,num+40000)=reshape(data(i(num):i(num)+patchsize-1,j(num):j(num)+patchsize-1,3,k(num)),1,patchsize*patchsize);
end
clear data;


% load('test_batch_pca.mat');
load('test_batch.mat');
data=double(data)/255;
data=data';
data=reshape(data,32,32,3,10000);
for num=1:10000
        patches(1:patchsize*patchsize,num+50000)=reshape(data(i(num):i(num)+patchsize-1,j(num):j(num)+patchsize-1,1,k(num)),1,patchsize*patchsize);
        patches(patchsize*patchsize+1:patchsize*patchsize*2,num+50000)=reshape(data(i(num):i(num)+patchsize-1,j(num):j(num)+patchsize-1,2,k(num)),1,patchsize*patchsize);
        patches(patchsize*patchsize*2+1:patchsize*patchsize*3,num+50000)=reshape(data(i(num):i(num)+patchsize-1,j(num):j(num)+patchsize-1,3,k(num)),1,patchsize*patchsize);
end
clear data;

toc
 
%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure
% the range of pixel values is also bounded between [0,1]
patches = normalizeData(patches);
 
end
 
 
%% ---------------------------------------------------------------
function patches = normalizeData(patches)
 
% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer
 
% Remove DC (mean of images). 把patches数组中的每个元素值都减去mean(patches)
patches = bsxfun(@minus, patches, mean(patches));
 
% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));%把patches的标准差变为其原来的3倍
patches = max(min(patches, pstd), -pstd) / pstd;%因为根据3sigma法则，95%以上的数据都在该区域内
                                                % 这里转换后将数据变到了-1到1之间
% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;
 
end