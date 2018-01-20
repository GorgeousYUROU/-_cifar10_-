function NN()
load data_batch_1.mat;
train_x=data;
train_y=labels;
load test_batch.mat;
test_x=data;
test_y=labels;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

rand('state',0)
nn = nnsetup([3072 1000 500 100 10]);
opts.numepochs         = 5;            %  Number of full sweeps through data
opts.batchsize         = 1000;         %  Take a mean gradient step over this many samples
opts.plot              = 1;            %  enable plotting
nn.weightPenaltyL2 = 1e-4;  %  L2 weight decay
nn = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
sprintf('使用有监督的NN网络的正确率：%2.2f%%',(1.0-er)*100)
end
