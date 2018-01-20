function datasum_50000()
load('data/data_batch_1.mat');
data1 = double(data);
labels1 = double(labels);
load('data/data_batch_2.mat');
data2 = double(data);
labels2 = double(labels);
load('data/data_batch_3.mat');
data3 = double(data);
labels3 = double(labels);
load('data/data_batch_4.mat');
data4 = double(data);
labels4 = double(labels);
load('data/data_batch_5.mat');
data5 = double(data);
labels5 = double(labels);
data = [data1; data2; data3; data4; data5];
labels = [labels1; labels2; labels3; labels4; labels5];
save('CNN/datasum_50000.mat','data','labels');
end