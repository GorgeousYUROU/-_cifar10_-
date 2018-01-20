function datasum()
load('data/data_batch_1.mat');
data1 = double(data(1:5000,:));
labels1 = double(labels(1:5000));
load('data/data_batch_2.mat');
data2 = double(data(1:5000,:));
labels2 = double(labels(1:5000));
load('data/data_batch_3.mat');
data3 = double(data(1:5000,:));
labels3 = double(labels(1:5000));
load('data/data_batch_4.mat');
data4 = double(data(1:5000,:));
labels4 = double(labels(1:5000));
data = [data1; data2; data3; data4];
labels = [labels1; labels2; labels3; labels4];
save('CNN/datasum.mat','data','labels');
end