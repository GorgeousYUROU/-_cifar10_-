x=input('请选择训练cifar10所需要的网络0表示CNN（15000样本）（8g内存选择）,1表示自我学习,2表示NN，3表示CNN（50000样本）:');
if x<1 
    fprintf('现在开始运行CNN（15000样本）\n');
    fprintf('现在开始进行数据合并\n');
    datasum();
    fprintf('现在开始使用线性自编码开始提取特征\n');
    linearDecoderExercise();
    fprintf('现在开始用单层卷积网络训练\n');
    cnnExercise();
    fprintf('结束运行CNN\n');
elseif x<2
    fprintf('现在开始运行自我学习\n');
    fprintf('现在开始对数据进行pca\n');
    pcaAll();
    fprintf('现在开始对数据进行合并，把数据分成有监督和无监督\n');
    dataClassify();
    fprintf('现在开始用自我学习网络训练\n');
    stlExercise();
    fprintf('结束运行自我学习\n');
elseif x<3
    fprintf('现在开始运行NN\n');
    NN();
    fprintf('结束运行NN\n');
elseif x<4
    fprintf('现在开始运行CNN(50000样本)\n');
    fprintf('现在开始进行数据合并\n');
    datasum_50000();
    fprintf('现在开始使用线性自编码开始提取特征\n');
    linearDecoderExercise();
    fprintf('现在开始用单层卷积网络训练\n');
    cnnExercise_50000();
    fprintf('结束运行CNN\n');
end