x=input('��ѡ��ѵ��cifar10����Ҫ������0��ʾCNN��15000��������8g�ڴ�ѡ��,1��ʾ����ѧϰ,2��ʾNN��3��ʾCNN��50000������:');
if x<1 
    fprintf('���ڿ�ʼ����CNN��15000������\n');
    fprintf('���ڿ�ʼ�������ݺϲ�\n');
    datasum();
    fprintf('���ڿ�ʼʹ�������Ա��뿪ʼ��ȡ����\n');
    linearDecoderExercise();
    fprintf('���ڿ�ʼ�õ���������ѵ��\n');
    cnnExercise();
    fprintf('��������CNN\n');
elseif x<2
    fprintf('���ڿ�ʼ��������ѧϰ\n');
    fprintf('���ڿ�ʼ�����ݽ���pca\n');
    pcaAll();
    fprintf('���ڿ�ʼ�����ݽ��кϲ��������ݷֳ��мල���޼ල\n');
    dataClassify();
    fprintf('���ڿ�ʼ������ѧϰ����ѵ��\n');
    stlExercise();
    fprintf('������������ѧϰ\n');
elseif x<3
    fprintf('���ڿ�ʼ����NN\n');
    NN();
    fprintf('��������NN\n');
elseif x<4
    fprintf('���ڿ�ʼ����CNN(50000����)\n');
    fprintf('���ڿ�ʼ�������ݺϲ�\n');
    datasum_50000();
    fprintf('���ڿ�ʼʹ�������Ա��뿪ʼ��ȡ����\n');
    linearDecoderExercise();
    fprintf('���ڿ�ʼ�õ���������ѵ��\n');
    cnnExercise_50000();
    fprintf('��������CNN\n');
end