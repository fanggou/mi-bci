%author:chen date:2024-7-31 
%programme:
%input:train_data, 3ά EEG���ݡ����У���һά�ǲ����㣬�ڶ�ά��ͨ������������ά����trials��С
%      train_label,train_data��Ӧ�ı�ǩ
%      sampleRate,������
%      m,CSP��-m����
%      freq,Ƶ���������飬�����˲�
%output:features_train �ںϺ����Ƶ�������������,features_train=(train_data,projMAll,classNum,sampleRate,m,freq)
%       projMAll �ɸ���Ƶ���������õ�ͶӰ����
%       classNum ��������������
function [features_train,projM_All,classNum]=FBCSP(train_data,train_label,sampleRate,m,freq)
[q,p,k]=size(train_data);   %��ȡ�ܵ�trial�������������k,ͨ����p,��������q
%% ��ȡ����ϲ�ͬ���˲���Ƶ��
features_train=[];          %����ѵ����csp�����ں����飬�����洢����Ƶ����CSP����
filter_data=zeros(size(train_data));%��ʼ���˲�������ݣ�filter_data���ڴ洢�˲��������
classNum=max(train_label);  %��ȡ�������
projM_All=zeros(p,p,max(train_label)*(size(freq,2)-1)); %����ͶӰ����ռ䣬���ڴ洢����Ƶ����ͶӰ����
for i=1:(length(freq)-1)
    lower=freq(i);  %��ȡ��Ƶ
    if lower==freq(size(freq,2))
        break;
    end
    higher=freq(i+1);%��ȡ��Ƶ
    %�Ը���Ƶ�������˲�
    filter_tmp=[];
    for j=1:k   %��ÿ��trial����ѭ���˲�,filter()���������˲�3ά���ݣ�
        filter_tmp=filter_param(train_data(:,:,j),lower,higher,sampleRate,4);
        filter_data(:,:,j)=filter_tmp;
    end
    % ����csp�˲�������csp�˲�������������ȡ
    projM=cspProjMatrix(filter_data,train_label); %Ҫѭ������ͶӰ������������CSP�˲�
    projM_All(:,:,1+(i-1)*classNum:i*classNum)=projM;   %�洢��ǰƵ��ͶӰ����
    feature=[];  %��������Ƶ����������
    
    for b=1:k    %ѭ����ȡ����
        feature(b,:)=cspFeature(projM,filter_data(:,:,b),m); %����������m��Ҫ����ͨ������һ�룬��Ȼ������ظ�����
    end
    tmp_data=feature;
    features_train=[features_train,tmp_data]; %ƴ�Ӹ�Ƶ����������
end