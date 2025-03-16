%author:chen date:2024-7-31
%ͨ���Բ�ͬƵ�������˲���Ȼ��ʹ��CSP�㷨��ȡÿ��Ƶ�������������ս�����Ƶ��������ƴ����һ��
%�γ����յ��������� features_train������������������ں����ķ������񡣸ú����ܹ������ά����ά��EEG���ݣ�����������������ȡ��

%input:train_data,3ά EEG���ݡ����У���һά�ǲ����㣬�ڶ�ά��ͨ������������ά����trials��С
%                 2ά EEG���ݡ����У���һλ�ǲ����㣬�ڶ�ά��ͨ������
%      projMAll,  ��ѵ�����������ø���Ƶ��CSPͶӰ����
%      classNum,  ������������
%      sampleRate,������
%               m,CSP��m����
%output:features, �ںϺ����Ƶ�������������
function features_train=FBCSPOnline(train_data,projMAll,classNum,sampleRate,m,freq)
    
    if nargin < 6
        error('�������㣬��Ҫ6���������');
    end

    if ndims(train_data)==3 %����EEG����Ϊ3ά
      
        % acquire and combine feature of different frequency bands
        [q,p,k]=size(train_data);%��ȡ�ܵ�trial����
        filter_data=zeros(size(train_data));
        features_train=[];      %����ѵ����csp�����ں�����
        numBands = length(freq)-1; % ��Ƶ������ΪƵ�ʵ�����-1
       
        for i=1:numBands
    
            lower=freq(i);  %��ȡ��Ƶ
            higher=freq(i+1);%��ȡ��Ƶ
    
            %�Ը���Ƶ�������˲�
            filter_tmp=[];
            for j=1:k   %��ÿ��trial����ѭ���˲�,matlab�е�filter()���������˲�3ά���ݣ�
                filter_tmp=filter_param(train_data(:,:,j),lower,higher,sampleRate,4);
                filter_data(:,:,j)=filter_tmp;
            end
    
            feature=[];  %��������Ƶ����������
            for b=1:k    %ѭ����ȡ����
                feature(b,:)=cspFeature(projMAll(:,:,1+(i-1)*classNum:i*classNum),filter_data(:,:,b),m); %����������m��Ҫ����ͨ������һ�룬��Ȼ������ظ�����
            end
    
            tmp_data=feature;
            features_train=[features_train,tmp_data]; %ƴ�Ӹ���Ƶ����������
    
        end
    else                  
       
        % ����EEG����Ϊ2ά
        features_train=[];      %����ѵ����csp�����ں�����
        numBands = length(freq)-1; % ��Ƶ������ΪƵ�ʵ�����-1
    
        for i=1:numBands
           
            lower=freq(i);  %��ȡ��Ƶ
            higher=freq(i+1);%��ȡ��Ƶ
           
            % �˲���ת��Ϊ [ͨ�� �� ʱ���]
            filtered = filter_param(train_data, lower, higher, sampleRate, 4);
            
            % ��ȡ��ǰ��Ƶ��ͶӰ���� (4������)
            currentProj = projMAll(:,:, (i-1)*classNum+1 : i*classNum);
            
            % ������ȡ
            feature = cspFeature(currentProj, filtered, m);
            features_train = [features_train, feature]; 
            
            %������Ϣ
            disp(['��Ƶ�� ', num2str(i), ' ͶӰ����ߴ�: ', num2str(size(currentProj))]);
            disp(['�˲������ݳߴ�: ', num2str(size(filtered))]);
           

        end
    end
end

