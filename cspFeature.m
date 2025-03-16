%%%%%%ͨ��ͶӰ�������������ȡ��ͨ�����ռ�ģʽ��CSP��ͶӰ������������ͶӰ���µĿռ䣬����ȡͶӰ�����ݵķ���������
%%%%%%Ȼ�����Щ�������й�һ����ȡ�������Եõ����յ��������� feature����Щ���������������ڷ�������ѵ���Ͳ��ԡ�
%���������
%  projM: cspͶӰ������ά����ά��Ϊ (ͨ����, ͨ����, �����)
%  x: һ��ʱ�䴰�ڵ�2άEEG���ݡ����У���һά�ǲ����㣻�ڶ�ά��ͨ��
%  m: ͶӰ���ݾ���ĵ�һ�к����һ�еĸ�����
%���������
%  feature: ������������ȡ��������

function feature = cspFeature(projM, data, m)
    % ������֤
    assert(isequal(size(projM,1), size(projM,2)), 'ͶӰ����ӦΪ����');
    assert(size(projM,1) == size(data,2), '����ͨ������ƥ��');
    
    % ���ļ����߼�
    classNo = size(projM, 3);  % ͶӰ����������
    feature = [];
    
    for k = 1:classNo
        % ��ȷ�˷�˳��: [ʱ����ͨ��] �� [ͨ����ͨ��] �� [ʱ����ͨ��]
        Z = data * projM(:,:,k); 
        
        % ��ȡǰm�ͺ�m��ͨ���ķ���
        var_front = var(Z(:, 1:m));        % [1��m]
        var_back = var(Z(:, end-m+1:end)); % [1��m]
        
        % ������һ��
        log_features = log( [var_front, var_back] / sum([var_front, var_back]) );
        feature = [feature, log_features];
    end
end

% 
% function feature=cspFeature(projM,data,m)
%     
%     % ���ά����֤
%     disp('====== cspFeatureά����֤ ======');
%     disp(['ͶӰ����ά��: ', num2str(size(projM))]); % ӦΪ [59 59 ...]
%     disp(['��������ά��: ', num2str(size(data))]); % ӦΪ [59 N]
%     assert(size(projM,3) == 4, 'ÿ����Ƶ��Ӧʹ��4��ͶӰ����');
% 
%     classNo=length(projM(1,1,:));  %��ȡ�������
%     channelNo=size(data,2);           %��ȡͨ������
%     feature=[];                    %������������
%     for k=1:classNo                %classNoΪ������
%         Z=data*projM(:,:,k); %projected data matrix
%         for j=1:m
%             feature=[feature; var(Z(:,j)); var(Z(:,channelNo-j+1))];  %var(A) �����Aû�з����ʱĬ���ǳ�N-1
%             %variances of the first and last m columns(��1�����m�еķ���)
%         end
%     end
%     feature=log(feature/sum(feature));%����ȡ���� feature ����������й�һ��������ÿ������ֵ��������������ܺͣ�Ȼ��ȡ������
% end                                  %��������Ŀ����Ϊ�˽�����ѹ����һ����С����ֵ��Χ�ڣ�ʹ������ֵ֮��Ĳ������������