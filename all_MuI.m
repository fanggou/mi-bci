%author:chen 
%input:fea_train(m*n), ��������ÿ�д���һ��������ÿ�д���һ������
%      label_train(m*1),��ǩ������ÿ��Ԫ�ض�Ӧһ�������ı�ǩ
%output:rank(n*2),the first dimension includes the Mutual information 
%       values,and the second dimension incoude the index
%       �������� sort_tmp������һ�������ľ���
%all_Mul�ú�������ÿ���������ǩ֮��Ļ���Ϣ��������һ������rank�����а�����������Ҫ������
%rank(1:k,2) ��ʾȡ rank ������ǰ k �е� 2 �е�ֵ����Щֵ��ʾ����Ϣ��ߵ�����������
%FBtrainf(:,rank(1:k,2)) ʹ����Щ������ FBtrainf ����ȡ��Ӧ�������������µ�ѵ���� selFeaTrain
function sort_tmp=all_MuI(fea_train,label_train)
n=size(fea_train,1); %��ȡ fea_train ��������������������������洢�ڱ��� n ��         
tmp=[];%��ʼ��һ���վ��� tmp�����ڴ洢ÿ�������Ļ���Ϣֵ�����Ӧ������
for i=1:size(fea_train,2)
    MuI=calc_MuI(fea_train(:,i),label_train,n);
    tmp=[tmp;MuI i];  %������Ϣֵ���������� i ��Ϊһ��׷�ӵ����� tmp ��                
end
sort_tmp=sortrows(tmp,'descend');%�Ծ��� tmp ���յ�һ�У�����Ϣֵ�����н�������;sort_tmp���洢�����ľ���