
%% �����ݽ����˲�
%���룺data ���˲�EEG����
%   low        ��ͨ�˲���������
%   high       ��ͨ�˲���������
%   sampleRate          ������
%   filterorder  butterworth�˲�������
%���أ�filterdata       �˲���EEG����
function filterdata=filter_param(data,low,high,sampleRate,filterorder)
%% �����˲�����
     filtercutoff = [low*2/sampleRate high*2/sampleRate]; 
     [filterParamB, filterParamA] = butter(filterorder,filtercutoff);
     filterdata= filter( filterParamB, filterParamA, data);
end

