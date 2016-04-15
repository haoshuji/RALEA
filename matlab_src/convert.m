function []=convert(dataset_name)
% data_train, data_test, ID_train, ID_test
clear 

load(sprintf('%s',dataset_name));

X=X';
data=[Y,X];

%% size of data
n = size(data,1);

%% rand permutate data
perm=randperm(n);
data=data(perm,:);

%% size_train ID_train
size_train=round(0.2*n);
ID_train=1:size_train;
ID_train=ID_train';
data_train=data(ID_train,:);

ID_temp=size_train+1:n;
ID_temp=ID_temp';
data_test=data(ID_temp,:);

%% size_test ID_test
size_test= n-size_train;
ID_test=[];
for i=1:20,
    ID_test = [ID_test; randperm(size_test)];
end

save(dataset_name);










