dataset = {'a8a' 'cod_rna' 'covtype' 'gisette' 'mushrooms' 'svmguide1'}; %'a8a' 
for t = 1:length(dataset)
    display(dataset{t});
    DiffRatio(strcat(dataset{t},'.mat'));
end
 