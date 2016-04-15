function DiffRatioTest()
%'covtype' 'covtype' 'spambase' 'svmguide1' 'magic04' 'a8a'  'mushrooms' 'w8a'

data='gisette';
options.numNoiseExperts=0;
load(sprintf('../../data/ori/%s.mat',data));
display(data);
%% Train Experts
%1. Perceptron
[n,d]=size(data_train);
d=d-1;
[w_1] = perceptron(data_train);
%2. ROMMA
[w_2] = ROMMA(data_train);
%3. ALMA
[w_3] = ALMA(data_train);
%4. PA-I
[w_4] = PA1(data_train);
%5. CW
[w_5] = AROW(data_train);
W=[w_1,w_2,w_3,w_4,w_5];
if options.numNoiseExperts > 0,
    display('Have Noise Experts');
    for i=1:options.numNoiseExperts,
        W=[W,zeros(d,1)];
    end
end

%% options
N=size(W,2);
n=size(data_test,1);
ID_list=ID_test;
options.eta    = (8*log(N)/n)^0.5;
options.t_tick = round(n/15);  

%%  
% for m=1:30,
%     del_AE2(m)=10^(-120+4*m);
%     del_AGF2(m)=10^(-60+2*m);
% end    
% options.multiplierAE2=10^20;
% options.multiplierAGF2=100;
% 
% del_AE=[0.2:0.05:0.85,0.875,0.938,0.969,0.984,0.992,0.996,0.998,1:0.002:2];  
% % del_AE=[0.998:0.002:2];
% del_AGF=[0.45:0.015:2];

%% del
if strcmp(data,'a8a')==1,
    for m=1:30,
        del_AE2(m)=10^(-60+2*m);
        del_AGF2(m)=10^(-30+m);
    end    
    options.multiplierAE2=10^20;
    options.multiplierAGF2=100;
    del_AE=[0.001,0.002,0.004,0.008,0.016,0.031,0.063,0.050,0.1,0.15,0.2,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.75,0.80,0.85,0.875,0.938,0.969,0.984,0.992,0.996,0.998];  
    del_AGF=[0.001,0.002,0.004,0.008,0.016,0.031,0.063,0.05:0.05:2];
elseif strcmp(data,'svmguide1Scale')==1,
    for m=1:30,
        del_AE2(m)=10^(-30+m);
        del_AGF2(m)=10^(-8+m*4/15);
    end    
    options.multiplierAE2=100;
    options.multiplierAGF2=10;
    del_AE=[0.2:0.03:2];%0.85,0.875,0.938,0.969,0.984,0.992,0.996,0.998,1:0.002:2];  
    del_AGF=[0.05:0.03:2];
elseif strcmp(data,'spambase')==1,
    for m=1:30,
        del_AE2(m)=10^(-30+m);
        del_AGF2(m)=10^(-20+m*2/3);
    end    
    options.multiplierAE2=10^6;
    options.multiplierAGF2=10;
%     del_AE=[0.05:0.05:2];
    del_AE=[0.2:0.05:0.85,0.875,0.938,0.969,0.984,0.992,0.996,0.998,1:0.002:2];  
    del_AGF=[0.05:0.02:2];
elseif strcmp(data,'mushrooms')==1,
     for m=1:30,
        del_AE2(m)=10^(-10+m*1/3);
        del_AGF2(m)=10^(-10+m*1/3);%10^(20/30+1/90*m);
    end    
    options.multiplierAE2=10^4;
    options.multiplierAGF2=10^4;%10        
    del_AE=[0.001,0.002,0.004,0.008,0.016,0.031,0.063,0.050,0.1,0.15,0.2,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.75,0.80,0.85,0.875,0.938,0.969,0.984,0.992,0.996,0.998];  
    del_AGF=[0.001,0.002,0.004,0.008,0.016,0.031,0.063,0.05:0.05:0.7,0.705:0.001:2];
elseif strcmp(data,'magic04Scale')==1,
    display('magic04 del');
    for m=1:30,
        del_AE2(m)=10^(-120+4*m);
        del_AGF2(m)=10^(-30+m);
    end    
    options.multiplierAE2=10^10;
    options.multiplierAGF2=100;        
    del_AE=[0.05:0.05:2];
    del_AGF=[0.05:0.05:2];
    %del_AE=[0.92:0.016:0.984,0.9848:0.0008:2];  
    %del_AGF=[0.716:0.0005:0.721,0.726:0.005:2];
elseif strcmp(data,'codrna')==1,
    for m=1:30,
        del_AE2(m)=10^(-120+4*m);
        del_AGF2(m)=10^(-120+4*m);
    end    
    options.multiplierAE2=10^30;
    options.multiplierAGF2=100;
    del_AE=[0.2:0.05:0.85,0.875,0.938,0.969,0.984,0.992,0.996,0.998:0.0002:2];
    del_AGF=[0.2:0.05:0.7,0.72:0.02:2];

elseif strcmp(data,'covtype')==1,
    for m=1:30,
        del_AE2(m)=10^(-270+9*m);
        del_AGF2(m)=10^(-150+5*m);
    end    
    options.multiplierAE2=10^50;
    options.multiplierAGF2=10^50;
    del_AE=[0.2:0.05:0.85,0.875,0.938,0.969,0.984,0.992,0.996,0.998,1:0.002:2];  
    del_AGF=[0.45:0.015:0.69,0.695:0.005:2];
elseif strcmp(data,'w8a')==1,
    for m=1:30,
        del_AE2(m)=10^(-60+2*m);
        del_AGF2(m)=10^(-10+1/3*m);
    end    
    options.multiplierAE2=100;
    options.multiplierAGF2=10;

    del_AE=[0.001,0.002,0.004,0.008,0.016,0.031,0.063,0.05:0.05:0.85,0.875,0.938,0.969,0.984,0.992,0.996,0.998];  
    del_AGF=[0.45:0.02:0.69,0.692:0.002:0.708,0.7082:0.0002:2];%,0.002,0.004,0.008,0.016,0.031,0.063,0.05:0.05:2];
elseif strcmp(data,'gisette')==1,
    display(data);
    for m=1:30,
        del_AE2(m)=10^(-60+2*m);
        del_AGF2(m)=10^(-60+2*m);
    end    
    options.multiplierAE2=100;
    options.multiplierAGF2=10;
    del_AE=[0.001,0.002,0.004,0.008,0.016,0.031,0.063,0.050,0.1,0.15,0.2,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.75,0.80,0.85,0.875,0.938,0.969,0.984,0.992,0.996,0.998];  
    del_AGF=[0.001,0.002,0.004,0.008,0.016,0.031,0.063,0.050,0.1,0.15,0.2,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.75,0.80,0.85,0.875,0.938,0.969,0.984,0.992,0.996,0.998];  
elseif strcmp(data,'ijcnn1')==1,
    for m=1:30,
        del_AE2(m)=10^(-120+4*m);
        del_AGF2(m)=10^(-120+4*m);
    end    
    options.multiplierAE2=100;
    options.multiplierAGF2=100;
    
    del_AE=[0.05:0.05:2];
    del_AGF=[0.05:0.05:2];
end


%% for each del    
ID = ID_list(1,:); 
que_list=[1:1:30];

for i=1:length(que_list),%length(que),%20, %size(que,2),
    j=que_list(i);
    fprintf(1,'running on the %d-th trial...\n',j);           
%% 3. AEWAF           
    options.delta=del_AE(j);
    [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=AEWAF(W, data_test, options, ID);            
    fprintf(1,'AE:%.3f,del:%.5f\t',NumReq/n,options.delta);

%% 4. AEWAF2            
    options.delta=del_AE2(j);         
    [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=AEWAF2(W, data_test, options, ID);            
    fprintf(1,'AE2:%.3f,reg:%.5f\n',NumReq/n,Regret/n);

end

fprintf(1,'\n');

for i=1:length(que_list),
    j=que_list(i);
    fprintf(1,'running on the %d-th trial...\n',j); 
%% 5. AGF
    options.delta=del_AGF(j);      
    [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=AGF(W, data_test, options, ID);             
    fprintf(1,'AF:%.3f,del:%.5f\t',NumReq/n,options.delta);
    
%% 5. AGF2:
    options.delta=del_AGF2(j);
    [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=AGF2(W, data_test, options, ID);                
    fprintf(1,'AF2:%.3f,reg:%.5f\n',NumReq/n,Regret/n);
end