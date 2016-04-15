function DiffRatioOri()
% Experiment_OL_K_M: the main procedure evaluating all the algorithm on the same dataset
%--------------------------------------------------------------------------
% Input:
%      dataset_name: the name of the dataset file
% Output:
%      a figure of online mistake rates for all the algorithms
%      a figure of online SV size for all the algorithms
%      a figure of online time consumption for all the algorithms
%      a table of the final mistake rates, SV size, time consumption for all the algorithms
%--------------------------------------------------------------------------

%%dataset
dataset ={ 'codrna'};% 'covtype' 'a8a' 'ijcnn1' 'magic04' 'gisette'   gisetteScale' 'magic04Scale' 'mushroomsScale' 'spambaseScale' 'codrnaScale' 'svmguide1Scale'
%       'w8a'};%  'magic04' 'a8a'  'gisette'  'mushrooms' 'spambase' 'svmguide1'   'covtype'};
%'mushrooms' 'svmguide1' 'w8a' 'ijcnn1' 'codrna' 'gisette'
%%for each data 'codrna'
options.numNoiseExperts=0;
loc='../noR/final/';

for t = 1:length(dataset)
    data=dataset{t};
    fileID=fopen(strcat('../noR/final/',data,'log.txt'),'w');    
        
    display(dataset{t});
    load(sprintf('../../data/ori/%s.mat',data));

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
    
    reg_EW=[]; num_EW=[]; time_EW=[]; REG_EW=[];NUMs_EW=[];TMs_EW=[];
    reg_AE=[]; num_AE=[]; time_AE=[]; REG_AE=[];NUMs_AE=[];TMs_AE=[];
    reg_AE2=[]; num_AE2=[]; time_AE2=[]; REG_AE2=[];NUMs_AE2=[];TMs_AE2=[];
    reg_RE=[]; num_RE=[]; time_RE=[]; REG_RE=[];NUMs_RE=[];TMs_RE=[];
    reg_RE2=[]; num_RE2=[]; time_RE2=[]; REG_RE2=[];NUMs_RE2=[];TMs_RE2=[];
    
    reg_GF=[]; num_GF=[]; time_GF=[]; REG_GF=[];NUMs_GF=[];TMs_GF=[];
    reg_AF=[]; num_AF=[]; time_AF=[]; REG_AF=[];NUMs_AF=[];TMs_AF=[];
    reg_AF2=[]; num_AF2=[]; time_AF2=[]; REG_AF2=[];NUMs_AF2=[];TMs_AF2=[];
    reg_RF=[]; num_RF=[]; time_RF=[]; REG_RF=[];NUMs_RF=[];TMs_RF=[];
    reg_RF2=[]; num_RF2=[]; time_RF2=[]; REG_RF2=[];NUMs_RF2=[];TMs_RF2=[];
    
    %% options
    N=size(W,2);
    n=size(data_test,1);
    ID_list=ID_test;
    options.eta    = (8*log(N)/n)^0.5;

    options.t_tick = round(n/15);

   
    %% delta
    if strcmp(data,'a8a')==1,
        for m=1:30,
            del_AE2(m)=10^(-60+2*m);
            del_AGF2(m)=10^(-30+m);
        end    
        options.multiplierAE2=10^20;
        options.multiplierAGF2=100;
        del_AE=[0.001,0.002,0.004,0.008,0.016,0.031,0.063,0.050,0.1,0.15,0.2,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.75,0.80,0.85,0.875,0.938,0.969,0.984,0.992,0.996,0.998];  
        del_AGF=[0.001,0.002,0.004,0.008,0.016,0.031,0.063,0.05:0.05:2];
    elseif strcmp(data,'svmguide1Scale')==1 || strcmp(data,'svmguide1')==1,        
        for m=1:30,
            del_AE2(m)=10^(-120+4*m);
            del_AGF2(m)=10^(-30+m);
        end    
        options.multiplierAE2=100;
        options.multiplierAGF2=100;
        del_AE=[0.2:0.03:2];%0.85,0.875,0.938,0.969,0.984,0.992,0.996,0.998,1:0.002:2];  
        del_AGF=[0.05:0.03:2];
    elseif strcmp(data,'spambaseScale')==1 || strcmp(data,'spambase')==1,
        for m=1:30,
            del_AE2(m)=10^(-30+m);
            del_AGF2(m)=10^(-20+m*2/3);
        end    
        options.multiplierAE2=10^6;
        options.multiplierAGF2=100;
        del_AE=[0.02:0.05:0.85,0.875,0.938,0.969,0.984,0.992,0.996,0.998,1:0.002:2];  
        del_AGF=[0.05:0.02:2];
    elseif strcmp(data,'magic04Scale')==1 || strcmp(data,'magic04')==1,
        display('magic04 del');
        for m=1:30,
            del_AE2(m)=10^(-120+4*m);
            del_AGF2(m)=10^(-30+m);
        end    
        options.multiplierAE2=10^10;
        options.multiplierAGF2=100;     
        del_AE=[0.05:0.05:2];
        del_AGF=[0.05:0.05:2];
    elseif strcmp(data,'mushroomsScale')==1 || strcmp(data,'mushrooms')==1,
        for m=1:30,
            del_AE2(m)=10^(-10+m*1/3);
            del_AGF2(m)=10^(-10+m*1/3);
        end    
        options.multiplierAE2=100;
        options.multiplierAGF2=100;        
        del_AE=[0.001,0.002,0.004,0.008,0.016,0.031,0.063,0.050:0.05:0.85,0.875,0.938,0.969,0.984,0.992,0.996,0.998];  
        del_AGF=[0.001,0.002,0.004,0.008,0.016,0.031,0.063,0.05:0.05:0.7,0.705:0.001:2];
    elseif strcmp(data,'codrnaScale')==1 || strcmp(data,'codrna')==1,
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
            del_AGF2(m)=10^(-180+6*m);
        end    
        options.multiplierAE2=10^50;
        options.multiplierAGF2=10^50;
        del_AE=[0.2:0.05:0.85,0.875,0.938,0.969,0.984,0.992,0.996,0.998,1:0.002:2];  
        del_AGF=[0.45:0.015:2];
    elseif strcmp(data,'w8a')==1,
        for m=1:30,
            del_AE2(m)=10^(-60+2*m);
            del_AGF2(m)=10^(-60+2*m);
        end    
        options.multiplierAE2=100;
        options.multiplierAGF2=100;
        del_AE=[0.001,0.002,0.004,0.008,0.016,0.031,0.063,0.05:0.05:0.85,0.875,0.938,0.969,0.984,0.992,0.996,0.998]; 
        del_AGF=[0.001,0.002,0.004,0.008,0.016,0.031,0.063,0.05:0.05:0.7,0.705:0.001:2];
%         del_AGF=[0.45:0.02:0.69,0.692:0.002:0.708,0.7082:0.0002:2];%,0.002,0.004,0.008,0.016,0.031,0.063,0.05:0.05:2];
    elseif strcmp(data,'gisette')==1,
        display(data);
        for m=1:30,
            del_AE2(m)=10^(-60+2*m);
            del_AGF2(m)=10^(-60+2*m);
        end    
        options.multiplierAE2=100;
        options.multiplierAGF2=10;
        del_AE=[0.001,0.002,0.004,0.008,0.016,0.031,0.063,0.05:0.05:0.85,0.875,0.938,0.969,0.984,0.992,0.996,0.998];  
        del_AGF=[0.001,0.002,0.004,0.008,0.016,0.031,0.063,0.05:0.05:0.7,0.705:0.001:2];
     elseif strcmp(data,'shuttle')==1,
        display(data);
        for m=1:30,
            del_AE2(m)=10^(-60+2*m);
            del_AGF2(m)=10^(-60+2*m);
        end    
        options.multiplierAE2=100;
        options.multiplierAGF2=10;
        del_AE=[0.001,0.002,0.004,0.008,0.016,0.031,0.063,0.05:0.05:0.85,0.875,0.938,0.969,0.984,0.992,0.996,0.998];  
        del_AGF=[0.001,0.002,0.004,0.008,0.016,0.031,0.063,0.05:0.05:0.7,0.705:0.001:2];
    elseif strcmp(data,'ijcnn1Scale')==1 || strcmp(data,'ijcnn1')==1,
        for m=1:30,
            del_AE2(m)=10^(-60+2*m);
            del_AGF2(m)=10^(-10+1/3*m);
        end    
        options.multiplierAE2=10^10;
        options.multiplierAGF2=10;
        del_AE=[0.05:0.05:2];
        del_AGF=[0.05:0.05:2];
    end
    
    %% each del
    que=[1:1:30];
    que_ran=[1/30:1/30:1];
    for j=1:length(que),%length(del_AE2),%length(que),%20, %size(que,2),
        fprintf(1,'running on the %d-th trial...\n',j);       
        m=que(j);
        last_max_del_AE2=del_AE2(m);        
        last_max_del_AE=del_AE(m);        
        last_max_del_AGF=del_AGF(m);
        last_max_del_AGF2=del_AGF2(m);
        
%         last_max_del_AE2=del_AE2(j);        
%         last_max_del_AE=del_AE(j);        
%         last_max_del_AGF=del_AGF(j);
%         last_max_del_AGF2=del_AGF2(j);
        %% Testing Active Greedy Forecaster
%         for i=1:1,
        for i=1:5,%size(ID_list,1),
            ID = ID_list(i,:);
            
%% 1. EWAF
            [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=EWAF(W, data_test, options, ID);
            reg_EW(i)    = Regret;
            num_EW(i)    = NumReq;
            time_EW(i)   = run_time;
            REG_EW(i,:)  = regrets;
            NUMs_EW(i,:) = NUMs;
            TMs_EW(i,:)  = TMs;

%% 2. Greedy Forecaster
            [ Regret, NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=GF(W, data_test, options, ID);
            reg_GF(i)    = Regret;
            num_GF(i)    = NumReq;
            time_GF(i)   = run_time;
            REG_GF(i,:)  = regrets;
            NUMs_GF(i,:) = NUMs;
            TMs_GF(i,:)  = TMs;           
             
%% 3. AEWAF     
            options.delta=last_max_del_AE;
            [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=AEWAF(W, data_test, options, ID);
            cur_query=NumReq/n;
            if i<1,
                tmp_del=[];
                tmp_que=[];
                if abs(cur_query-last_que2)/last_que2 > query_tol,
                    tmp_max_del=max_del1;
                    tmp_min_del=min_del1;
                    num_while=1;                
                    while abs(cur_query-last_que2)/last_que2 > query_tol && num_while<=num_search,
                        if cur_query > last_que2,
                            tmp_min_del=options.delta;
                        else
                            tmp_max_del=options.delta;
                        end
                        tmp_middle = tmp_max_del-(tmp_max_del-tmp_min_del)/2;
                        options.delta=tmp_middle;
                        [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=AEWAF(W, data_test, options, ID);
                        cur_query=NumReq/n;
                        tmp_del(num_while)=options.delta;
                        tmp_que(num_while)=cur_query;
                        num_while=num_while+1;
                    end
                    if num_while > num_search,
                        [C,I]=min(abs(tmp_que-last_que2));
                        options.delta=tmp_del(I);
                        [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=AEWAF(W, data_test, options, ID);                        
                    end
                    last_max_del_AE=options.delta;
                end            
            end
            reg_AE(i)    = Regret;
            num_AE(i)    = NumReq;
            time_AE(i)   = run_time;
            REG_AE(i,:)  = regrets;
            NUMs_AE(i,:) = NUMs;
            TMs_AE(i,:)  = TMs;
            last_que=NumReq/n;
%% 4. AEWAF2    
            options.delta=last_max_del_AE2;         
            [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=AEWAF2(W, data_test, options, ID);
            cur_query=NumReq/n;
            if i<1,
                tmp_del=[];
                tmp_que=[];
                if abs(cur_query-last_que)/last_que > query_tol,
                    tmp_max_del=max_del;
                    tmp_min_del=min_del;
                    num_while=1;                
                    while abs(cur_query-last_que)/last_que > query_tol && num_while<=num_search,
                        if cur_query > last_que,
                            tmp_min_del=options.delta;
                        else
                            tmp_max_del=options.delta;
                        end
                        tmp_middle = tmp_max_del-(tmp_max_del-tmp_min_del)/2;
                        options.delta=tmp_middle;
                        tmp=randperm(20);
                        rand_id=tmp(1);
                        tmpID=ID_list(rand_id,:);
                        [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=AEWAF2(W, data_test, options, tmpID);
                        cur_query=NumReq/n;
                        tmp_del(num_while)=options.delta;
                        tmp_que(num_while)=cur_query;
                        num_while=num_while+1;
                    end
                    if num_while > num_search,
                        [C,I]=min(abs(tmp_que-last_que));
                        options.delta=tmp_del(I);
                        [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=AEWAF2(W, data_test, options, ID);                           
                    end
                    last_max_del_AE2=options.delta;
                end
            end
            reg_AE2(i)    = Regret;
            num_AE2(i)    = NumReq;
            time_AE2(i)   = run_time;
            REG_AE2(i,:)  = regrets;
            NUMs_AE2(i,:) = NUMs;
            TMs_AE2(i,:)  = TMs;

%% 5. AGF       
            options.delta=last_max_del_AGF;
            [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=AGF(W, data_test, options, ID);
            cur_query=NumReq/n;
            if i<1,
                tmp_del=[];
                tmp_que=[];
                if abs(cur_query-last_que)/last_que > query_tol,
                    tmp_max_del=max_del1;
                    tmp_min_del=min_del1;
                    num_while=1;                
                    while abs(cur_query-last_que)/last_que > query_tol && num_while<=num_search,
                        if cur_query > last_que,
                            tmp_min_del=options.delta;
                        else
                            tmp_max_del=options.delta;
                        end
                        tmp_middle = tmp_max_del-(tmp_max_del-tmp_min_del)/2;
                        options.delta=tmp_middle;
                        [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=AGF(W, data_test, options, ID);
                        cur_query=NumReq/n;
                        tmp_del(num_while)=options.delta;
                        tmp_que(num_while)=cur_query;
                        num_while=num_while+1;
                    end
                    if num_while > num_search,
                        [C,I]=min(abs(tmp_que-last_que));
                        options.delta=tmp_del(I);
                        [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=AGF(W, data_test, options, ID);                        
                    end
                    last_max_del_AGF=options.delta;
                end  
            end
            reg_AF(i)    = Regret;
            num_AF(i)    = NumReq;
            time_AF(i)   = run_time;
            REG_AF(i,:)  = regrets;
            NUMs_AF(i,:) = NUMs;
            TMs_AF(i,:)  = TMs;
     
            last_que2=NumReq/n;

%% 6. AGF2:     
            options.delta=last_max_del_AGF2;
            [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=AGF2(W, data_test, options, ID);
            cur_query=NumReq/n;
            if i<1,
                tmp_del=[];
                tmp_que=[];
                if abs(cur_query-last_que2)/last_que2 > query_tol,
                    tmp_max_del=max_del1;
                    tmp_min_del=min_del1;
                    num_while=1;                
                    while abs(cur_query-last_que2)/last_que2 > query_tol && num_while<=num_search,
                        if cur_query > last_que2,
                            tmp_min_del=options.delta;
                        else
                            tmp_max_del=options.delta;
                        end
                        tmp_middle = tmp_max_del-(tmp_max_del-tmp_min_del)/2;
                        options.delta=tmp_middle;
                        [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=AGF2(W, data_test, options, ID);
                        cur_query=NumReq/n;
                        tmp_del(num_while)=options.delta;
                        tmp_que(num_while)=cur_query;
                        num_while=num_while+1;
                    end
                    if num_while > num_search,
                        [C,I]=min(abs(tmp_que-last_que));
                        options.delta=tmp_del(I);
                        [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=AGF2(W, data_test, options, ID);                        
                    end
                    last_max_del_AGF2=options.delta;
                end    
            end
            reg_AF2(i)    = Regret;
            num_AF2(i)    = NumReq;
            time_AF2(i)   = run_time;
            REG_AF2(i,:)  = regrets;
            NUMs_AF2(i,:) = NUMs;
            TMs_AF2(i,:)  = TMs;
%             fprintf(1,'AF2:%.3f,del:%.3f\n',NumReq/n,options.delta);
            last_que=num_AF2(i)/n;
        end

        rho_RGF           = mean(num_AF2)/n;
        options.rho_RGF   = rho_RGF;       
        

        rho_REWAF         = mean(num_AE2)/n;
        options.rho_REWAF = rho_REWAF;
        
%         for i=1:1,
        for i=1:5,%size(ID_list,1),
            ID = ID_list(i,:);

            %% 1. Random Exponentially Weighted Forecaster
            options.rho_REWAF=que_ran(j);%mean(num_AE)/n;  
            [ Regret, NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=REWAF(W, data_test, options, ID);
            reg_RE(i)    = Regret;
            num_RE(i)    = NumReq;
            time_RE(i)   = run_time;
            REG_RE(i,:)  = regrets;
            NUMs_RE(i,:) = NUMs;
            TMs_RE(i,:)  = TMs;
%%             
            options.rho_REWAF=que_ran(j);%mean(num_AE2)/n;  
            [ Regret, NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=REWAF(W, data_test, options, ID);
            reg_RE2(i)    = Regret;
            num_RE2(i)    = NumReq;
            time_RE2(i)   = run_time;
            REG_RE2(i,:)  = regrets;
            NUMs_RE2(i,:) = NUMs;
            TMs_RE2(i,:)  = TMs;
%%            
            % 2. Random Greedy Forecaster
            options.rho_RGF= que_ran(j);%mean(num_AF)/n;
            [ Regret, NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=RGF(W, data_test, options, ID);
            reg_RF(i)    = Regret;
            num_RF(i)    = NumReq;
            time_RF(i)   = run_time;
            REG_RF(i,:)  = regrets;
            NUMs_RF(i,:) = NUMs;
            TMs_RF(i,:)  = TMs;
%%            
            options.rho_RGF= mean(num_AF2)/n;
            [ Regret, NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=RGF(W, data_test, options, ID);
            reg_RF2(i)    = Regret;
            num_RF2(i)    = NumReq;
            time_RF2(i)   = run_time;
            REG_RF2(i,:)  = regrets;
            NUMs_RF2(i,:) = NUMs;
            TMs_RF2(i,:)  = TMs;
        end
        
        fprintf(fileID,'-------------------------------------------------------------------------------\n');
        fprintf(fileID,'Algorithm : Regret Rates, Numer of Required Lables,  Running Time\n');
        fprintf(fileID,'&EWAF   &%.3f \t\\%%$\\pm$ %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f  \t \\\\\n', mean(reg_EW)/n*100, std(reg_EW)/n*100, mean(num_EW)/n*100, std(num_EW)/n*100, mean(time_EW));
        fprintf(fileID,'&REWAF  &%.3f \t\\%%$\\pm$ %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f  \t \\\\\n', mean(reg_RE)/n*100, std(reg_RE)/n*100, mean(num_RE)/n*100, std(num_RE)/n*100, mean(time_RE));
        fprintf(fileID,'&REWAF2 &%.3f \t\\%%$\\pm$ %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f  \t \\\\\n', mean(reg_RE2)/n*100, std(reg_RE2)/n*100, mean(num_RE2)/n*100, std(num_RE2)/n*100, mean(time_RE2));
        fprintf(fileID,'&AEWAF  &%.3f \t\\%%$\\pm$ %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f  \t \\\\\n', mean(reg_AE)/n*100, std(reg_AE)/n*100, mean(num_AE)/n*100, std(num_AE)/n*100, mean(time_AE));
        fprintf(fileID,'&AEWAF2 &%.3f \t\\%%$\\pm$ %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f  \t \\\\\n', mean(reg_AE2)/n*100, std(reg_AE2)/n*100, mean(num_AE2)/n*100, std(num_AE2)/n*100, mean(time_AE2));
        fprintf(fileID,'&GF     &%.3f \t\\%%$\\pm$ %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f  \t \\\\\n', mean(reg_GF)/n*100, std(reg_GF)/n*100, mean(num_GF)/n*100, std(num_GF)/n*100, mean(time_GF));    
        fprintf(fileID,'&RGF    &%.3f \t\\%%$\\pm$ %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f  \t \\\\\n', mean(reg_RF)/n*100, std(reg_RF)/n*100, mean(num_RF)/n*100, std(num_RF)/n*100, mean(time_RF));    
        fprintf(fileID,'&RGF2   &%.3f \t\\%%$\\pm$ %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f  \t \\\\\n', mean(reg_RF2)/n*100, std(reg_RF2)/n*100, mean(num_RF2)/n*100, std(num_RF2)/n*100, mean(time_RF2));    
        fprintf(fileID,'&AGF    &%.3f \t\\%%$\\pm$ %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f  \t \\\\\n', mean(reg_AF)/n*100, std(reg_AF)/n*100, mean(num_AF)/n*100, std(num_AF)/n*100, mean(time_AF));
        fprintf(fileID,'&AGF2   &%.3f \t\\%%$\\pm$ %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f  \t \\\\\n', mean(reg_AF2)/n*100, std(reg_AF2)/n*100, mean(num_AF2)/n*100, std(num_AF2)/n*100, mean(time_AF2));
        fprintf(fileID,'rho_RGF=%.3f  \n', rho_RGF );
        fprintf(fileID,'rho_REWAF=%.3f  \n', rho_REWAF);
        fprintf(fileID,'-------------------------------------------------------------------------------\n');
        
        EW_time(j)=mean(time_EW);
        REW_time(j)=mean(time_RE);
        AEW_time(j)=mean(time_AE);
        AEW2_time(j)=mean(time_AE2);

        GF_time(j)=mean(time_GF);
        RGF_time(j)=mean(time_RF);
        AGF_time(j)=mean(time_AF);
        AGF2_time(j)=mean(time_AF2);
        
        AEW_que(j)=mean(num_AE)/n;
        AEW_reg(j)=mean(reg_AE)/n*100;
        AEW2_que(j)=mean(num_AE2)/n;
        AEW2_reg(j)=mean(reg_AE2)/n*100;
        EW_reg(j)=mean(reg_EW)/n*100;
        REW_reg(j)=mean(reg_RE)/n*100;
        REW2_reg(j)=mean(reg_RE2)/n*100;

        AGF_que(j)=mean(num_AF)/n;
        AGF_reg(j)=mean(reg_AF)/n*100;    
        GF_reg(j)=mean(reg_GF)/n*100;
        RGF_reg(j)=mean(reg_RF)/n*100;
        RGF2_reg(j)=mean(reg_RF2)/n*100;

        AGF2_que(j)=mean(num_AF2)/n;
        AGF2_reg(j)=mean(reg_AF2)/n*100;
        
        REW_std(j)=std(reg_RE)/n*100;
        REW2_std(j)=std(reg_RE2)/n*100;
        AEW_std(j)=std(reg_AE)/n*100;
        AEW2_std(j)=std(reg_AE2)/n*100;

        RGF_std(j)=std(reg_RF)/n*100;
        RGF2_std(j)=std(reg_RF2)/n*100;
        AGF_std(j)=std(reg_AF)/n*100;
        AGF2_std(j)=std(reg_AF2)/n*100;
        
        REW_que=mean(num_RE)/n*100;
        RGF_que=mean(num_RF)/n*100;
        
    end
    dlmwrite(fullfile(loc,dataset{t}),AEW_que, 'delimiter', '\t');    
    dlmwrite(fullfile(loc,dataset{t}),EW_reg, '-append','delimiter', '\t');
    dlmwrite(fullfile(loc,dataset{t}),REW_reg, '-append','delimiter', '\t');
    dlmwrite(fullfile(loc,dataset{t}),AEW_reg, '-append','delimiter', '\t'); %3
    
    dlmwrite(fullfile(loc,dataset{t}),AEW2_que, '-append','delimiter', '\t');
    dlmwrite(fullfile(loc,dataset{t}),AEW2_reg, '-append','delimiter', '\t');
    dlmwrite(fullfile(loc,dataset{t}),REW2_reg, '-append','delimiter', '\t'); %6
    
    dlmwrite(fullfile(loc,dataset{t}),AGF_que, '-append','delimiter', '\t');
    dlmwrite(fullfile(loc,dataset{t}),GF_reg, '-append','delimiter', '\t');
    dlmwrite(fullfile(loc,dataset{t}),RGF_reg, '-append','delimiter', '\t');
    dlmwrite(fullfile(loc,dataset{t}),AGF_reg, '-append','delimiter', '\t'); %10

    dlmwrite(fullfile(loc,dataset{t}),AGF2_que, '-append','delimiter', '\t');
    dlmwrite(fullfile(loc,dataset{t}),AGF2_reg, '-append','delimiter', '\t');
    dlmwrite(fullfile(loc,dataset{t}),RGF2_reg, '-append','delimiter', '\t'); %13
    
    dlmwrite(fullfile(loc,dataset{t}),REW_std, '-append','delimiter', '\t');
    dlmwrite(fullfile(loc,dataset{t}),REW2_std, '-append','delimiter', '\t');
    dlmwrite(fullfile(loc,dataset{t}),AEW_std, '-append','delimiter', '\t');
    dlmwrite(fullfile(loc,dataset{t}),AEW2_std, '-append','delimiter', '\t'); %17

    dlmwrite(fullfile(loc,dataset{t}),RGF_std, '-append','delimiter', '\t');
    dlmwrite(fullfile(loc,dataset{t}),RGF2_std, '-append','delimiter', '\t');
    dlmwrite(fullfile(loc,dataset{t}),AGF_std, '-append','delimiter', '\t');
    dlmwrite(fullfile(loc,dataset{t}),AGF2_std, '-append','delimiter', '\t'); %21
    
    dlmwrite(fullfile(loc,dataset{t}),EW_time, '-append','delimiter', '\t');
    dlmwrite(fullfile(loc,dataset{t}),REW_time, '-append','delimiter', '\t');
    dlmwrite(fullfile(loc,dataset{t}),AEW_time, '-append','delimiter', '\t');
    dlmwrite(fullfile(loc,dataset{t}),AEW2_time, '-append','delimiter', '\t'); %25

    dlmwrite(fullfile(loc,dataset{t}),GF_time, '-append','delimiter', '\t');
    dlmwrite(fullfile(loc,dataset{t}),RGF_time, '-append','delimiter', '\t');
    dlmwrite(fullfile(loc,dataset{t}),AGF_time, '-append','delimiter', '\t');
    dlmwrite(fullfile(loc,dataset{t}),AGF2_time, '-append','delimiter', '\t'); %29
    
    %dlmwrite(fullfile(loc,dataset{t}),REW_que, '-append','delimiter', '\t');
    %dlmwrite(fullfile(loc,dataset{t}),RGF_que, '-append','delimiter', '\t');    
    display(dataset{t});
    
    fclose(fileID);
end