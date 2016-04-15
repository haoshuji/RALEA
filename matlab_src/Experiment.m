function Experiment(dataset_name)
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

%% load dataset
load(sprintf('data/%s',dataset_name));

%% Train Experts
%1. Perceptron
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

N=size(W,2);

%% options
n=size(data_test,1);
ID_list=ID_test;
options.eta    = (8*log(N)/n)^0.5;

options.t_tick = round(n/15);




%% delta
D=0.05:0.05:0.2;
for j=1:4,
    fprintf(1,'running on the %d-th trial...\n',j);
    
    delta=D(j);
    options.delta=delta;    


    %% Testing Active Greedy Forecaster
    for i=1:size(ID_list,1),
        %       fprintf(1,'running on the %d-th trial...\n',i);
        ID = ID_list(i,:);

        % 1. EWAF

        [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=EWAF(W, data_test, options, ID);
        reg_EW(i)    = Regret;
        num_EW(i)    = NumReq;
        time_EW(i)   = run_time;
        REG_EW(i,:)  = regrets;
        NUMs_EW(i,:) = NUMs;
        TMs_EW(i,:)  = TMs;

        % 2. Greedy Forecaster
        [ Regret, NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=GF(W, data_test, options, ID);
        reg_GF(i)    = Regret;
        num_GF(i)    = NumReq;
        time_GF(i)   = run_time;
        REG_GF(i,:)  = regrets;
        NUMs_GF(i,:) = NUMs;
        TMs_GF(i,:)  = TMs;


        % 3. AEWAF
        [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=AEWAF(W, data_test, options, ID);
        reg_AE(i)    = Regret;
        num_AE(i)    = NumReq;
        time_AE(i)   = run_time;
        REG_AE(i,:)  = regrets;
        NUMs_AE(i,:) = NUMs;
        TMs_AE(i,:)  = TMs;


        % 4. Active Greedy Forecaster
        [ Regret, NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=AGF(W, data_test, options, ID);
        reg_AF(i)    = Regret;
        num_AF(i)    = NumReq;
        time_AF(i)   = run_time;
        REG_AF(i,:)  = regrets;
        NUMs_AF(i,:) = NUMs;
        TMs_AF(i,:)  = TMs;

    end

    rho_RGF           = mean(num_AF)/n;
    options.rho_RGF   = rho_RGF;

    rho_REWAF         = mean(num_AE)/n;
    options.rho_REWAF = rho_REWAF;

    for i=1:size(ID_list,1),
        %     fprintf(1,'running on the %d-th trial...\n',i);
        ID = ID_list(i,:);
        
        % 1. Random Exponentially Weighted Forecaster
        [ Regret, NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=REWAF(W, data_test, options, ID);
        reg_RE(i)    = Regret;
        num_RE(i)    = NumReq;
        time_RE(i)   = run_time;
        REG_RE(i,:)  = regrets;
        NUMs_RE(i,:) = NUMs;
        TMs_RE(i,:)  = TMs;

        % 2. Random Greedy Forecaster
        [ Regret, NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=RGF(W, data_test, options, ID);
        reg_RF(i)    = Regret;
        num_RF(i)    = NumReq;
        time_RF(i)   = run_time;
        REG_RF(i,:)  = regrets;
        NUMs_RF(i,:) = NUMs;
        TMs_RF(i,:)  = TMs;
    end


    %% print and plot results
    figure
    mean_reg_EW = mean(REG_EW);
    plot(regrets_idx, mean_reg_EW,'k.-');
    hold on
    mean_reg_GF = mean(REG_GF);
    plot(regrets_idx, mean_reg_GF,'c-*');
    mean_reg_RE = mean(REG_RE);
    plot(regrets_idx, mean_reg_RE,'m-x');
    mean_reg_RF = mean(REG_RF);
    plot(regrets_idx, mean_reg_RF,'g-+');
    mean_reg_AE = mean(REG_AE);
    plot(regrets_idx, mean_reg_AE,'r-o');
    mean_reg_AF = mean(REG_AF);
    plot(regrets_idx, mean_reg_AF,'b-s');
    legend('EWAF','GF','REWAF','RGF','AEWAF','AGF');
    XLABEL('Number of samples');
    YLABEL('Online average rate of regret')
    grid

    fprintf(1,'-------------------------------------------------------------------------------\n');
    fprintf('Algorithm : Regret Rates, Numer of Required Lables,  Running Time\n');
    fprintf(1,'EWAF   &%.3f \t\\%%$\\pm$ %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f  \t \\\\\n', mean(reg_EW)/n*100, std(reg_EW)/n*100, mean(num_EW)/n*100, std(num_EW)/n*100, mean(time_EW));
    fprintf(1,'REWAF  &%.3f \t\\%%$\\pm$ %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f  \t \\\\\n', mean(reg_RE)/n*100, std(reg_RE)/n*100, mean(num_RE)/n*100, std(num_RE)/n*100, mean(time_RE));
    fprintf(1,'AEWAF  &%.3f \t\\%%$\\pm$ %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f  \t \\\\\n', mean(reg_AE)/n*100, std(reg_AE)/n*100, mean(num_AE)/n*100, std(num_AE)/n*100, mean(time_AE));
    fprintf(1,'GF     &%.3f \t\\%%$\\pm$ %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f  \t \\\\\n', mean(reg_GF)/n*100, std(reg_GF)/n*100, mean(num_GF)/n*100, std(num_GF)/n*100, mean(time_GF));    
    fprintf(1,'RGF    &%.3f \t\\%%$\\pm$ %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f  \t \\\\\n', mean(reg_RF)/n*100, std(reg_RF)/n*100, mean(num_RF)/n*100, std(num_RF)/n*100, mean(time_RF));    
    fprintf(1,'AGF    &%.3f \t\\%%$\\pm$ %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f  \t \\\\\n', mean(reg_AF)/n*100, std(reg_AF)/n*100, mean(num_AF)/n*100, std(num_AF)/n*100, mean(time_AF));
    fprintf(1,'rho_RGF=%.3f  \n', rho_RGF );
    fprintf(1,'rho_REWAF=%.3f  \n', rho_REWAF);
    fprintf(1,'delta=%.3f  \n', delta );
    fprintf(1,'-------------------------------------------------------------------------------\n');
end

%%
% figure
% mean_NUM_GF = mean(NUMs_GF);
% plot(regrets_idx, mean_NUM_GF,'c.-');
% hold on
% mean_NUM_RF = mean(NUMs_RF);
% plot(regrets_idx, mean_NUM_RF,'b-o');
% mean_NUM_AF = mean(NUMs_AF);
% plot(regrets_idx, mean_NUM_AF,'r-x');
% legend('GF','RGF','AGF','Location','NorthWest');
% XLABEL('Number of samples');
% YLABEL('Online average number of requested labels')
% grid
% 
% figure
% mean_TM_GF = log(mean(TMs_GF))/log(10);
% plot(regrets_idx, mean_TM_GF,'c.-');
% hold on
% mean_TM_RF = log(mean(TMs_RF))/log(10);
% plot(regrets_idx, mean_TM_RF,'b-o');
% mean_TM_AF = log(mean(TMs_AF))/log(10);
% plot(regrets_idx, mean_TM_AF,'r-x');
% legend('GF','RGF','AGF','Location','NorthWest');
% XLABEL('Number of samples');
% YLABEL('average time cost (log_{10} t)')
% grid