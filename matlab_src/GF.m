function [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=GF(W, data, options, ID)
%% GF: Greedy Forecaster for expert advice
%%-------------------------------------------------------------------------
%%
%%
%%-------------------------------------------------------------------------
%% initialize parameters
N       = size(W,2);
t_tick  = options.t_tick;
eta     = options.eta ;
%% preprocessing dataset
Y = data(:,1); Y = (Y+1)/2;    %% convert {-1, 1} to {0, 1}
X = data(:,2:end);
%% initialize stats
L_EXP     = zeros(N,1);        % loss of all experts
L_F       = 0;             % loss of AGF
NumReq    = 0;

%% initialize for ploting
regrets     = [];
regrets_idx = [];
NUMs        = [];
TMs         = [];

%% loop
tic
for t=1:length(ID),
    id=ID(t);

%     display(t);
    %% compute f_t
    x_t=X(id,:)';
    f=max(0, min(1, W'*x_t+0.5));
    
    %% random guess of N-th classfier
    if options.numNoiseExperts > 0,
        for i=1:options.numNoiseExperts,
            f(N-i-1)=rand;
        end
    end

    %% compute ell(f_t,0) and ell(f_t,1)
    ell_1=abs(f-1);
    ell_0=abs(f-0);

    %% compute p_t
    temp_alpha = exp(eta*(L_F-L_EXP-ell_1));
    temp_beta  = exp(eta*(L_F-L_EXP-ell_0));

    sum_alpha = sum(temp_alpha);
    sum_beta  = sum(temp_beta);
    bar_p_t   = 1/2+1/(2*eta)*log(sum_alpha/sum_beta);
    p_t       = max(0,min(1,bar_p_t));
    
    %% update loss_expert
    NumReq=NumReq+1;
    y_t=Y(id);
    ell_EXP=abs(f-y_t);
    L_EXP=L_EXP+ell_EXP;

    %% update loss_F
    ell_F=abs(p_t-y_t);
    L_F=L_F+ell_F;
    
    %% compute regret
    Regret=L_F-min(L_EXP);
    
    %% record performance
    run_time = toc;
    if (mod(t,t_tick)==0),
        regrets = [regrets Regret/t];
        regrets_idx = [regrets_idx t];
        NUMs = [NUMs NumReq];
        TMs=[TMs run_time];
    end

end


Regret=L_F-min(L_EXP);
run_time = toc;
% fprintf(1,'GF:  Regret: %d; Numer: %d; Time: %d.\n',Regret, NumReq, run_time);

