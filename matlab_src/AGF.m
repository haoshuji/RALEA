function [ Regret, NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=AGF(W, data, options, ID)
%% AGF: Active Greedy Forecaster for expert advice
%%-------------------------------------------------------------------------
%%
%%
%%-------------------------------------------------------------------------
%% initialize parameters
N       = size(W,2);
t_tick  = options.t_tick;
eta     = options.eta ;
delta   = options.delta;
%% preprocessing dataset
Y = data(:,1); Y = (Y+1)/2;                  % convert {-1, 1} to {0, 1}
X = data(:,2:end);

%% initialize stats
L_EXP     = zeros(N,1); % loss of all experts
hat_L_EXP = zeros(N,1); % loss of experts known by the forecaster
L_F       = 0;          % loss of AGF
hat_L_F   = 0;
NumReq    = 0;          % number required

%% initialize for ploting
regrets     = [];
regrets_idx = [];
NUMs        = [];
TMs         = [];

%% loop
tic
for t=1:length(ID),
    id=ID(t);

    %% compute f_t
    x_t=X(id,:)';
    f=max(0, min(1, W'*x_t+0.5));
    %% random guess of N-th classfier
    if options.numNoiseExperts > 0,
        for i=1:options.numNoiseExperts,
            f(N-i-1)=rand;
        end
    end

    %% compute hat_p_t
    ell_1=abs(f-1);
    ell_0=abs(f-0);

    temp_alpha = exp(eta*(hat_L_F-hat_L_EXP-ell_1));
    temp_beta  = exp(eta*(hat_L_F-hat_L_EXP-ell_0));

    sum_alpha = sum(temp_alpha);
    sum_beta  = sum(temp_beta);
    bar_p_t   = 1/2+1/(2*eta)*log(sum_alpha/sum_beta);
    hat_p_t   = max(0,min(1,bar_p_t));

    %% update  L_EXP
    y_t=Y(id);
    ell_EXP=abs(f-y_t);
    L_EXP=L_EXP+ell_EXP;

    %% update L_F
    ell_F=abs(hat_p_t-y_t);
    L_F=L_F+ell_F;


    %% update hat_L_EXP and hat_L_F
    diff=f-bar_p_t;
    if max(diff)>delta||min(diff)<-delta,
        NumReq=NumReq+1;
        hat_L_EXP = hat_L_EXP+ell_EXP;
        hat_L_F   = hat_L_F+ell_F;
    end

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
% fprintf(1,'AGF: Regret: %d; Numer: %d; Time: %d.\n',  Regret, NumReq, run_time);

