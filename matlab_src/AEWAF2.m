function [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=AEWAF2(W, data, options, ID)
%% AEWAF:  Active Exponentially Weighted Forecaster
%%-------------------------------------------------------------------------
%%
%%
%%-------------------------------------------------------------------------
%% initialize parameters
N       = size(W,2);
t_tick  = options.t_tick;
eta     = options.eta ;
delta   = options.delta;
w       = ones(N,1)/N;
%% initialize dataset
Y = data(:,1); Y = (Y+1)/2;    %% convert {-1, 1} to {0, 1}
X = data(:,2:end);
%% initialize stats
L_EXP       = zeros(N,1);    % loss of all experts
L_F         = 0;             % loss of forecasster
L_hat_EXP   = zeros(N,1);

NumReq      = 1;
NumUnReq    = 0;
%% initialize value for plotting figures
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
    %% compute p_t
    p_t=w'*f;
    
    %% compute sum_wij, max_wij
    mu=NumUnReq/NumReq;
    for i=1:N,
        abs_fij(:,i)= exp(-eta*((1+mu)*L_hat_EXP(i)+L_hat_EXP)).*abs(f(i)-f);
    end
    max_wij=max(max(abs_fij));
%         tmp=max(abs_fij);
%         max_wij(max_wij<tmp)=tmp;
%         if max(abs_fij)>max_wij,
%             max_wij=
%         end
%         for j=1:N,            
%             abs_fij= exp(-eta*((1+mu)*L_hat_EXP(i)+L_hat_EXP(j)))*abs(f(i)-f(j));
%             %abs_fij=abs_fij/sum_wij;
%             if abs_fij > max_wij,
%                 max_wij = abs_fij;
%             end
%         end
%     end
    
%     max_wij=max_wij*N*N;
    %% update loss_expert
    y_t=Y(id);
    ell_EXP=abs(f-y_t);
    L_EXP=L_EXP+ell_EXP;
    
    %% update w    
    if  max_wij*options.multiplierAE2>delta,
        NumReq=NumReq+1;
        w=w.*exp(-eta*ell_EXP);
        L_hat_EXP=L_hat_EXP+ell_EXP;
        sum_w=sum(w);
        w=w/sum_w;
    else
        NumUnReq=NumUnReq+1;
    end

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

