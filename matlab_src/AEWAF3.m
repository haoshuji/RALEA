function [ Regret,NumReq, run_time, regrets,regrets_idx,NUMs,TMs]=AEWAF3(W, data, options, ID)
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

NumReq      = 0;
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
    f(N)=rand;
    f(N-1)=rand;
    
    %% compute p_t
    p_t=w'*f;
    
    %% compute sum_wij, max_wij
    max_wij=0.0;
    sum_wij = 0.0;
%     for i=1:N,
%         for j=i:N,
%             if NumReq == 0,
%                 mul=0;
%             else 
%                 mul=NumUnReq/NumReq;
%             end
%             sum_wij = sum_wij+exp(-eta*(L_hat_EXP(i,1)+mul*L_hat_EXP(i,1)+L_hat_EXP(j,1)));
%         end
%     end
%     
%     for i=1:N,
%         for j=i:N,
%             if NumReq == 0,
%                 mul=0;
%             else 
%                 mul=NumUnReq/NumReq;
%             end
%             abs_fij= exp(-eta*(L_hat_EXP(i,1)+mul*L_hat_EXP(i,1)+L_hat_EXP(j,1)))*abs(f(i)-f(j));
%             abs_fij=abs_fij/sum_wij;
%             if abs_fij > max_wij,
%                 max_wij = abs_fij;
%             end
%         end
%     end
    sum_fij=0;
    pro=w.*f;
    pro=pro/norm(pro);
    entropy=0;
    for i=1:N,
        entropy=entropy+(-pro(i)*log(pro(i)));
    end
    
    
    %% update loss_expert
    y_t=Y(id);
    ell_EXP=abs(f-y_t);
    L_EXP=L_EXP+ell_EXP;
    
    %% update w    
%     if  max_wij*100>delta,
    if  entropy<delta,
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

