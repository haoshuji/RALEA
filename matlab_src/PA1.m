function [classifier] = PA1(data)
% PA1_K_M: online Passive-Aggressive algorithm-I(Koby Crammer, Ofer Dekel, Joseph Keshet, Shai Shalev-Shwartz, and Yoram Singer. Online passive-aggressive algorithms. JMLR, 7:551–585, 2006.)
%--------------------------------------------------------------------------
% Input:
%        Y:    the column vector of lables, where Y(t) is the lable for t-th instance ;
%        K:    precomputed kernel for all the example, i.e., K_{ij}=K(x_i,x_j);
%  options:    a struct containing C, tau, rho, sigma, t_tick;
%  id_list:    a random permutation of the 1,2,...,T;
% Output:
%  classifier:  a struct containing SV (the set of idexes for all the support vectors) and alpha (corresponding weights)
%   err_count:  total number of training errors
%    run_time:  time consumed by this algorithm at a time
%    mistakes:  a vector of online mistake rate
% mistake_idx:  a vector of index, in which every idex is a time and corresponds to a mistake rate in the vector above
%         SVs:  a vector recording the online number of support vectors for every idex in mistake_idx
%         TMs:  a vector recording the online time consumption
%--------------------------------------------------------------------------

%% initialize parameters
[n,d]       = size(data);

Y = data(:,1);
X = data(:,2:end);
w =zeros(d-1,1);

C         = 5;
err_count = 0;
%% loop
tic
for t = 1:n,

    x_t=X(t,:)';
    f_t=w'*x_t;

    hat_y_t = sign(f_t);
    if (hat_y_t==0)
        hat_y_t=1;
    end
    y_t=Y(t);
    if hat_y_t~=y_t,
        err_count = err_count + 1;
    end
    
    l_t=max(0,1-y_t*f_t);
    if l_t>0,
        norm_t=norm(x_t);
        tau_t=min(C, l_t/norm_t^2);
        w=w+tau_t*y_t*x_t;

    end

    run_time = toc;
end

classifier=w;
fprintf(1,'The number of mistakes = %d\n', err_count);
run_time = toc;
