function [classifier] = ALMA(data)
% ALMA_K_M:Approximate Large Margin algorithm (Claudio Gentile. A new approximate maximal margin classification algorithm. JMLR, 2:213–242, 2001.)
%--------------------------------------------------------------------------
% Input:
%        Y:    the column vector of lables, where Y(t) is the lable for t-th instance ;
%        K:    precomputed kernel for all the example, i.e., K_{ij}=K(x_i,x_j);
%  options:    a struct containing t_tick;
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

err_count = 0;

alpha     = 0.9;
B         = 1/alpha;
C         = sqrt(2);
p         = 2;
k         = 1;
%% loop
tic
for t = 1:n,
    
    x_t=X(t,:)';
    
    x_t_hat=x_t/norm(x_t);
    f_t_hat=w'*x_t_hat;

    hat_y_t = sign(f_t_hat);
    if (hat_y_t==0)
        hat_y_t=1;
    end
    
    y_t=Y(t);
    if (hat_y_t~=y_t),
        err_count = err_count + 1;
    end

    gamma_t=B*sqrt((p-1)/k);
    
    if y_t*f_t_hat<=(1-alpha)*gamma_t,
        
        eta_t=C/sqrt((p-1)*k);
        w=w+eta_t*y_t*x_t_hat;
   
        w=w/max(1, norm(w));
        k=k+1;
        
    end

    run_time=toc;
end

classifier=w;
fprintf(1,'The number of mistakes = %d\n', err_count);
run_time = toc;