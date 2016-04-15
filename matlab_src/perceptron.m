function [classifier] = perceptron(data)
% Perceptron_K_M: an online learning algorithm (F. Rosenblatt. The perceptron: A probabilistic model for information storage and organization in the brain. Psychological Review, 65:386–407, 1958.)
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
%        M_ds:  the number of strong double updating 
%        M_dw:  the number of weak double updating 
%         M_s:  the number of single updating
%--------------------------------------------------------------------------

%% initialize parameters
[n,d]       = size(data);

Y = data(:,1);
X = data(:,2:end);
w =zeros(d-1,1);

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
    if (hat_y_t~=y_t),
        w=w+y_t*x_t;
        err_count = err_count + 1;     
    end
    
    run_time = toc;
end

classifier=w;
fprintf(1,'The number of mistakes = %d\n', err_count);
run_time = toc;