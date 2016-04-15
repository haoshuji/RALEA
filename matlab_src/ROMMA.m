function [classifier] = ROMMA(data)
% ROMMA_K_M: Relaxed Online Maximum Margin Algorithm (Yi Li and Philip M. Long. The relaxed online maxiumu margin algorithm. In NIPS, pages 498-504, 1999.)
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
        err_count=err_count+1;
    end


    if norm(w)==0,
        if  hat_y_t~=y_t,
            w=w+y_t*x_t;
        end
    else
        if norm(x_t)~=0
            if  hat_y_t~=y_t,
                c_t=(norm(x_t)^2*norm(w)^2-y_t*f_t)/(norm(x_t)^2*norm(w)^2-f_t^2);
                d_t=(norm(w)^2*(y_t-f_t))/(norm(x_t)^2*norm(w)^2-f_t^2);
                w=c_t*w+d_t*x_t;
            end
        end
    end
    run_time = toc;

end

classifier=w;
fprintf(1,'The number of mistakes = %d\n', err_count);
run_time = toc;