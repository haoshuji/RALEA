function [classifier] = AROW(data)
% CW: online Confidence Weighted algorithm
%--------------------------------------------------------------------------
% Input:
%        Y:    the vector of lables
%        K:    precomputed kernel for all the example, i.e., K_{ij}=K(x_i,x_j)
%  id_list:    a randomized ID list
%  options:    a struct containing rho, sigma, C, n_label and n_tick;
% Output:
%   err_count:  total number of training errors
%    run_time:  time consumed by this algorithm once
%    mistakes:  a vector of mistake rate
% mistake_idx:  a vector of number, in which every number corresponds to a
%               mistake rate in the vector above
%         SVs:  a vector records the number of support vectors
%     size_SV:  the size of final support set
%--------------------------------------------------------------------------

%% initialize parameters
[n, d]=size(data);

Y = data(:,1);
X = data(:,2:end);

err_count=0;

gamma = 1;
mu  = zeros(d-1, 1);
Sigma = eye(d-1);


%% loop
tic
% n;
for t = 1:n,

    x_t=X(t,:)';
%     if(mod(t,100)==0)
%         display(t);
%     end
    m_t = (mu')*x_t;            % decision function
    v_t=x_t'*Sigma*x_t;
    
    hat_y_t = sign(m_t);        % prediction
    if (hat_y_t==0)
        hat_y_t=1;
    end
    
    y_t=Y(t);
    if hat_y_t~=y_t,
        err_count=err_count+1;
    end
    
    if m_t*y_t<1,
        beta_t=1/(v_t+gamma);
        alpha_t=max(0, 1-y_t*m_t)*beta_t;
        mu=mu+alpha_t*Sigma*y_t*x_t;
        Sigma=Sigma-beta_t*Sigma*x_t*x_t'*Sigma;
    end

end

run_time = toc;

classifier=mu;
fprintf(1,'The number of mistakes = %d\n', err_count);
run_time = toc;
