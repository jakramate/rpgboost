function [fv, dfv] = grad_theta_exp(w, F, h, x, y, lambda, k, reg_type)

% h = learner's output 
ht = -y .* (F + h(x, w));

n = length(y);
% computing function value at current 'w'
if k == 0  % for gboost_v2
	reg = lambda;
else    
	reg = lambda*sqrt(k+1)/sqrt(n);
end

% compute regularisation values
if reg_type == 1  
    % L1-regularisation
    reg_fv    = reg * (sqrt((w(1) + 1e-10)^2) + sqrt((w(2) + 1e-10)^2));
    reg_dfv_1 = reg * ((w(1)+1e-10)/sqrt((w(1) + 1e-10)^2));
    reg_dfv_2 = reg * ((w(2)+1e-10)/sqrt((w(2) + 1e-10)^2));
else 
    % L2 regularisation
    reg_fv    = reg * (w(1)^2 + w(2)^2);
    reg_dfv_1 = reg * 2 * w(1);
    reg_dfv_2 = reg * 2 * w(2);
end


fv = sum(exp(ht))/n + reg_fv;


% computing the gradients for 'w'
dfv    = zeros(length(w),1);
% precomputing the common term
cterm  = exp(ht); 
dfv(1) = sum((-y .* tanh(x*w(4:end) + w(3))) .* cterm)/n + reg_dfv_1;
dfv(2) = sum(-y .* cterm)/n + reg_dfv_2;
dfv(3) = sum((-y .* w(1) .* sech(x*w(4:end) + w(3)).^2) .* cterm)/n;
dfv(4:end) = (((-y .* w(1) .* sech(x*w(4:end) + w(3)).^2) .* cterm)' * x)/n;

end
