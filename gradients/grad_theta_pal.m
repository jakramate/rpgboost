function [fv, dv] = grad_theta_pal(w, F, h, x, y, lambda, k, q, Q, reg_type)


n = length(y);
ncls   = size(y,2);


w = reshape(w, length(w)/ncls, ncls);

% h = learner's output 
ht = (F + h(x, w));


% computing function value at current 'w'
if k == 0  % for gboost_v2
	reg = lambda;
else    
    %Q = 1;
	%reg = lambda .* sqrt( q*k/n * log(n*Q))^1.5 + log(log(n))/n;
    reg = lambda .* sqrt( q)*k/n; 
    %reg = lambda;
end

% compute regularisation values
if reg_type == 1  
    % L1-regularisation
    reg_fv    = reg * sum(sum(sqrt((w(1:2,:) + 1e-10).^2))); 
    reg_dfv_1 = reg * sum(((w(1,:)+1e-10)./sqrt((w(1,:) + 1e-10).^2)));
    reg_dfv_2 = reg * sum(((w(2,:)+1e-10)./sqrt((w(2,:) + 1e-10).^2)));
else 
    % L2 regularisation
    reg_fv    = reg * (w(1)^2 + w(2)^2);
    reg_dfv_1 = reg * 2 * w(1);
    reg_dfv_2 = reg * 2 * w(2);
end


sm = softmax(ht);

fv = sum(sum( -y .* log(sm))/n) + reg_fv;


% computing the gradients for 'w'
dfv    = zeros(size(w,1), ncls);

for l=1:ncls
    % precomputing the common term    
    cterm  = -y(:,l) + sum(y .* sm(:,l),2);  
    dfv(1,l) = sum(cterm .* tanh(x*w(4:end,l) + w(3,l)))/n + reg_dfv_1;
    dfv(2,l) = sum(cterm)/n + reg_dfv_2;
    dfv(3,l) = sum(cterm .* w(1,l) .* sech(x*w(4:end,l) + w(3,l)).^2)/n;
    dfv(4:end,l) = ((cterm .* (w(1,l) .* sech(x*w(4:end,l) + w(3,l)).^2))' * x)/n;
end

dv   = cat(2, dfv(:));

end
