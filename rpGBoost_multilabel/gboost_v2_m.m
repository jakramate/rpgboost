function [w, obj, trainerr, testerr] = gboost_v2_m(grad_fun, h, x, y, T, ep, lamb, reg_type, q, Q, xs, ys)

% BOOST_V2  Gradient Boosting.   
% grad_fun  -- func handle for the grad of the loss (user supplied)
% h         -- func handle to base learner (user supplied)
% x         -- design matrix
% y         -- response vector
% T         -- boosting round
% ep        -- shrinkage (learning rate)
% lamb      -- regularisation parameters

d   = size(x,2);
ncls   = size(y,2);
F   = ones(size(y)) .* mean(y);
w   = zeros(d+3, ncls, T);
obj = zeros(1,T);                 % storing objective function values

%>>>
if nargin == 7, needplots=0; testerr=nan; 
else needplots=1; testerr= zeros(1,T); end
trainerr=zeros(1,T);
%<<<

for t=1:T 
    options.maxIter    = 10;
    options.Display    = false;
    
    % fitting non-linear weak learner    
    % checkgrad(func2str(grad_fun), randn(d+3,1), 1e-10, F, h, x, y, lamb, 0)    
    [theta, fv, ~, ~] = minFunc(grad_fun, randn((d+3)*ncls, 1), options, F, h, x, y, lamb, 0, q, Q, reg_type);  
    
    theta = reshape(theta, d+3, ncls);

    F  = F + (ep * h(x, theta));     % updating the ensemble
        
    w(:,:,t) = theta;
    obj(t) = fv;
    
    %>>>
%     trainerr(t) = sum(y ~= sign(eval_ensemble(h, w, ep, x, y)))/length(y); 
%     if needplots,
%       testerr(t) = sum(ys ~= sign(eval_ensemble(h, w, ep, xs, y)))/length(ys); % for plots
%     end
    %<<<       
end
end
