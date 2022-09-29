function [w, best_k, obj, trainerr, testerr] = rpgboost_v2_m(grad_fun, h, x, y, T, K, ep, lamb, rp_type, reg_type, q, Q, xs, ys)
% RPGBOOST_V2  Random Projection based Gradient Boosting.
% grad_fun  -- func handle for the grad of the loss (user supplied)
% h         -- func handle to base learner (user supplied)
% x         -- design matrix
% y         -- response vector
% T         -- boosting round
% K         -- array of projection dimensions
% ep        -- shrinkage (learning rate)
% lamb      -- regularisation parameters
% rp_type   -- type of projection matrix

d      = size(x,2);
ncls   = size(y,2);
F      = ones(size(y)) .* mean(y); % there will be ncls of Fs
w      = zeros(d+3, ncls, T);
p      = zeros(size(x,2)+3, ncls);
obj    = zeros(1,T);             % storing objective function values
best_k = zeros(1,T);

%>>>
if nargin == 9, needplots=0; testerr=nan;
else needplots=1; testerr= zeros(1,T); end
trainerr=zeros(1,T);
%<<<

for t=1:T
    
    if length(K) > 1
        rm = rp(d, K(t), rp_type);
    else
        rm = rp(d, K, rp_type);
    end
    
    best_obj = inf;
    while 1
        k  = size(rm,2);
        xp = x * rm;
        
        options.maxIter    = 10;
        options.Display    = false;
        % fitting non-linear weak learner
        %checkgrad(func2str(grad_fun), randn((k+3)*ncls, 1), 1e-10, F, h, xp, y, lamb, k, reg_type)
        [theta, fv, ~, ~] = minFunc(grad_fun, randn((k+3)*ncls, 1), options, F, h, xp, y, lamb, k, q, Q, reg_type);
        
        theta      = reshape(theta, k+3, ncls);
        p(1:3,:)   = theta(1:3,:);
        p(4:end,:) = rm * theta(4:end,:);
        
        if best_obj - fv > 1e-5 && k < d
            best_obj   = fv;
            best_p     = p;
            if length(K) > 1  % for when an array of ks is supplied
                break;
            end
            best_k(t)  = k;
            rm         = augmented_rp(rm, rp_type);
        else
            break;
        end
    end
    
    F  = F + (ep * h(x, best_p));     % updating the ensemble
    
    w(:,:,t) = best_p;
    obj(t) = best_obj;
    %>>>
%     yhat = (softmax(eval_ensemble(h, w, ep, x, y)) > 1/ncls);
%     trainerr(t) = Hamming_score(yhat, y);
%     
%     if needplots
%         yhat = (softmax(eval_ensemble(h, w, ep, xs, y)) > 1/ncls);
%         testerr(t) = Hamming_score(yhat, ys);
%     end
    
    if length(K)>1
        best_k=K; % FIXED SCHEDULE (otherwise it returns 0)
    end
end
end
