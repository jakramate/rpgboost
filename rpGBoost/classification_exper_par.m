
%% added 'parfor' for faster cross validation
close all
% gradient boost experiment
T     = 1000 ;     % boosting rounds
%ep    = 1e-1;      % shrinkage (learning rate) - commented out to set it outside
lamb1 = 1e-2;       % reg param \tilde{lambda} for base learner params a,b.
lamb2 = 1e-5;       % regularisation parameter for alpha (only for rpgboost_v1)
K     = 1   ;       % projection dimension, K is the starting value for 'k'
reg_type = 1;       % always use 1 here
reps=10; % nr of independent repetitions

%>>> % these can be set outside and commented out here
%METHODS=[0 0 0 1]; % selection of methods we want to run
%rp_type = 's';  %'g','s'    % how rp is generated. 'g' for iid Gaussian
%loss = 'exp'; % 'exp','log'
%k_type='krnd'; % others don't work well.
%<<<

if strcmp(loss,'log')
    fun_psdo = @psdo_log;  fun_grad_alpha = @grad_alpha_log;
    fun_grad_theta = @grad_theta_log;
elseif strcmp(loss,'exp')
    fun_psdo = @psdo_exp;  fun_grad_alpha = @grad_alpha_exp;
    fun_grad_theta = @grad_theta_exp;
else
    disp('error:unknown loss')
end

%>>> for GPU processing
if gpuDeviceCount > 0
    learner = @tanh_learner_gpu;  % can speed up a bit for larger datasets
else
    learner = @tanh_learner;
end
%<<<

% loading dataset
for data=datasets
    load(cell2mat(data)); load([cell2mat(data) '_perms'])
    y = castLabel(y, -1);
    %nt   = floor(size(x,1)* 0.8); % nr training points
    % NOTE for RaSE datasets we'll want to set this outside to vary it

    if METHODS(4) || METHODS(3)
        if strcmp(k_type,'klearn')
            K=1;
        elseif strcmp(k_type,'k1')
            SCHEDULE = ones(1,T)*1;
            K=SCHEDULE;
        elseif strcmp(k_type,'k12345')
            SCHEDULE = [ones(1,round(T/5))*1, ones(1,round(T/5))*2, ones(1,round(T/5))*3, ones(1,round(T/5))*4, ones(1,round(T/5))*5];
    		K=SCHEDULE;
        elseif strcmp(k_type,'krnd')
            rng(0,'twister'); % fixing the random seed
            %SCHEDULE = -log(rand(1,T)); SCHEDULE=round(SCHEDULE/max(SCHEDULE)*rank(x)/2+1);K=SCHEDULE;
            SCHEDULE = -log(rand(1,T)); SCHEDULE=round(SCHEDULE/max(SCHEDULE)*rank(x)-1);K=SCHEDULE;
        else, disp('error: unknown k-schedule');
        end
    end

    lamb1_range=[1./10.^(1:2:7), 0]; lamb2_range=[1./10.^(1:2:7), 0]; % if we regularise the classifier weights we use this line - works better
    %%%lamb1_range=[0]; lamb2_range=[0];  % if we don't regularise the classifier weights we use this line

    % variables to store results - we will only use v4 (rp or rs ensemble) or v2 (ensemble on the original data)
    err_v1  = zeros(reps,1); err_v2  = err_v1; err_v3  = err_v1; err_v4  = err_v1; err_v5  = err_v1; err_v6=err_v1; err_stump = err_v1;
    %>>>
    result1.err = zeros(reps,1); % error in the last round
    result1.lambda1 = zeros(reps,1);
    result1.k = zeros(reps,T); % one row for each repeat
    result1.testerr = zeros(reps,T); % one row for each repeat
    result1.trainerr = zeros(reps,T); % one row for each repeat
    result1.alpha=zeros(reps,T); result1.a=zeros(reps,T);result1.b=zeros(reps,T);result1.v=zeros(reps,T);
    %
    result2.err = zeros(reps,1); % error in the last round
    result2.lambda1 = zeros(reps,1);
    result2.k = zeros(reps,T); % one row for each repeat
    result2.testerr = zeros(reps,T); % one row for each repeat
    result2.trainerr = zeros(reps,T); % one row for each repeat
    result2.alpha=zeros(reps,T); result2.a=zeros(reps,T);result2.b=zeros(reps,T);result2.v=zeros(reps,T);
    %
    result3.err = zeros(reps,1); % error in the last round
    result3.lambda1 = zeros(reps,1);
    result3.lambda2 = zeros(reps,1);
    result3.k = zeros(reps,T); % one row for each repeat
    result3.testerr = zeros(reps,T); % one row for each repeat
    %%%result3.alphas = zeros(reps,T);       % one row for each repeat
    result3.trainerr = zeros(reps,T); % one row for each repeat
    result3.alpha=zeros(reps,T); result3.a=zeros(reps,T);result3.b=zeros(reps,T);result3.v=zeros(reps,T);
    %
    result4.err = zeros(reps,1); % error in the last round
    result4.lambda1 = zeros(reps,1);
    result4.k = zeros(reps,T); % one row for each repeat
    result4.testerr = zeros(reps,T); % one row for each repeat
    result4.trainerr = zeros(reps,T); % one row for each repeat
    result4.alpha=zeros(reps,T); result4.a=zeros(reps,T);result4.b=zeros(reps,T);result4.v=zeros(reps,T);
    %<<<

    for i=1:reps
        perm = perms(i,:); % use pre-defined permuatations to make results reproducible
        xt   = x(perm(1:nt),:);
        yt   = y(perm(1:nt));
        xs   = x(perm(nt+1:end),:);
        ys   = y(perm(nt+1:end));
        [xt, xs] = standardise(xt, xs);

        cvstep = floor(length(yt)/5);    % 5-fold cross-validation to set the regularisation param of the classifier weights
        findex = randi([1 5], 1, length(yt));

        %         % AdaBoost + Decision Stump
        %         weak_learner = tree_node_w(1);
        %         [MLearners, MWeights] = GentleAdaBoost(weak_learner, xt', yt', T);
        %         ResultM = sign(Classify(MLearners, MWeights, xs'));
        %         err_stump(i)  = sum(ys' ~= ResultM) / length(ys);
        %
        %         %% gradient boost with alpha learning dubbed 'v1'

        if METHODS(1) % obsolete
            suffix=[{'v1'} '_' loss 'ep=' num2str(ep) 'nt=' num2str(nt)];
            bestcv = 1e10;
            for lamb1 = lamb1_range
                fv = zeros(5,1);  % storing function values
                parfor f=1:5
                    xcvt = xt(findex~=f,:);
                    ycvt = yt(findex~=f);
                    xcvs = xt(findex==f,:);
                    ycvs = yt(findex==f);
                    [w, obj]  = gboost_v1(fun_psdo, fun_grad_alpha, learner, xcvt, ycvt, T, ep, lamb1, reg_type, xs,ys);
                    %fv(f) = obj(end);  % cv based-on function value
                    fv(f) = sum(ycvs ~= sign(eval_ensemble(learner, w, ep, xcvs, ycvt)))/length(ycvs);
                end
                if mean(fv) < bestcv
                    best_lamb1_v1(i) = lamb1;
                    bestcv = mean(fv);
                end
            end
            % now training the model with selected lamb1
            [wg1, obj1, trainerr,testerr]  = gboost_v1(fun_psdo, fun_grad_alpha, learner, xt, yt, T, ep, best_lamb1_v1(i), reg_type, xs,ys);
            err_v1(i) = sum(ys ~= sign(eval_ensemble(learner, wg1, ep, xs, yt)))/length(ys);
            %>>>
            result1.err(i) = err_v1(i); % one number for each repeat
            result1.lambda1(i) =  best_lamb1_v1(i); % one number for each repeat
            %  n/a result1.lambda2(i) = best_lamb2_rb1; % one number for each repeat
            %  n/a  result1.k(i,:) = best_k; % one row for each repeat
            result1.obj(i,:) = obj1;
            result1.trainerr(i,:) = trainerr; % one row for each repeat
            result1.testerr(i,:)= testerr; % for plots
            result1.alpha(i,:)=wg1(1,:); result1.a(i,:)=wg1(2,:); result1.b(i,:)=wg1(3,:);result1.v(i,:)=wg1(4,:);
            results.result1=result1; % all results
            %<<<
        end % if METHODS(1)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if METHODS(2) % this is gradient boosting ensemble on the original full data
            suffix=[{'v2'} '_' loss 'ep=' num2str(ep) 'nt=' num2str(nt)];
            %% (alpha is absorbed in w(1))
            %if i==1
            bestcv = 1e10;
            for lamb1 = lamb1_range
                fv = zeros(5,1);  % storing function values
                parfor f=1:5
                    xcvt = xt(findex~=f,:);
                    ycvt = yt(findex~=f);
                    xcvs = xt(findex==f,:);
                    ycvs = yt(findex==f);
                    [w, obj]  = gboost_v2(fun_grad_theta, learner, xcvt, ycvt, T, ep, lamb1, reg_type,    xs,ys);
                    %fv(f) = obj(end);   % cv based-on function value
                    fv(f) = sum(ycvs ~= sign(eval_ensemble(learner, w, ep, xcvs, ycvt)))/length(ycvs);
                end
                if mean(fv) < bestcv
                    best_lamb1_v2(i) = lamb1;
                    bestcv = mean(fv);
                end
            end
            %end
            % now training the model with selected lamb1
            [wg2, obj2,  trainerr,testerr]  = gboost_v2(fun_grad_theta, learner, xt, yt, T, ep, best_lamb1_v2(i),reg_type,     xs,ys);
            err_v2(i) = sum(ys ~= sign(eval_ensemble(learner, wg2, ep, xs, yt)))/length(ys);
            %>>>
            result2.err(i) = err_v2(i); % one number for each repeat
            result2.lambda1(i) =  best_lamb1_v2(i); % one number for each repeat
            %%  n/a   result2.k(i,:) = best_k; % one row for each repeat
            result2.obj(i,:) = obj2;
            result2.trainerr(i,:) = trainerr; % one row for each repeat
            result2.testerr(i,:)= testerr; % for plots
            result2.a(i,:)=wg2(1,:); result2.b(i,:)=wg2(2,:);result2.v(i,:)=wg2(3,:);
            results.result2=result2; % all results
            %<<<
        end % if METHODS(2)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if METHODS(3) % obsolete
            %% random projection boost with alpha learning
            suffix=['_' rp_type '_' loss '_' k_type 'ep=' num2str(ep) 'nt=' num2str(nt)];
            bestcv = 1e10;
            best_k_v3 = zeros(1,T);
            for lamb1 = lamb1_range
                for lamb2 = lamb2_range
                    fv = zeros(5,1);  % storing function values
                    k  = zeros(5,T);
                    parfor f=1:5
                        xcvt = xt(findex~=f,:);
                        ycvt = yt(findex~=f);
                        xcvs = xt(findex==f,:);
                        ycvs = yt(findex==f);
                        [w, k(f,:), obj]  = rpgboost_v1(fun_psdo, fun_grad_alpha, learner, xcvt, ycvt, T, K, ep, lamb2, lamb1, rp_type, reg_type,    xs,ys);  % K is the starting value of 'k'
                        %fv(f) = obj(end);   % cv based-on function value
                        fv(f) = sum(ycvs ~= sign(eval_ensemble(learner, w, ep, xcvs, ycvt)))/length(ycvs);
                    end
                    if mean(fv) < bestcv % cv criteria is the generalisation error
                        best_lamb1_v3(i) = lamb1;
                        best_lamb2_v3(i) = lamb2;
                        [min_v, min_idx] = min(fv);
                        best_k_v3        = k(min_idx,:);
                        bestcv = mean(fv);
                    end
                end
            end

            [wg3, ~, obj3,  trainerr,testerr]  = rpgboost_v1(fun_psdo, fun_grad_alpha, learner, xt, yt, T, best_k_v3, ep, best_lamb2_v3(i), best_lamb1_v3(i), rp_type, reg_type,    xs,ys);
            err_v3(i) = sum(ys ~= sign(eval_ensemble(learner, wg3, ep, xs, yt)))/length(ys);
            %>>>
            result3.err(i) = err_v3(i); % one number for each repeat
            result3.lambda1(i) = best_lamb1_v3(i); % one number for each repeat
            result3.lambda2(i) = best_lamb2_v3(i); % one number for each repeat
            result3.k(i,:) = best_k_v3; % one row for each repeat
            result3.obj(i,:) = obj3;
            result3.trainerr(i,:) = trainerr; % one row for each repeat
            result3.testerr(i,:)= testerr; % for plots
            result3.alpha(i,:)=wg3(1,:); result3.a(i,:)=wg3(2,:); result3.b(i,:)=wg3(3,:);result3.v(i,:)=wg3(4,:);
            results.result3=result3; % all results
            %<<<
        end % if METHODS(3)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if METHODS(4)  %% random projection or random subspace ensemble
            suffix=['_' rp_type '_' loss '_' k_type 'ep=' num2str(ep) 'nt=' num2str(nt)];
            bestcv = 1e10;
            best_k_v4 = zeros(1,T);
            for lamb1 = lamb1_range
                fv = zeros(5,1);  % storing function values
                k  = zeros(5,T);
                parfor f=1:5
                    xcvt = xt(findex~=f,:);
                    ycvt = yt(findex~=f);
                    xcvs = xt(findex==f,:);
                    ycvs = yt(findex==f);
                    [w, k(f,:), obj]  = rpgboost_v2(fun_grad_theta, learner, xcvt, ycvt, T, K, ep, lamb1, rp_type, reg_type, xs,ys);
                    %fv(f) = obj(end);   % cv based-on function value
                    fv(f) = sum(ycvs ~= sign(eval_ensemble(learner, w, ep, xcvs, ycvt)))/length(ycvs);
                end
                if mean(fv) < bestcv
                    best_lamb1_v4(i)  = lamb1;
                    [min_v, min_idx]  = min(fv);
                    best_k_v4         = k(min_idx,:);
                    bestcv            = mean(fv);
                end
            end
            [wg4, ~, obj4,  trainerr,testerr]  = rpgboost_v2(fun_grad_theta, learner, xt, yt, T, best_k_v4, ep, best_lamb1_v4(i), rp_type, reg_type,    xs,ys);
            err_v4(i) = sum(ys ~= sign(eval_ensemble(learner, wg4, ep, xs, yt)))/length(ys);
            %>>>
            result4.err(i) = err_v4(i); % one number for each repeat
            result4.lambda1(i) = best_lamb1_v4(i); % one number for each repeat
            % n/a result4.lambda2(i) = best_lamb2_rb1; % one number for each repeat
            result4.k(i,:) = best_k_v4; % one row for each repeat
            result4.obj(i,:) = obj4;
            result4.trainerr(i,:) = trainerr; % one row for each repeat
            result4.testerr(i,:)= testerr; % for plots
            result4.a(i,:)=wg4(1,:); result4.b(i,:)=wg4(2,:);result4.v(i,:)=wg4(3,:);
            results.result4=result4; % all results
            %<<<
        end % if METHODS(4)

        fprintf('[repetition %d] errors gb1=%4.3f, gb2=%4.3f, rb1=%4.3f, rb2=%4.3f\n', i,...
            err_v1(i),err_v2(i), err_v3(i), err_v4(i));
        save([cell2mat(data),'_result_',cell2mat(suffix)],'results','ep');
    end

    fprintf('[%s] T=%d, e=%6.5f\n',...
        cell2mat(data), T, ep);
    fprintf('mean errors gb1=%4.3f, gb2=%4.3f, rb1=%4.3f, rb2=%4.3f\n',...
        mean(err_v1),mean(err_v2),mean(err_v3),mean(err_v4));
end

