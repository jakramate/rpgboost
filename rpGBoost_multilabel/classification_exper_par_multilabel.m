%ToDo: need to also record running time

%% added parellel for loop 'parfor' for faster cross validation
close all
% gradient boost experiment
T     = 1000 ;       % boosting rounds
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
elseif strcmp(loss,'pal')
    fun_grad_theta = @grad_theta_pal;
else
    disp('error:unknown loss')
end

%>>> for GPU processing
if gpuDeviceCount > 0
    gpuDevice(1);
    learner = @tanh_learner_gpu;  % can speed up a bit for larger datasets
else
    learner = @tanh_learner;
end
%<<<

% loading dataset
for data=datasets
    load(cell2mat(data)); load([cell2mat(data) '_perms'])
    if size(y,2) < 2
        y = onehot(y); % for multiclass dataset
    end

    % for multi-label dataset
    ncls   = size(y,2);

    nt   = floor(size(x,1)* 0.8); % nr training points
    % NOTE for RaSE datasets we'll want to set this outside to vary it

    q = max(sum(y,2));
    Q = size(y,2);

    if METHODS(4)|| METHODS(3)
        if strcmp(k_type,'klearn')
            K=1;
        elseif strcmp(k_type,'k1')
            SCHEDULE = ones(1,T)*1; K=SCHEDULE;
        elseif strcmp(k_type,'k12345')
            SCHEDULE = [ones(1,round(T/5))*1, ones(1,round(T/5))*2, ones(1,round(T/5))*3, ones(1,round(T/5))*4, ones(1,round(T/5))*5];
            K=SCHEDULE;
        elseif strcmp(k_type,'krnd')
            rng(0,'twister'); % fixing the random seed
            SCHEDULE = -log(rand(1,T)); SCHEDULE=round(SCHEDULE/max(SCHEDULE)*rank(x)/2+1);K=SCHEDULE;
        else, disp('error: unknown k-schedule');
        end
    end

    lamb1_range=[1./10.^(1:2:7), 0]; lamb2_range=[1./10.^(1:2:7), 0]; % if we regularise the classifier weights we use this line - works better
    %%%lamb1_range=[0]; lamb2_range=[0];  % if we don't regularise the classifier weights we use this line

    % variables to store results - we will only use v4 (rp or rs ensemble) or v2 (ensemble on the original data)
    err_v1  = zeros(reps,1); err_v2  = err_v1; err_v3  = err_v1; err_v4  = err_v1; err_v5  = err_v1; err_v6=err_v1; err_stump = err_v1;
    %>>>

    result2.err = zeros(reps,15); % error in the last round
    result2.lambda1 = zeros(reps,1);
    result2.k = zeros(reps,T); % one row for each repeat
    result2.testerr = zeros(reps,T); % one row for each repeat
    result2.trainerr = zeros(reps,T); % one row for each repeat
    result2.alpha=zeros(reps,T); result2.a=zeros(reps,ncls,T);result2.b=zeros(reps,ncls,T);result2.v=zeros(reps,ncls,T);

    result4.err = zeros(reps,15); % mlc 15 performance scores
    result4.lambda1 = zeros(reps,1);
    result4.k = zeros(reps,T); % one row for each repeat
    result4.testerr = zeros(reps,T); % one row for each repeat
    result4.trainerr = zeros(reps,T); % one row for each repeat
    result4.alpha=zeros(reps,T); result4.a=zeros(reps,ncls,T);result4.b=zeros(reps,ncls,T);result4.v=zeros(reps,ncls,T);
    %<<<

    for i=1:reps
        perm = perms(i,:); % use pre-defined permuatations to make results reproducible
        xt   = x(perm(1:nt),:);
        yt   = y(perm(1:nt),:);  % changed from y(perm(1:nt)) to support multi-class multi-label
        xs   = x(perm(nt+1:end),:);
        ys   = y(perm(nt+1:end),:); % changed from y(perm(1:nt)) to support multi-class multi-label
        [xt, xs] = standardise(xt, xs);

        cvstep = floor(length(yt)/5);    % 5-fold cross-validation to set the regularisation param of the classifier weights
        findex = randi([1 5], 1, length(yt));


        if METHODS(1) 
            method.name={'COCOA'}; % ensemble
            method.param=cell(length(method.name),1);
            %parameter set
            for m= 1:length(method.name)
                %parameter is controlled on SetmethodnameParameter.m See those file
                [method.param{m}]=feval(['Set',method.name{m},'Parameter'],[]);
            end
            method.base.name='ridge';
            method.base.param.lambda=0.01;
            method.th.type='SCut';
            method.th.param=0.5;

            %training
            [model,train_time]= MLC_train(xt,yt,method);
            %testing
            [conf,test_time]  = MLC_test(xt,yt,xs,model,method);
            %Thresholding
            [pred] = Thresholding(conf, method.th, ys);
            %Evalution
            [result1.err(i,:), metList] = Evaluation(ys,conf,pred,train_time,test_time);
            results.result1 = result1;
        end % if METHODS(1)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if METHODS(2) % this is gradient boosting ensemble on the original full data
            suffix=[{'v2'} '_' loss 'ep=' num2str(ep) 'nt=' num2str(nt)];
            %% (alpha is absorbed in w(1))
            %if i==1
            bestcv = 1e10;
            for lamb1 = lamb1_range
                fv = zeros(5,1);  % storing function values
                for f=1:5
                    xcvt = xt(findex~=f,:);
                    ycvt = yt(findex~=f,:); % changed from yt(findex~=f) to support multi-class multi-label
                    xcvs = xt(findex==f,:);
                    ycvs = yt(findex==f,:); % changed from yt(findex==f) to support multi-class multi-label
                    [w, obj]  = gboost_v2_m(fun_grad_theta, learner, xcvt, ycvt, T/10, ep, lamb1, reg_type,  q, Q,  xs,ys);

                    smax  = softmax(eval_ensemble(learner, w, ep, xcvs, ycvt));
                    yhat  = smax > 1/ncls;
                    [res, ~] = Evaluation(ycvs,softmax(eval_ensemble(learner, w, ep, xcvs, ycvt)),yhat,0,0);
                    fv(f) = res(13);
                end
                if mean(fv) < bestcv
                    best_lamb1_v2(i) = lamb1;
                    bestcv = mean(fv);
                end
            end
            %end
            % now training the model with selected lamb1
            [wg2, obj2, trainerr, testerr]  = gboost_v2_m(fun_grad_theta, learner, xt, yt, T, ep, best_lamb1_v2(i),reg_type, q, Q,    xs,ys);
            % evaluation; using the MLC toolbox
            smax = softmax(eval_ensemble(learner, wg2, ep, xs, yt));
            yhat = smax > 1/ncls;
            [res, measures] = Evaluation(ys,softmax(eval_ensemble(learner, w, ep, xs, yt)),yhat,0,0);

            result2.err(i,:) = res; % 15 performance measures given out by Evaluation
            result2.lambda1(i) =  best_lamb1_v2(i); % one number for each repeat
            %%  n/a   result2.k(i,:) = best_k; % one row for each repeat
            result2.obj(i,:) = obj2;
            result2.trainerr(i,:) = trainerr; % one row for each repeat
            result2.testerr(i,:)= testerr; % for plots            
            result2.a(i,:,:)=wg2(1,:,:); result2.b(i,:,:)=wg2(2,:,:);result2.v(i,:,:)=wg2(3,:,:);            
            results.result2=result2; % all results
            %<<<
        end % if METHODS(2)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if METHODS(3) % obsolete
            method.name={'ECC','rCC'};    % ensemble
            method.param=cell(length(method.name),1);
            %parameter set
            for m= 1:length(method.name)
                %parameter is controlled on SetmethodnameParameter.m See those file
                [method.param{m}]=feval(['Set',method.name{m},'Parameter'],[]);
            end
            % base classifier
            method.base.name='ridge';
            method.base.param.lambda=0.01;
            method.th.type='SCut';
            method.th.param=0.5;

            %training
            [model,train_time]= MLC_train(xt,yt,method);
            %testing
            method.name={'ECC'};
            [conf,test_time]  = MLC_test(xt,yt,xs,model,method);
            %Thresholding
            [pred] = Thresholding(conf, method.th, ys);
            %Evalution
            [result3.err(i,:), metList] = Evaluation(ys,conf,pred,train_time,test_time);
            results.result3 = result3;
        end % if METHODS(3)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if METHODS(4)  %% random projection or random subspace ensemble
            suffix=[rp_type '_' loss '_' k_type '_ep=' replace(num2str(ep),'.','') '_nt=' num2str(nt)];
            bestcv = 1e10;
            best_k_v4 = zeros(1,T);
            for lamb1 = lamb1_range
                fv = zeros(5,1);  % storing function values
                k  = zeros(5,T);
                for f=1:5
                    xcvt = xt(findex~=f,:);
                    ycvt = yt(findex~=f,:); % changed from yt(findex~=f) to support multi-class multi-label
                    xcvs = xt(findex==f,:);
                    ycvs = yt(findex==f,:); % changed from yt(findex==f) to support multi-class multi-label
                    [w, k(f,:), obj]  = rpgboost_v2_m(fun_grad_theta, learner, xcvt, ycvt, T/10, K, ep, lamb1, rp_type, reg_type, q, Q, xs,ys);

                    smax  = softmax(eval_ensemble(learner, w, ep, xcvs, ycvt));
                    yhat  = smax > 1/ncls;
                    [res, ~] = Evaluation(ycvs,softmax(eval_ensemble(learner, w, ep, xcvs, ycvt)),yhat,0,0);
                    % list of available performance measures 1 to 15
                    %    {'top1'}    {'top3'}      {'top5'}       {'dcg1'}
                    %    {'dcg3'}    {'dcg5'}      {'auc'}        {'exact'}
                    %    {'hamming'} {'macroF1'}   {'microF1'}    {'fscore'}
                    %    {'acc'}     {'trainT'}    {'testT'}
                    fv(f) = res(13); % hamming score; cross validate based on the hamming score
                end
                if mean(fv) < bestcv
                    best_lamb1_v4(i)  = lamb1;
                    [min_v, min_idx]  = min(fv);
                    best_k_v4         = k(min_idx,:);
                    bestcv            = mean(fv);
                end
            end
            [wg4, ~, obj4, trainerr, testerr]  = rpgboost_v2_m(fun_grad_theta, learner, xt, yt, T, best_k_v4, ep, best_lamb1_v4(i), rp_type, reg_type, q, Q, xs, ys);

            % evaluation; using the MLC toolbox
            smax = softmax(eval_ensemble(learner, wg4, ep, xs, yt));
            yhat = smax > 1/ncls;
            [res, measures] = Evaluation(ys,softmax(eval_ensemble(learner, w, ep, xs, yt)),yhat,0,0);

            result4.err(i,:) = res; % 15 performance measures given out by Evaluation
            result4.lambda1(i) = best_lamb1_v4(i); % one number for each repeat
            % n/a result4.lambda2(i) = best_lamb2_rb1; % one number for each repeat
            result4.k(i,:) = best_k_v4; % one row for each repeat
            result4.obj(i,:) = obj4;
            result4.trainerr(i,:) = trainerr; % one row for each repeat
            result4.testerr(i,:)= testerr; % for plots
            result4.a(i,:,:)=wg4(1,:,:); result4.b(i,:,:)=wg4(2,:,:);result4.v(i,:,:)=wg4(3,:,:);
            results.result4=result4; % all results
            %<<<
        end % if METHODS(4)


        if METHODS(5)
            %suffix={'competitors'};
            method.name={'fRAkEL','rCC'};
            method.param=cell(length(method.name),1);
            %parameter set
            for m= 1:length(method.name)
                %parameter is controlled on SetmethodnameParameter.m See those file
                [method.param{m}]=feval(['Set',method.name{m},'Parameter'],[]);
            end
            method.base.name='ridge';
            method.base.param.lambda=0.01;
            method.th.type='SCut';
            method.th.param=0.5;

            %training
            [model,train_time]= MLC_train(xt,yt,method);
            %testing
            [conf,test_time]  = MLC_test_new(xt,yt,xs,ys, model,method);
            %Thresholding
            [pred] = Thresholding(conf, method.th, ys);
            %Evalution
            [result5.err(i,:), metList] = Evaluation(ys,conf,pred,train_time,test_time);
            results.result5 = result5;
        end

        %fprintf('[repetition %d] errors gb1=%4.3f, gb2=%4.3f, rb1=%4.3f, rb2=%4.3f\n', i,...
        %    err_v1(i),err_v2(i), err_v3(i), err_v4(i));
        fprintf('[repetition %d]\n', i);
        
        save([cell2mat(data),'_result_acc_', cell2mat(suffix)],'results','ep');
    end

    fprintf('[%s] T=%d, e=%6.5f\n',...
        cell2mat(data), T, ep);
    fprintf('mean errors gb1=%4.3f, gb2=%4.3f, rb1=%4.3f, rb2=%4.3f\n',...
        mean(err_v1),mean(err_v2),mean(err_v3),mean(err_v4));
end

