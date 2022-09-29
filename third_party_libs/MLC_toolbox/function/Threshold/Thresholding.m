function [pred]= Thresholding(conf,method,Y)
%% Input
%conf: confidence values (NtxL)
%method: paramters.
%% Output
%pred: binary prediction (NtxL)
%% Option
%method.type='Rcut': Rank-based Cut (not implemented yet)
    %method.param: controls RCut
%method.type='Pcut': Proportion-based Cut (not implemented yet)
    %method.param: controls Pcut
%method.type='Scut': Score-based Cut (available)
    %method.param: threshold value for Scut (default,0.5);
%% Reference (APA style from google scholar)
%Tang, L., Rajan, S., & Narayanan, V. K. (2009, April). Large scale multi-label classification via metalabeler. 
%In Proceedings of the 18th international conference on World wide web (pp. 211-220). ACM.
%Fan, R. E., & Lin, C. J. (2007). A study on threshold selection for multi-label classification. Department of Computer Science, National Taiwan University, 1-23.

%error check
if ~isfield(method,'type')
    warning('thresholding strategy is not set\n we select Scut')
    method.type='Scut';
end

%% Initialization 
[numNt,numL]=size(conf);
%thresholding
switch method.type
    case {'Scut','SCut','scut'}
    %if param is not set 
    if ~isfield(method,'param')
        warning('epsilon for the threshold is not set\n we set 0.5');
        method.param=0.5;
    end
    %thresholding
    pred=conf;
    pred(pred>method.param)=1;
    pred(pred<=method.param)=0;
        
    case {'Rcut','RCut','rcut'}
        if ~isfield(method,'param')
            warning('number of rank is not set we use label cardinality')
            tmp=sum(Y,2);
            LC=ceil(mean(tmp));
            method.param=LC;
        end
        [~,ranks]=sort(conf,2,'descend');
        [numNt,numL]=size(conf);
        pred=zeros(numNt,numL);
        for i=1:numNt
            pred(i,ranks(i,1:method.param))=1;
        end
    case {'PCut','Pcut','pcut'}
        if ~isfield(method,'param')
            warning('number of rank is not set we use label cardinality')
            tmp=sum(Y) ./ size(Y,1);
            method.param= ceil(tmp)*numNt;
        end
        if length(method.param)~=numL
            error('proportion score must be design for each label')
        end
        [~,ranks]=sort(conf,'descend');
        pred=zeros(numNt,numL);
        for i=1:numL
           pred(ranks(1:method.param(i),i),i)=1;
        end
    otherwise
    % the other methods are not implemented yet. someone help!
    error('%s is not surpported',method.type)
end

% Prevent from null prediction for ridge regression and knn (improve their performance)
if isempty(find(sum(Y,2)==0,1))
    idzero = find(sum(pred,2)==0);
    [valmax,idmax] = max(conf(idzero,:),[],2);
    id = find(valmax ~= 0);
    pred(sub2ind(size(pred),idzero(id),idmax(id))) = 1;
end