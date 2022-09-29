%%% Parameter Analysis with one parameter

%% Initial Setting 
addAllpath; rng('default');

%% Datasets to use 
dataNames={'enron','yeast'};
numCV=10; numTrial=1; % less than 10


%% Select methods to use
%CAUTION: functionNames is for the plot
functionName={'RAkEL+LP'};
method.name={'RAkEL','LP'};
method=SetALLParams(method);
method.base.name='linear_svm';
method.base.param.svmparam='-s 2 -q';
method.base.param.lambda=10;
method.th.type='SCut';
method.th.param=0.5;

% changeMethod
change.name='RAkEL';
change.param ='numk';
change.value=[3 5 7 9];
Result=cell(length(dataNames),1);

for countData=1:length(dataNames)
    dataname=dataNames{countData};
    Result{countData}=cell(length(change.value),1);
    for countParam=1:length(change.value)
        [method]=paramChange(method,change,countParam);
        [res]=conductExpriments(method,numTrial,numCV,dataname);
        Result{countData}{countParam}=res;
    end
end


criteria={'top1','auc','exact','hamming','macroF1','microF1'};
fileNames=criteria;
% Visualization of results
for countData=1:length(dataNames)
    dataname=dataNames{countData};
    for i=1:length(criteria)
        criterion=criteria{i};
        getFigure(Result{countData},dataname,change,criterion,['Param',dataname,'-',criterion,'.png']);
    end
end

