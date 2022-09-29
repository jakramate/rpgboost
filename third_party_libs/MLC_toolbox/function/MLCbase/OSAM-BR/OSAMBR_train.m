function[model,time]=OSAMBR_train(X,Y,method)  % BR with UnderSampling
%% Input 
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%% Output
%model: weight matrix
% time: computation time

%% Initialization
[numN numF]=size(X);
[numNL,numL]=size(Y);
model=cell(2*numL,1);
ratio=method.param{1}.ratio; % ratio means D+ : D-= 1:ratio 
time=cell(numL+1,1);

%Learning model
fprintf('CALL: %s\n',method.base.name);
for label=1:numL
    % to observe CPUtime
    tmptime=cputime;
    % number of positive label
    numPos=sum(Y(:,label));
    % number of negative label
    numNeg=numN-numPos;
    % if negative label is large enough
    if  numNeg > numPos * ratio;
        %number of sample instances
        numSample= (numPos * ratio) -numPos;
        %indices of instances with positive/negative labels
        posInd=find(Y(:,label)==1);
        negInd=find(Y(:,label)==0);
        %Sample negative instances without duplication
        tmpInd=randi(numPos,numSample,1);
        %obtain real indices
        posInd= [posInd; posInd(tmpInd)];
        %reduce the size of problem
        tmpX=X([posInd;  negInd],:);
        tmpY=Y([posInd;  negInd],label);
        %keep index to perform MLKNN 
        model{label+numL}=posInd;
    else
        %else 
        posInd=find(Y(:,label)==1);
        tmpX=X;
        tmpY=Y(:,label);
        model{label+numL}=posInd;
    end
    time{end}=time{end}+cputime-tmptime;
    [model{label},method,time{label}]=feval([method.base.name,'_train'],tmpX,tmpY,method);
end


