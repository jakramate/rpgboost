% make sure that an observation is a row vector.

function [X, Xt] = standardise(x, xt)

DPNT   = size(x, 1);
TPNT   = size(xt,1);

xa = x;

% standarding train/test sets, using only the training examples
offset = mean(xa);
var    = std(xa);
var(var==0) = var(var==0) + 1;
scale  = 1./var;

X      = x - repmat(offset, DPNT, 1);
X      = X .* repmat(scale , DPNT, 1);
Xt     = xt -  repmat(offset, TPNT, 1);
Xt     = Xt .* repmat(scale , TPNT, 1);



