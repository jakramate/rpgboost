% compute the softmax

function ret = softmax(z)
ez = exp(z);
denom = repmat(sum(ez,2),1,size(ez,2));
denom(denom==0) = eps;
ret = ez ./ repmat(sum(ez,2),1,size(ez,2));    % size(ez,1) represents CLS
ret(isnan(ret)) = 1;