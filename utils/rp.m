% creating a random matrix
function rm = rp(n, k, rp_type)
if strcmp(rp_type,'s')  % random coordinate projection
    RP=[eye(k) zeros(k,n-k)];
    ind=randperm(n);
    rm = sparse(RP(:,ind)');
elseif strcmp(rp_type,'g')
    rm = randn(n,k) / sqrt(n);
else
    disp('error: unknown RP type')
end
end
