function [res] = eval_ensemble(h, ws, ep, xs, yt)
res  = ones(size(xs,1),1)* (sum(yt)/length(yt));
%res2 = ones(size(xs,1),1)*mean(yt);
%T   = size(ws,2);

if size(ws,1) == size(xs,2) + 4
    % for algorithm V1
%     for j=1:T
%         res = res + ep * ws(1,j) * h(xs, ws(2:end,j));
%     end

    ht  = h(xs, ws(2:end,:));
    
    res = res + sum(ep * bsxfun(@times, ws(1,:), ht), 2);
else
    % for algorithm V2
%     for j=1:T        
%         res = res + ep * h(xs, ws(:,j));
%     end
    ht  = h(xs, ws);
    res = res + sum(ep * ht, 2);    
end
end