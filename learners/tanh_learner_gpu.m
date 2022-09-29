function [h] = tanh_learner_gpu(x, w)
    %h = w(1)*tanh(x*w(4:end) + w(3)) + w(2);
    %h = bsxfun(@plus, x*w(4:end,:), w(3,:));
    gx = gpuArray(x);
    gw = gpuArray(w);
    h = gx*gw(4:end,:) + repmat(gw(3,:),size(gx,1),1);
    h = tanh(h) .* repmat(gw(1,:),size(gx,1),1);
    h = h + repmat(gw(2,:),size(gx,1),1);        
    h = gather(h);
end