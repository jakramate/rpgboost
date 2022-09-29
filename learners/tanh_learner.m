function [h] = tanh_learner(x, w)
    %h = w(1)*tanh(x*w(4:end) + w(3)) + w(2);
    h = bsxfun(@plus, x*w(4:end,:), w(3,:));
    h = bsxfun(@times, tanh(h), w(1,:));
    h = bsxfun(@plus, h, w(2,:));        
end