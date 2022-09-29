function [one_hot_y] = onehot(y)

one_hot_y = (y == 1:length(unique(y)));