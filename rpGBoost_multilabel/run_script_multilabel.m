clear
datasets={'birds_new', 'scene_new', 'yeast_new','emotions_new', 'flags_new'};

%[b b b b b] : selection of methods we want to run. Use the following:


METHODS=[0 1 0 1 0]; 
for loss={'pal'} 
    for rp_type={'g','s'}  % 'g' = random projection; 's' = random feature subspace
        for k_type={'krnd'}    % this samples the target dimensions randomly
            for ep=[0.1, 1]  % this is a learning rate parameter
                classification_exper_par_multilabel;  % function that manages the algorithm calls
            end
        end
    end
end

fprintf('COCOA   -- AUC %.3f -- HAM %.3f\n', mean(results.result1.err(:,[5,9])))
fprintf('ECC     -- AUC %.3f -- HAM %.3f\n', mean(results.result3.err(:,[5,9])))
fprintf('fRAkEL  -- AUC %.3f -- HAM %.3f\n', mean(results.result5.err(:,[5,9])))

fprintf('GBOOST  -- AUC %.3f -- HAM %.3f\n', mean(results.result2.err(:,[5,9])))
fprintf('RPBOOST -- AUC %.3f -- HAM %.3f\n', mean(results.result4.err(:,[5,9])))