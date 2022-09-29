datasets = {'mice','musk'};
for nt= [200, 100, 50]

    METHODS=[0 0 0 1]; % selection of methods we want to run

    %ResultFolder='Results_LONG/';
    for loss={'exp','log'}
        for rp_type={'g','s'}
            for k_type={'krnd'}%'k1','k12345',}
        		for ep=[0.1, 1]
        			classification_exper_par;
        		end
            end
        end
    end

    METHODS=[0 1 0 0]; % selection of methods we want to run
    %ResultFolder='Results_RASE/';
    for loss={'exp','log'}
        %  for rp_type={'g','s'}
        %  for k_type={'krnd','k12345','k1','klearn'}
        for ep=[0.1, 1]
            classification_exper_par;
        end
        % end
        % end
    end

end % nt
%%%%%%%%%%%%%%%%%%%%%

