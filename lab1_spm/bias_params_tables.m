function [tissue_wise, subject_wise] = bias_params_tables(dice_results)
% Computes the mean and std tables from the dice results
tissue_wise = zeros(4, 8);
for i=1:3
    for j=1:4
        tissue_wise(j,2*(i-1)+1) = mean(dice_results(j,:,i)); 
        tissue_wise(j,2*(i-1)+2) = std(dice_results(j,:,i));
    end
end
for j=1:4
    tissue_wise(j,7) = mean(reshape(dice_results(j,:,:),1,[]));
    tissue_wise(j,8) = std(reshape(dice_results(j,:,:),1,[]));
end

subject_wise = zeros(4, 10);
for i=1:5
    for j=1:4
        subject_wise(j,2*(i-1)+1) = mean(dice_results(j,i,:));
        subject_wise(j,2*(i-1)+2) = std(dice_results(j,i,:));
    end
end
end

