%% MISA LAB 1, Joaquin Oscar Seia - Kaouther Mouheb 
clear all; 
close all; 
clc;
% Modify paths to your device:
addpath('D:\Master\Girona\Segmentation\labs\lab1\spm12\spm12');
base_data_path = 'D:\Master\Girona\Segmentation\labs\lab1\';

% %% Parameters exploration plots:
% % Regularization on T1 images
% dice_results_reg_t1 = zeros(4, 5, 3);
% regs = [0 0.0001 0.01 1];
% settings = struct();
% settings.biasreg = 0;
% settings.biasfwhm = 60;
% settings.write = [0 0];
% for j=1:length(regs)
%     for i=1:5
%         settings.biasreg = regs(j);
%         % Running everything for one case:
%         res_seg = segment_brain_tissues(fullfile(base_data_path, num2str(i), '/T1.nii,1'), settings);
%         labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
%         dice_results_reg_t1(j, i, :) = dice(res_seg , labels);
%     end
% end
% save('dice_reg_T1.mat','dice_results_reg_t1');
% 
% %%
% table1_t1 = zeros(4, 8);
% for i=1:3
%     table1_t1(1,2*(i-1)+1) = mean(dice_results_reg_t1(1,:,i)); table1_t1(1,2*(i-1)+2) = std(dice_results_reg_t1(1,:,i));
%     table1_t1(2,2*(i-1)+1) = mean(dice_results_reg_t1(2,:,i)); table1_t1(2,2*(i-1)+2) = std(dice_results_reg_t1(2,:,i));
%     table1_t1(3,2*(i-1)+1) = mean(dice_results_reg_t1(3,:,i)); table1_t1(3,2*(i-1)+2) = std(dice_results_reg_t1(3,:,i));
%     table1_t1(4,2*(i-1)+1) = mean(dice_results_reg_t1(4,:,i)); table1_t1(4,2*(i-1)+2) = std(dice_results_reg_t1(4,:,i));
% end
% table1_t1(1,7) = mean(reshape(dice_results_reg_t1(1,:,:),1,[])); table1_t1(1,8) = std(reshape(dice_results_reg_t1(1,:,:),1,[]));
% table1_t1(2,7) = mean(reshape(dice_results_reg_t1(2,:,:),1,[])); table1_t1(2,8) = std(reshape(dice_results_reg_t1(2,:,:),1,[]));
% table1_t1(3,7) = mean(reshape(dice_results_reg_t1(3,:,:),1,[])); table1_t1(3,8) = std(reshape(dice_results_reg_t1(3,:,:),1,[]));
% table1_t1(4,7) = mean(reshape(dice_results_reg_t1(4,:,:),1,[])); table1_t1(4,8) = std(reshape(dice_results_reg_t1(4,:,:),1,[]));
% table1_t1
% 
% table2_t1 = zeros(4, 10);
% for i=1:5
%     table2_t1(1,2*(i-1)+1) = mean(dice_results_reg_t1(1,i,:)); table2_t1(1,2*(i-1)+2) = std(dice_results_reg_t1(1,i,:));
%     table2_t1(2,2*(i-1)+1) = mean(dice_results_reg_t1(2,i,:)); table2_t1(2,2*(i-1)+2) = std(dice_results_reg_t1(2,i,:));
%     table2_t1(3,2*(i-1)+1) = mean(dice_results_reg_t1(3,i,:)); table2_t1(3,2*(i-1)+2) = std(dice_results_reg_t1(3,i,:));
%     table2_t1(4,2*(i-1)+1) = mean(dice_results_reg_t1(4,i,:)); table2_t1(4,2*(i-1)+2) = std(dice_results_reg_t1(4,i,:));
% end
% table2_t1
% 
% %%
% % FWHM on T1 images
% dice_results_fwhm_t1 = zeros(5, 5, 3);
% fwhms = [40 60 80 100 120];
% settings = struct();
% settings.biasreg = 0.01;
% settings.biasfwhm = 60;
% settings.write = [0 0];
% for j=1:length(fwhms)
%     for i=1:5
%         settings.biasfwhm = fwhms(j);
%         % Running everything for one case:
%         res_seg = segment_brain_tissues(fullfile(base_data_path, num2str(i), '/T1.nii,1'), settings);
%         labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
%         dice_results_fwhm_t1(j, i, :) = dice(res_seg , labels);
%     end
% end
% save('dice_fwhm_T1.mat','dice_results_fwhm_t1');
% 
% %%
% table3_t1= zeros(5, 8);
% for i=1:3
%     table3_t1(1,2*(i-1)+1) = mean(dice_results_fwhm_t1(1,:,i)); table3_t1(1,2*(i-1)+2) = std(dice_results_fwhm_t1(1,:,i));
%     table3_t1(2,2*(i-1)+1) = mean(dice_results_fwhm_t1(2,:,i)); table3_t1(2,2*(i-1)+2) = std(dice_results_fwhm_t1(2,:,i));
%     table3_t1(3,2*(i-1)+1) = mean(dice_results_fwhm_t1(3,:,i)); table3_t1(3,2*(i-1)+2) = std(dice_results_fwhm_t1(3,:,i));
%     table3_t1(4,2*(i-1)+1) = mean(dice_results_fwhm_t1(4,:,i)); table3_t1(4,2*(i-1)+2) = std(dice_results_fwhm_t1(4,:,i));
%     table3_t1(5,2*(i-1)+1) = mean(dice_results_fwhm_t1(5,:,i)); table3_t1(5,2*(i-1)+2) = std(dice_results_fwhm_t1(5,:,i));
% end
% table3_t1(1,7) = mean(reshape(dice_results_fwhm_t1(1,:,:),1,[])); table3_t1(1,8) = std(reshape(dice_results_fwhm_t1(1,:,:),1,[]));
% table3_t1(2,7) = mean(reshape(dice_results_fwhm_t1(2,:,:),1,[])); table3_t1(2,8) = std(reshape(dice_results_fwhm_t1(2,:,:),1,[]));
% table3_t1(3,7) = mean(reshape(dice_results_fwhm_t1(3,:,:),1,[])); table3_t1(3,8) = std(reshape(dice_results_fwhm_t1(3,:,:),1,[]));
% table3_t1(4,7) = mean(reshape(dice_results_fwhm_t1(4,:,:),1,[])); table3_t1(4,8) = std(reshape(dice_results_fwhm_t1(4,:,:),1,[]));
% table3_t1(5,7) = mean(reshape(dice_results_fwhm_t1(5,:,:),1,[])); table3_t1(5,8) = std(reshape(dice_results_fwhm_t1(5,:,:),1,[]));
% table3_t1
% 
% table4_t1 = zeros(5, 10);
% for i=1:5
%     table4_t1(1,2*(i-1)+1) = mean(dice_results_fwhm_t1(1,i,:)); table4_t1(1,2*(i-1)+2) = std(dice_results_fwhm_t1(1,i,:));
%     table4_t1(2,2*(i-1)+1) = mean(dice_results_fwhm_t1(2,i,:)); table4_t1(2,2*(i-1)+2) = std(dice_results_fwhm_t1(2,i,:));
%     table4_t1(3,2*(i-1)+1) = mean(dice_results_fwhm_t1(3,i,:)); table4_t1(3,2*(i-1)+2) = std(dice_results_fwhm_t1(3,i,:));
%     table4_t1(4,2*(i-1)+1) = mean(dice_results_fwhm_t1(4,i,:)); table4_t1(4,2*(i-1)+2) = std(dice_results_fwhm_t1(4,i,:));
%     table4_t1(5,2*(i-1)+1) = mean(dice_results_fwhm_t1(5,i,:)); table4_t1(5,2*(i-1)+2) = std(dice_results_fwhm_t1(5,i,:));
% end
% table4_t1
% 
% %%
% % Regularization on T2 images
% dice_results_reg_t2 = zeros(5, 5, 3);
% regs = [0 0.0001 0.001 0.01 1];
% settings = struct();
% settings.biasreg = 0;
% settings.biasfwhm = 60;
% settings.write = [0 0];
% for j=1:length(regs)
%     for i=1:5
%         settings.biasreg = regs(j);
%         % Running everything for one case:
%         res_seg = segment_brain_tissues(fullfile(base_data_path, num2str(i), '/T2_FLAIR.nii,1'), settings);
%         labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
%         dice_results_reg_t2(j, i, :) = dice(res_seg , labels);
%     end
% end
% save('dice_reg_T2.mat','dice_results_reg_t2');
% 
% table1_t2 = zeros(4, 8);
% for i=1:3
%     table1_t2(1,2*(i-1)+1) = mean(dice_results_reg_t2(1,:,i)); table1_t2(1,2*(i-1)+2) = std(dice_results_reg_t2(1,:,i));
%     table1_t2(2,2*(i-1)+1) = mean(dice_results_reg_t2(2,:,i)); table1_t2(2,2*(i-1)+2) = std(dice_results_reg_t2(2,:,i));
%     table1_t2(3,2*(i-1)+1) = mean(dice_results_reg_t2(3,:,i)); table1_t2(3,2*(i-1)+2) = std(dice_results_reg_t2(3,:,i));
%     table1_t2(4,2*(i-1)+1) = mean(dice_results_reg_t2(4,:,i)); table1_t2(4,2*(i-1)+2) = std(dice_results_reg_t2(4,:,i));
% end
% table1_t2(1,7) = mean(reshape(dice_results_reg_t2(1,:,:),1,[])); table1_t2(1,8) = std(reshape(dice_results_reg_t2(1,:,:),1,[]));
% table1_t2(2,7) = mean(reshape(dice_results_reg_t2(2,:,:),1,[])); table1_t2(2,8) = std(reshape(dice_results_reg_t2(2,:,:),1,[]));
% table1_t2(3,7) = mean(reshape(dice_results_reg_t2(3,:,:),1,[])); table1_t2(3,8) = std(reshape(dice_results_reg_t2(3,:,:),1,[]));
% table1_t2(4,7) = mean(reshape(dice_results_reg_t2(4,:,:),1,[])); table1_t2(4,8) = std(reshape(dice_results_reg_t2(4,:,:),1,[]));
% table1_t2
% 
% table2_t2 = zeros(4, 10);
% for i=1:5
%     table2_t2(1,2*(i-1)+1) = mean(dice_results_reg_t2(1,i,:)); table2_t2(1,2*(i-1)+2) = std(dice_results_reg_t2(1,i,:));
%     table2_t2(2,2*(i-1)+1) = mean(dice_results_reg_t2(2,i,:)); table2_t2(2,2*(i-1)+2) = std(dice_results_reg_t2(2,i,:));
%     table2_t2(3,2*(i-1)+1) = mean(dice_results_reg_t2(3,i,:)); table2_t2(3,2*(i-1)+2) = std(dice_results_reg_t2(3,i,:));
%     table2_t2(4,2*(i-1)+1) = mean(dice_results_reg_t2(4,i,:)); table2_t2(4,2*(i-1)+2) = std(dice_results_reg_t2(4,i,:));
% end
% table2_t2
% 
% % FWHM on T1 images
% dice_results_fwhm_t2 = zeros(5, 5, 3);
% fwhms = [40 60 80 100 120];
% settings = struct();
% settings.biasreg = 0.01;
% settings.biasfwhm = 60;
% settings.write = [0 0];
% for j=1:length(fwhms)
%     for i=1:5
%         settings.biasfwhm = fwhms(j);
%         % Running everything for one case:
%         res_seg = segment_brain_tissues(fullfile(base_data_path, num2str(i), '/T2_FLAIR.nii,1'), settings);
%         labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
%         dice_results_fwhm_t2(j, i, :) = dice(res_seg , labels);
%     end
% end
% save('dice_fwhm_T2.mat','dice_results_fwhm_t2');
% 
% table3_t2= zeros(4, 8);
% for i=1:3
%     table3_t2(1,2*(i-1)+1) = mean(dice_results_fwhm_t2(1,:,i)); table3_t2(1,2*(i-1)+2) = std(dice_results_fwhm_t2(1,:,i));
%     table3_t2(2,2*(i-1)+1) = mean(dice_results_fwhm_t2(2,:,i)); table3_t2(2,2*(i-1)+2) = std(dice_results_fwhm_t2(2,:,i));
%     table3_t2(3,2*(i-1)+1) = mean(dice_results_fwhm_t2(3,:,i)); table3_t2(3,2*(i-1)+2) = std(dice_results_fwhm_t2(3,:,i));
%     table3_t2(4,2*(i-1)+1) = mean(dice_results_fwhm_t2(4,:,i)); table3_t2(4,2*(i-1)+2) = std(dice_results_fwhm_t2(4,:,i));
%     table3_t2(5,2*(i-1)+1) = mean(dice_results_fwhm_t2(5,:,i)); table3_t2(5,2*(i-1)+2) = std(dice_results_fwhm_t2(5,:,i));
% end
% table3_t2(1,7) = mean(reshape(dice_results_fwhm_t2(1,:,:),1,[])); table3_t2(1,8) = std(reshape(dice_results_fwhm_t2(1,:,:),1,[]));
% table3_t2(2,7) = mean(reshape(dice_results_fwhm_t2(2,:,:),1,[])); table3_t2(2,8) = std(reshape(dice_results_fwhm_t2(2,:,:),1,[]));
% table3_t2(3,7) = mean(reshape(dice_results_fwhm_t2(3,:,:),1,[])); table3_t2(3,8) = std(reshape(dice_results_fwhm_t2(3,:,:),1,[]));
% table3_t2(4,7) = mean(reshape(dice_results_fwhm_t2(4,:,:),1,[])); table3_t2(4,8) = std(reshape(dice_results_fwhm_t2(4,:,:),1,[]));
% table3_t2(5,7) = mean(reshape(dice_results_fwhm_t2(5,:,:),1,[])); table3_t2(5,8) = std(reshape(dice_results_fwhm_t2(5,:,:),1,[]));
% table3_t2
% 
% table4_t2 = zeros(4, 10);
% for i=1:5
%     table4_t2(1,2*(i-1)+1) = mean(dice_results_fwhm_t2(1,i,:)); table4_t2(1,2*(i-1)+2) = std(dice_results_fwhm_t2(1,i,:));
%     table4_t2(2,2*(i-1)+1) = mean(dice_results_fwhm_t2(2,i,:)); table4_t2(2,2*(i-1)+2) = std(dice_results_fwhm_t2(2,i,:));
%     table4_t2(3,2*(i-1)+1) = mean(dice_results_fwhm_t2(3,i,:)); table4_t2(3,2*(i-1)+2) = std(dice_results_fwhm_t2(3,i,:));
%     table4_t2(4,2*(i-1)+1) = mean(dice_results_fwhm_t2(4,i,:)); table4_t2(4,2*(i-1)+2) = std(dice_results_fwhm_t2(4,i,:));
%     table4_t2(5,2*(i-1)+1) = mean(dice_results_fwhm_t2(5,i,:)); table4_t2(5,2*(i-1)+2) = std(dice_results_fwhm_t2(5,i,:));
% end
% table4_t2

%% Run segmentations and metrics

settings = struct();
settings.write = [0 0];

% 1 - Using only T1
dice_results = zeros(5, 3);
for i=1:5
    % Running everything for one case:
    settings.biasreg = 0.0001;
    settings.biasfwhm = 40;
    
    structural_fn = fullfile(base_data_path, num2str(i), '/T1.nii,1');
    vol = niftiread(fullfile(base_data_path, num2str(i), '/T1.nii')); 
    res_seg = segment_brain_tissues(structural_fn, settings);
    labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
    dice_results(i, :) = dice(res_seg , labels);

    % plot example slice
    slice_i = 24;
    figure;
    t = tiledlayout(3,1);
    nexttile
    imshow(uint8(vol(:, :, slice_i)));
    title("T1 Slice");
    
    nexttile
    imagesc(labels(:, :, slice_i));
    axis square off
    title("Ground Truth");
    
    nexttile
    imagesc(res_seg(:, :, slice_i));
    axis square off
    title("Segmentation");
    
    fig_fn = strcat("figs/results/s", num2str(i),"_t1_ss.png");
    exportgraphics(t,fig_fn)

    % Save results with correct spacing
    template_fn = [structural_fn];
    template_spm = spm_vol(template_fn);
    new_nii = spm_create_vol(template_spm);
    new_nii.fname = strcat('result_s',num2str(i) ,'_t1.nii');
    spm_write_vol(new_nii, res_seg);
end
save('dice_t1.mat','dice_results');

% Get statistics
figure
csf = dice_results(:, 1);
gm = dice_results(:, 2);
wm = dice_results(:, 3);
boxplot([csf, gm, wm],'Labels',{'CSF', 'GM','WM'})
title('Dice Scores per tissue type - T1')


% 1 - Using only T2
dice_results = zeros(5, 3);
for i=1:5
    % Running everything for one case:
    settings.biasreg = 1;
    settings.biasfwhm = 120;
    
    structural_fn = fullfile(base_data_path, num2str(i), '/T2_FLAIR.nii,1');
    vol = niftiread(fullfile(base_data_path, num2str(i), '/T2_FLAIR.nii')); 
    res_seg = segment_brain_tissues(structural_fn, settings);
    labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
    dice_results(i, :) = dice(res_seg , labels);

    % plot example slice
    slice_i = 24;
    figure;
    t = tiledlayout(3,1);
    nexttile
    imshow(uint8(vol(:, :, slice_i)));
    title("T2 Slice");
    
    nexttile
    imagesc(labels(:, :, slice_i));
    axis square off
    title("Ground Truth");
    
    nexttile
    imagesc(res_seg(:, :, slice_i));
    axis square off
    title("Segmentation");
    
    fig_fn = strcat("figs/results/s", num2str(i),"_t2_ss.png");
    exportgraphics(t,fig_fn)

    % Save results with correct spacing
    template_fn = [structural_fn];
    template_spm = spm_vol(template_fn);
    new_nii = spm_create_vol(template_spm);
    new_nii.fname = strcat('result_s',num2str(i) ,'_t2.nii');
    spm_write_vol(new_nii, res_seg);
end
save('dice_t2.mat','dice_results');

% Get statistics
figure
csf = dice_results(:, 1);
gm = dice_results(:, 2);
wm = dice_results(:, 3);
boxplot([csf, gm, wm],'Labels',{'CSF', 'GM','WM'})
title('Dice Scores per tissue type - T2')

