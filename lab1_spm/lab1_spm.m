%% MISA LAB 1, Joaquin Oscar Seia - Kaouther Mouheb 
clear all; 
close all; 
clc;
% Modify paths to your device:
addpath('/home/jseia/Desktop/MATLAB/spm12')
base_data_path = '/home/jseia/Desktop/MAIA/Clases/spain/misa/misa_lab/lab1_spm/data/P2_data/';

% % Run segmentations and metrics
% dice_results = zeros(5, 3);
% for i=1:5
%     % Running everything for one case:
%     res_seg = segment_brain_tissues(fullfile(base_data_path, num2str(i), '/T1.nii,1'));
%     labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
%     dice_results(i, :) = dice(res_seg , labels);
% end
% save('dice.mat','dice_results');
% 
% % Get statistics
% figure
% csf = dice_results(:, 1);
% gm = dice_results(:, 2);
% wm = dice_results(:, 3);
% boxplot([csf, gm, wm],'Labels',{'CSF', 'GM','WM'})
% title('Dice Scores per tissue type')

%% Parameters exploration plots:
% Regularization on T1 images
dice_results_reg_t1 = zeros(4, 5, 3);
regs = [0 0.0001 0.01 1];
settings = struct();
settings.biasreg = 0;
settings.biasfwhm = 60;
settings.write = [0 0];
for j=1:length(regs)
    for i=1:5
        settings.biasreg = regs(j);
        % Running everything for one case:
        res_seg = segment_brain_tissues(fullfile(base_data_path, num2str(i), '/T1.nii,1'), settings);
        labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
        dice_results_reg_t1(j, i, :) = dice(res_seg , labels);
    end
end
save('dice_reg_T1.mat','dice_results_reg_t1');

%%
table1_t1 = zeros(4, 8);
for i=1:3
    table1_t1(1,2*(i-1)+1) = mean(dice_results_reg_t1(1,:,i)); table1_t1(1,2*(i-1)+2) = std(dice_results_reg_t1(1,:,i));
    table1_t1(2,2*(i-1)+1) = mean(dice_results_reg_t1(2,:,i)); table1_t1(2,2*(i-1)+2) = std(dice_results_reg_t1(2,:,i));
    table1_t1(3,2*(i-1)+1) = mean(dice_results_reg_t1(3,:,i)); table1_t1(3,2*(i-1)+2) = std(dice_results_reg_t1(3,:,i));
    table1_t1(4,2*(i-1)+1) = mean(dice_results_reg_t1(4,:,i)); table1_t1(4,2*(i-1)+2) = std(dice_results_reg_t1(4,:,i));
end
table1_t1(1,7) = mean(reshape(dice_results_reg_t1(1,:,:),1,[])); table1_t1(1,8) = std(reshape(dice_results_reg_t1(1,:,:),1,[]));
table1_t1(2,7) = mean(reshape(dice_results_reg_t1(2,:,:),1,[])); table1_t1(2,8) = std(reshape(dice_results_reg_t1(2,:,:),1,[]));
table1_t1(3,7) = mean(reshape(dice_results_reg_t1(3,:,:),1,[])); table1_t1(3,8) = std(reshape(dice_results_reg_t1(3,:,:),1,[]));
table1_t1(4,7) = mean(reshape(dice_results_reg_t1(4,:,:),1,[])); table1_t1(4,8) = std(reshape(dice_results_reg_t1(4,:,:),1,[]));
table1_t1

table2_t1 = zeros(4, 10);
for i=1:5
    table2_t1(1,2*(i-1)+1) = mean(dice_results_reg_t1(1,i,:)); table2_t1(1,2*(i-1)+2) = std(dice_results_reg_t1(1,i,:));
    table2_t1(2,2*(i-1)+1) = mean(dice_results_reg_t1(2,i,:)); table2_t1(2,2*(i-1)+2) = std(dice_results_reg_t1(2,i,:));
    table2_t1(3,2*(i-1)+1) = mean(dice_results_reg_t1(3,i,:)); table2_t1(3,2*(i-1)+2) = std(dice_results_reg_t1(3,i,:));
    table2_t1(4,2*(i-1)+1) = mean(dice_results_reg_t1(4,i,:)); table2_t1(4,2*(i-1)+2) = std(dice_results_reg_t1(4,i,:));
end
table2_t1

%%
% FWHM on T1 images
dice_results_fwhm_t1 = zeros(5, 5, 3);
fwhms = [40 60 80 100 120];
settings = struct();
settings.biasreg = 0.01;
settings.biasfwhm = 60;
settings.write = [0 0];
for j=1:length(fwhms)
    for i=1:5
        settings.biasfwhm = fwhms(j);
        % Running everything for one case:
        res_seg = segment_brain_tissues(fullfile(base_data_path, num2str(i), '/T1.nii,1'), settings);
        labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
        dice_results_fwhm_t1(j, i, :) = dice(res_seg , labels);
    end
end
save('dice_fwhm_T1.mat','dice_results_fwhm_t1');

%%
table3_t1= zeros(5, 8);
for i=1:3
    table3_t1(1,2*(i-1)+1) = mean(dice_results_fwhm_t1(1,:,i)); table3_t1(1,2*(i-1)+2) = std(dice_results_fwhm_t1(1,:,i));
    table3_t1(2,2*(i-1)+1) = mean(dice_results_fwhm_t1(2,:,i)); table3_t1(2,2*(i-1)+2) = std(dice_results_fwhm_t1(2,:,i));
    table3_t1(3,2*(i-1)+1) = mean(dice_results_fwhm_t1(3,:,i)); table3_t1(3,2*(i-1)+2) = std(dice_results_fwhm_t1(3,:,i));
    table3_t1(4,2*(i-1)+1) = mean(dice_results_fwhm_t1(4,:,i)); table3_t1(4,2*(i-1)+2) = std(dice_results_fwhm_t1(4,:,i));
    table3_t1(5,2*(i-1)+1) = mean(dice_results_fwhm_t1(5,:,i)); table3_t1(5,2*(i-1)+2) = std(dice_results_fwhm_t1(5,:,i));
end
table3_t1(1,7) = mean(reshape(dice_results_fwhm_t1(1,:,:),1,[])); table3_t1(1,8) = std(reshape(dice_results_fwhm_t1(1,:,:),1,[]));
table3_t1(2,7) = mean(reshape(dice_results_fwhm_t1(2,:,:),1,[])); table3_t1(2,8) = std(reshape(dice_results_fwhm_t1(2,:,:),1,[]));
table3_t1(3,7) = mean(reshape(dice_results_fwhm_t1(3,:,:),1,[])); table3_t1(3,8) = std(reshape(dice_results_fwhm_t1(3,:,:),1,[]));
table3_t1(4,7) = mean(reshape(dice_results_fwhm_t1(4,:,:),1,[])); table3_t1(4,8) = std(reshape(dice_results_fwhm_t1(4,:,:),1,[]));
table3_t1(5,7) = mean(reshape(dice_results_fwhm_t1(5,:,:),1,[])); table3_t1(5,8) = std(reshape(dice_results_fwhm_t1(5,:,:),1,[]));
table3_t1

table4_t1 = zeros(5, 10);
for i=1:5
    table4_t1(1,2*(i-1)+1) = mean(dice_results_fwhm_t1(1,i,:)); table4_t1(1,2*(i-1)+2) = std(dice_results_fwhm_t1(1,i,:));
    table4_t1(2,2*(i-1)+1) = mean(dice_results_fwhm_t1(2,i,:)); table4_t1(2,2*(i-1)+2) = std(dice_results_fwhm_t1(2,i,:));
    table4_t1(3,2*(i-1)+1) = mean(dice_results_fwhm_t1(3,i,:)); table4_t1(3,2*(i-1)+2) = std(dice_results_fwhm_t1(3,i,:));
    table4_t1(4,2*(i-1)+1) = mean(dice_results_fwhm_t1(4,i,:)); table4_t1(4,2*(i-1)+2) = std(dice_results_fwhm_t1(4,i,:));
    table4_t1(5,2*(i-1)+1) = mean(dice_results_fwhm_t1(5,i,:)); table4_t1(5,2*(i-1)+2) = std(dice_results_fwhm_t1(5,i,:));
end
table4_t1

%%
% Regularization on T2 images
dice_results_reg_t2 = zeros(5, 5, 3);
regs = [0 0.0001 0.001 0.01 1];
settings = struct();
settings.biasreg = 0;
settings.biasfwhm = 60;
settings.write = [0 0];
for j=1:length(regs)
    for i=1:5
        settings.biasreg = regs(j);
        % Running everything for one case:
        res_seg = segment_brain_tissues(fullfile(base_data_path, num2str(i), '/T2_FLAIR.nii,1'), settings);
        labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
        dice_results_reg_t2(j, i, :) = dice(res_seg , labels);
    end
end
save('dice_reg_T2.mat','dice_results_reg_t2');

table1_t2 = zeros(4, 8);
for i=1:3
    table1_t2(1,2*(i-1)+1) = mean(dice_results_reg_t2(1,:,i)); table1_t2(1,2*(i-1)+2) = std(dice_results_reg_t2(1,:,i));
    table1_t2(2,2*(i-1)+1) = mean(dice_results_reg_t2(2,:,i)); table1_t2(2,2*(i-1)+2) = std(dice_results_reg_t2(2,:,i));
    table1_t2(3,2*(i-1)+1) = mean(dice_results_reg_t2(3,:,i)); table1_t2(3,2*(i-1)+2) = std(dice_results_reg_t2(3,:,i));
    table1_t2(4,2*(i-1)+1) = mean(dice_results_reg_t2(4,:,i)); table1_t2(4,2*(i-1)+2) = std(dice_results_reg_t2(4,:,i));
end
table1_t2(1,7) = mean(reshape(dice_results_reg_t2(1,:,:),1,[])); table1_t2(1,8) = std(reshape(dice_results_reg_t2(1,:,:),1,[]));
table1_t2(2,7) = mean(reshape(dice_results_reg_t2(2,:,:),1,[])); table1_t2(2,8) = std(reshape(dice_results_reg_t2(2,:,:),1,[]));
table1_t2(3,7) = mean(reshape(dice_results_reg_t2(3,:,:),1,[])); table1_t2(3,8) = std(reshape(dice_results_reg_t2(3,:,:),1,[]));
table1_t2(4,7) = mean(reshape(dice_results_reg_t2(4,:,:),1,[])); table1_t2(4,8) = std(reshape(dice_results_reg_t2(4,:,:),1,[]));
table1_t2

table2_t2 = zeros(4, 10);
for i=1:5
    table2_t2(1,2*(i-1)+1) = mean(dice_results_reg_t2(1,i,:)); table2_t2(1,2*(i-1)+2) = std(dice_results_reg_t2(1,i,:));
    table2_t2(2,2*(i-1)+1) = mean(dice_results_reg_t2(2,i,:)); table2_t2(2,2*(i-1)+2) = std(dice_results_reg_t2(2,i,:));
    table2_t2(3,2*(i-1)+1) = mean(dice_results_reg_t2(3,i,:)); table2_t2(3,2*(i-1)+2) = std(dice_results_reg_t2(3,i,:));
    table2_t2(4,2*(i-1)+1) = mean(dice_results_reg_t2(4,i,:)); table2_t2(4,2*(i-1)+2) = std(dice_results_reg_t2(4,i,:));
end
table2_t2

% FWHM on T1 images
dice_results_fwhm_t2 = zeros(5, 5, 3);
fwhms = [40 60 80 100 120];
settings = struct();
settings.biasreg = 0.01;
settings.biasfwhm = 60;
settings.write = [0 0];
for j=1:length(fwhms)
    for i=1:5
        settings.biasfwhm = fwhms(j);
        % Running everything for one case:
        res_seg = segment_brain_tissues(fullfile(base_data_path, num2str(i), '/T2_FLAIR.nii,1'), settings);
        labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
        dice_results_fwhm_t2(j, i, :) = dice(res_seg , labels);
    end
end
save('dice_fwhm_T2.mat','dice_results_fwhm_t2');

table3_t2= zeros(4, 8);
for i=1:3
    table3_t2(1,2*(i-1)+1) = mean(dice_results_fwhm_t2(1,:,i)); table3_t2(1,2*(i-1)+2) = std(dice_results_fwhm_t2(1,:,i));
    table3_t2(2,2*(i-1)+1) = mean(dice_results_fwhm_t2(2,:,i)); table3_t2(2,2*(i-1)+2) = std(dice_results_fwhm_t2(2,:,i));
    table3_t2(3,2*(i-1)+1) = mean(dice_results_fwhm_t2(3,:,i)); table3_t2(3,2*(i-1)+2) = std(dice_results_fwhm_t2(3,:,i));
    table3_t2(4,2*(i-1)+1) = mean(dice_results_fwhm_t2(4,:,i)); table3_t2(4,2*(i-1)+2) = std(dice_results_fwhm_t2(4,:,i));
    table3_t2(5,2*(i-1)+1) = mean(dice_results_fwhm_t2(5,:,i)); table3_t2(5,2*(i-1)+2) = std(dice_results_fwhm_t2(5,:,i));
end
table3_t2(1,7) = mean(reshape(dice_results_fwhm_t2(1,:,:),1,[])); table3_t2(1,8) = std(reshape(dice_results_fwhm_t2(1,:,:),1,[]));
table3_t2(2,7) = mean(reshape(dice_results_fwhm_t2(2,:,:),1,[])); table3_t2(2,8) = std(reshape(dice_results_fwhm_t2(2,:,:),1,[]));
table3_t2(3,7) = mean(reshape(dice_results_fwhm_t2(3,:,:),1,[])); table3_t2(3,8) = std(reshape(dice_results_fwhm_t2(3,:,:),1,[]));
table3_t2(4,7) = mean(reshape(dice_results_fwhm_t2(4,:,:),1,[])); table3_t2(4,8) = std(reshape(dice_results_fwhm_t2(4,:,:),1,[]));
table3_t2(5,7) = mean(reshape(dice_results_fwhm_t2(5,:,:),1,[])); table3_t2(5,8) = std(reshape(dice_results_fwhm_t2(5,:,:),1,[]));
table3_t2

table4_t2 = zeros(4, 10);
for i=1:5
    table4_t2(1,2*(i-1)+1) = mean(dice_results_fwhm_t2(1,i,:)); table4_t2(1,2*(i-1)+2) = std(dice_results_fwhm_t2(1,i,:));
    table4_t2(2,2*(i-1)+1) = mean(dice_results_fwhm_t2(2,i,:)); table4_t2(2,2*(i-1)+2) = std(dice_results_fwhm_t2(2,i,:));
    table4_t2(3,2*(i-1)+1) = mean(dice_results_fwhm_t2(3,i,:)); table4_t2(3,2*(i-1)+2) = std(dice_results_fwhm_t2(3,i,:));
    table4_t2(4,2*(i-1)+1) = mean(dice_results_fwhm_t2(4,i,:)); table4_t2(4,2*(i-1)+2) = std(dice_results_fwhm_t2(4,i,:));
    table4_t2(5,2*(i-1)+1) = mean(dice_results_fwhm_t2(5,i,:)); table4_t2(5,2*(i-1)+2) = std(dice_results_fwhm_t2(5,i,:));
end
table4_t2