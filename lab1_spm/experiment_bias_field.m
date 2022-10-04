%% MISA LAB 1, Joaquin Oscar Seia - Kaouther Mouheb 
% This notebook contains the experiments on the parameters of bias field
% correction

%% Bias field correction
clear; close all; clc;
% addpath('D:\Master\Girona\Segmentation\labs\lab1\spm12\spm12') %import SPM
% base_data_path = 'D:\Master\Girona\Segmentation\labs\lab1\'; % where the data is 
addpath('/home/jseia/Desktop/MATLAB/spm12') %import SPM
base_data_path = '/home/jseia/Desktop/MAIA/Clases/spain/misa/misa_lab/lab1_spm/data/P2_data'; % where the data is 

%% Parameters exploration on T1 images
% Regularization
dice_results_reg_t1 = zeros(4, 5, 3);
regs = [0 0.0001 0.01 1];
settings = struct();
settings.biasreg = 0;
settings.biasfwhm = 60;
settings.write = [0 0];
for j=1:length(regs)
    for i=1:5
        settings.biasreg = regs(j);
        structural_fns = struct();
        structural_fns.ch1 = fullfile(base_data_path, num2str(i), '/T1.nii,1'); % scan path
        out_fn = fullfile(['figs/bias_exp/seg_res_t1.nii']);
        [res_seg, ~, ~, ~, ~] = segment_brain_tissues(structural_fns, settings, out_fn);
        labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
        dice_results_reg_t1(j, i, :) = dice(res_seg , labels);
    end
end
save('dice_reg_T1.mat','dice_results_reg_t1');
mean_std_dice_reg_t1_tissue_wise, mean_std_dice_reg_t1_subject_wise = ...
    bias_params_tables(dice_results_reg_t1);
save('dice/mean_std_dice_reg_t1_tissue_wise.mat','mean_std_dice_reg_t1_tissue_wise');
save('dice/mean_std_dice_reg_t1_subject_wise.mat','mean_std_dice_reg_t1_subject_wise');

% FWHM
dice_results_fwhm_t1 = zeros(5, 5, 3);
fwhms = [40 60 80 100 120];
settings = struct();
settings.biasreg = 0.01;
settings.biasfwhm = 60;
settings.write = [0 0];
for j=1:length(fwhms)
    for i=1:5
        settings.biasfwhm = fwhms(j);
        structural_fns = struct();
        structural_fns.ch1 = fullfile(base_data_path, num2str(i), '/T1.nii,1'); % scan path
        out_fn = fullfile(['figs/bias_exp/seg_res_t1.nii']);
        [res_seg, ~, ~, ~, ~] = segment_brain_tissues(structural_fns, settings, out_fn);
        labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
        dice_results_fwhm_t1(j, i, :) = dice(res_seg , labels);
    end
end
save('dice/dice_fwhm_T1.mat','dice_results_fwhm_t1');
mean_std_dice_fwhm_t1_tissue_wise, mean_std_dice_fwhm_t1_subject_wise = ...
    bias_params_tables(dice_results_fwhm_t1);
save('dice/mean_std_dice_fwhm_t1_tissue_wise.mat','mean_std_dice_fwhm_t1_tissue_wise');
save('dice/mean_std_dice_fwhm_t1_subject_wise.mat','mean_std_dice_fwhm_t1_subject_wise');

%% Parameters exploration on T2 images
% Regularization
dice_results_reg_t2 = zeros(4, 5, 3);
regs = [0 0.0001 0.01 1];
settings = struct();
settings.biasreg = 0;
settings.biasfwhm = 60;
settings.write = [0 0];
for j=1:length(regs)
    for i=1:5
        settings.biasreg = regs(j);
        structural_fns = struct();
        structural_fns.ch1 = fullfile(base_data_path, num2str(i), '/T2_FLAIR.nii,1'); % scan path
        out_fn = fullfile(['figs/bias_exp/seg_res_t2.nii']);
        [res_seg, ~, ~, ~, ~] = segment_brain_tissues(structural_fns, settings, out_fn);
        labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
        dice_results_reg_t2(j, i, :) = dice(res_seg , labels);
    end
end
save('dice_reg_T2.mat','dice_results_reg_t2');
mean_std_dice_reg_t2_tissue_wise, mean_std_dice_reg_t2_subject_wise = ...
    bias_params_tables(dice_results_reg_t2);
save('dice/mean_std_dice_reg_t2_tissue_wise.mat','mean_std_dice_reg_t2_tissue_wise');
save('dice/mean_std_dice_reg_t2_subject_wise.mat','mean_std_dice_reg_t2_subject_wise');

% FWHM
dice_results_fwhm_t2 = zeros(5, 5, 3);
fwhms = [40 60 80 100 120];
settings = struct();
settings.biasreg = 0.01;
settings.biasfwhm = 60;
settings.write = [0 0];
for j=1:length(fwhms)
    for i=1:5
        settings.biasfwhm = fwhms(j);
        structural_fns = struct();
        structural_fns.ch1 = fullfile(base_data_path, num2str(i), '/T2_FLAIR.nii,1'); % scan path
        out_fn = fullfile(['figs/bias_exp/seg_res_t2.nii']);
        [res_seg, ~, ~, ~, ~] = segment_brain_tissues(structural_fns, settings, out_fn);
        labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
        dice_results_fwhm_t2(j, i, :) = dice(res_seg , labels);
    end
end
save('dice/dice_fwhm_T2.mat','dice_results_fwhm_t2');
mean_std_dice_fwhm_t2_tissue_wise, mean_std_dice_fwhm_t2_subject_wise= ...
    bias_params_tables(dice_results_fwhm_t2);
save('dice/mean_std_dice_fwhm_t2_tissue_wise.mat','mean_std_dice_fwhm_t2_tissue_wise');
save('dice/mean_std_dice_fwhm_t2_subject_wise.mat','mean_std_dice_fwhm_t2_subject_wise');

%% Parameters exploration on T1+T2 images
% Regularization
dice_results_reg_t1_t2 = zeros(4, 5, 3);
regs = [0 0.0001 0.01 1];
settings = struct();
settings.biasreg = 0;
settings.biasfwhm = 60;
settings.write = [0 0];
for j=1:length(regs)
    for i=1:5
        settings.biasreg = regs(j);
        structural_fns = struct();
        structural_fns.ch1 = fullfile(base_data_path, num2str(i), '/T1.nii,1'); % scan path
        structural_fns.ch2 = fullfile(base_data_path, num2str(i), '/T2_FLAIR.nii,1'); % scan path
        out_fn = fullfile(['figs/bias_exp/seg_res_t1_t2.nii']);
        [res_seg, ~, ~, ~, ~] = segment_brain_tissues(structural_fns, settings, out_fn);
        labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
        dice_results_reg_t1_t2(j, i, :) = dice(res_seg , labels);
    end
end
save('dice_reg_T1_T2.mat','dice_results_reg_t1_t2');
mean_std_dice_reg_t1_t2_tissue_wise, mean_std_dice_reg_t1_t2_subject_wise = ...
    bias_params_tables(dice_results_reg_t1_t2);
save('dice/mean_std_dice_reg_t1_t2_tissue_wise.mat','mean_std_dice_reg_t1_t2_tissue_wise');
save('dice/mean_std_dice_reg_t1_t2_subject_wise.mat','mean_std_dice_reg_t1_t2_subject_wise');

% FWHM
dice_results_fwhm_t1_t2 = zeros(5, 5, 3);
fwhms = [40 60 80 100 120];
settings = struct();
settings.biasreg = 0.01;
settings.biasfwhm = 60;
settings.write = [0 0];
for j=1:length(fwhms)
    for i=1:5
        settings.biasfwhm = fwhms(j);
        structural_fns = struct();
        structural_fns.ch1 = fullfile(base_data_path, num2str(i), '/T1.nii,1'); % scan path
        structural_fns.ch2 = fullfile(base_data_path, num2str(i), '/T2_FLAIR.nii,1'); % scan path
        out_fn = fullfile(['figs/bias_exp/seg_res_t1_t2.nii']);
        [res_seg, ~, ~, ~, ~] = segment_brain_tissues(structural_fns, settings, out_fn);
        labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
        dice_results_fwhm_t1_t2(j, i, :) = dice(res_seg , labels);
    end
end
save('dice/dice_fwhm_T1.mat','dice_results_fwhm_t1_t2');
mean_std_dice_fwhm_t1_t2_tissue_wise, mean_std_dice_fwhm_t1_t2_subject_wise = ...
    bias_params_tables(dice_results_fwhm_t1_t2);
save('dice/mean_std_dice_fwhm_t1_t2_tissue_wise.mat','mean_std_dice_fwhm_t1_t2_tissue_wise');
save('dice/mean_std_dice_fwhm_t1_t2_subject_wise.mat','mean_std_dice_fwhm_t1_t2_subject_wise');