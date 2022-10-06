%% MISA LAB 1, Joaquin Oscar Seia - Kaouther Mouheb 
% This notebook contains the experiments on the segmentation parameters

%% Bias field correction
clear; close all; clc;
% addpath('D:\Master\Girona\Segmentation\labs\lab1\spm12\spm12') %import SPM
% base_data_path = 'D:\Master\Girona\Segmentation\labs\lab1\'; % where the data is 
addpath('/home/jseia/Desktop/MATLAB/spm12') %import SPM
base_data_path = '/home/jseia/Desktop/MAIA/Clases/spain/misa/misa_lab/lab1_spm/data/P2_data'; % where the data is 

%% Parameters exploration on T2 images
% Regularization
dice_results_ngauss_t2 = zeros(4, 5, 3);
settings = struct();
settings.biasreg = 1;
settings.biasfwhm = 120;
settings.write = [0 0];
settings.ngauss = 1;
for j=1:4
    for i=1:5
        settings.ngauss = j;
        structural_fns = struct();
        structural_fns.ch1 = fullfile(base_data_path, num2str(i), '/T2_FLAIR.nii,1'); % scan path
        out_fn = fullfile(['figs/seg_params_exp/seg_ngauss_t2_ng_2', num2str(i), num2str(j),'.nii']);
        [res_seg, ~, ~, ~, ~] = segment_brain_tissues(structural_fns, settings, out_fn);
        labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
        dice_results_ngauss_t2(j, i, :) = dice(res_seg , labels);
    end
end
save('dice/dice_ngauss_T2.mat','dice_results_ngauss_t2');
mean_std_dice_ngauss_t2_tissue_wise, mean_std_dice_ngauss_t2_subject_wise = ...
    bias_params_tables(dice_results_ngauss_t2);
save('dice/mean_std_dice_ngauss_t2_tissue_wise.mat','mean_std_dice_ngauss_t2_tissue_wise');
save('dice/mean_std_dice_ngauss_t2_subject_wise.mat','mean_std_dice_ngauss_t2_subject_wise');

%% CLEANUP
% Parameters exploration on T2 images
% Regularization
dice_results_cleanup_t2 = zeros(4, 5, 3);
settings = struct();
settings.biasreg = 1;
settings.biasfwhm = 120;
settings.write = [0 0];
settings.cleanup = 1;
for j=1:2
    for i=1:5
        if i == 2
            settings.ngauss = 2;
        end
        settings.cleanup = j;
        structural_fns = struct();
        structural_fns.ch1 = fullfile(base_data_path, num2str(i), '/T2_FLAIR.nii,1'); % scan path
        out_fn = fullfile(['figs/seg_params_exp/seg_cleanup_t2_', num2str(i), num2str(j),'.nii']);
        [res_seg, ~, ~, ~, ~] = segment_brain_tissues(structural_fns, settings, out_fn);
        labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
        dice_results_cleanup_t2(j, i, :) = dice(res_seg , labels);
    end
end
save('dice/dice_cleanup_T2.mat','dice_results_cleanup_t2');
mean_std_dice_cleanup_t2_tissue_wise, mean_std_dice_cleanup_t2_subject_wise = ...
    bias_params_tables(dice_results_cleanup_t2);
save('dice/mean_std_dice_cleanup_t2_tissue_wise.mat','mean_std_dice_cleanup_t2_tissue_wise');
save('dice/mean_std_dice_cleanup_t2_subject_wise.mat','mean_std_dice_cleanup_t2_subject_wise');

% Parameters exploration on T1 images
% Regularization
dice_results_cleanup_t1 = zeros(4, 5, 3);
settings = struct();
settings.biasreg = 1;
settings.biasfwhm = 120;
settings.write = [0 0];
settings.cleanup = 1;
for j=1:2
    for i=1:5
        if i == 2
            settings.ngauss = 2;
        end
        settings.cleanup = j;
        structural_fns = struct();
        structural_fns.ch1 = fullfile(base_data_path, num2str(i), '/T1.nii,1'); % scan path
        out_fn = fullfile(['figs/seg_params_exp/seg_cleanup_t1_', num2str(i), num2str(j),'.nii']);
        [res_seg, ~, ~, ~, ~] = segment_brain_tissues(structural_fns, settings, out_fn);
        labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
        dice_results_cleanup_t1(j, i, :) = dice(res_seg , labels);
    end
end
save('dice/dice_cleanup_T1.mat','dice_results_cleanup_t1');
mean_std_dice_cleanup_t1_tissue_wise, mean_std_dice_cleanup_t1_subject_wise = ...
    bias_params_tables(dice_results_cleanup_t1);
save('dice/mean_std_dice_cleanup_t1_tissue_wise.mat','mean_std_dice_cleanup_t1_tissue_wise');
save('dice/mean_std_dice_cleanup_t1_subject_wise.mat','mean_std_dice_cleanup_t1_subject_wise');

% Parameters exploration on T1+T2 images
% Regularization
dice_results_cleanup_t1_t2 = zeros(4, 5, 3);
settings = struct();
settings.biasreg = 1;
settings.biasfwhm = 120;
settings.write = [0 0];
settings.cleanup = 1;
for j=1:2
    for i=1:5
        if i == 2
            settings.ngauss = 2;
        end
        settings.cleanup = j;
        structural_fns = struct();
        structural_fns.ch1 = fullfile(base_data_path, num2str(i), '/T1.nii,1'); % scan path
        structural_fns.ch2 = fullfile(base_data_path, num2str(i), '/T2_FLAIR.nii,1'); % scan path
        out_fn = fullfile(['figs/seg_params_exp/seg_cleanup_t1_t2_', num2str(i), num2str(j),'.nii']);
        [res_seg, ~, ~, ~, ~] = segment_brain_tissues(structural_fns, settings, out_fn);
        labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
        dice_results_cleanup_t1_t2(j, i, :) = dice(res_seg , labels);
    end
end
save('dice/dice_cleanup_T1_T2.mat','dice_results_cleanup_t1_t2');
mean_std_dice_cleanup_t1_t2_tissue_wise, mean_std_dice_cleanup_t1_t2_subject_wise = ...
    bias_params_tables(dice_results_cleanup_t1_t2);
save('dice/mean_std_dice_cleanup_t1_t2_tissue_wise.mat','mean_std_dice_cleanup_t1_t2_tissue_wise');
save('dice/mean_std_dice_cleanup_t1_t2_subject_wise.mat','mean_std_dice_cleanup_t1_t2_subject_wise');