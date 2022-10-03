%% MISA LAB 1, Joaquin Oscar Seia - Kaouther Mouheb 
clear all; 
close all; 
clc;

% Modify paths to your device:
addpath('D:\Master\Girona\Segmentation\labs\lab1\spm12\spm12')
base_data_path = 'D:\Master\Girona\Segmentation\labs\lab1\';

results_t1 = zeros(5, 16);
results_t2 = zeros(5, 16);

%% Segmentation of volume 1: 

% 1 - T1:
volume = niftiread(fullfile(base_data_path, num2str(1), '/T1.nii')); % reading the original scan
labels = double(niftiread(fullfile(base_data_path, num2str(1), '/LabelsForTesting.nii')));

% Bias field correction parameters obtained from experiment_bias_field.m
bias_reg = 0.0001;
bias_fwhm = 40; 

% segmentation results
structural_fn = fullfile(base_data_path, num2str(1), '/T1.nii,1');
[res_seg, correct] = segment_brain_tissues(structural_fn, bias_reg, bias_fwhm);

%calculate statistics 
metrics = get_metrics(res_seg, labels, volume, correct);
results_t1(1, :) = metrics;

plot_results(volume, labels, res_seg);

exportgraphics(gcf,"figs/results/seg_res_s1_t1.png");

%save results 
% Save results with correct spacing
template_fn = [structural_fn];
template_spm = spm_vol(template_fn);
new_nii = spm_create_vol(template_spm);
new_nii.fname = 'figs/results/seg_res_s1_t1.nii';
spm_write_vol(new_nii, res_seg);


% 
% 2 - T2 FLAIR:
volume = niftiread(fullfile(base_data_path, num2str(1), '/T2_FLAIR.nii')); % reading the original scan
labels = double(niftiread(fullfile(base_data_path, num2str(1), '/LabelsForTesting.nii')));

% Bias field correction parameters obtained from experiment_bias_field.m
bias_reg = 0.0001;
bias_fwhm = 120;

% segmentation results
structural_fn = fullfile(base_data_path, num2str(1), '/T2_FLAIR.nii,1');
[res_seg, correct] = segment_brain_tissues(structural_fn, bias_reg, bias_fwhm);

%calculate statistics 
metrics = get_metrics(res_seg, labels, volume, correct);
results_t2(1, :) = metrics;

plot_results(volume, labels, res_seg);

exportgraphics(gcf,"figs/results/seg_res_s1_t2.png");

%save results 
% Save results with correct spacing
template_fn = [structural_fn];
template_spm = spm_vol(template_fn);
new_nii = spm_create_vol(template_spm);
new_nii.fname = 'figs/results/seg_res_s1_t2.nii';
spm_write_vol(new_nii, res_seg);









