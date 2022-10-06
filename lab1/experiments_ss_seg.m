%% MISA LAB 1, Joaquin Oscar Seia - Kaouther Mouheb 
% This notebook contains the experiments on skull stripping and tissue
% segmentation

%% Performing Skull Stripping
clear; close all; clc;
% addpath('D:\Master\Girona\Segmentation\labs\lab1\spm12\spm12') %import SPM
% base_data_path = 'D:\Master\Girona\Segmentation\labs\lab1\'; % where the data is 
addpath('/home/jseia/Desktop/MATLAB/spm12') %import SPM
base_data_path = '/home/jseia/Desktop/MAIA/Clases/spain/misa/misa_lab/lab1_spm/data/P2_data'; % where the data is 

% Store the quantitative results
results_t1 = zeros(5, 3);
results_t2 = zeros(5, 3);
results_t1_t2 = zeros(5, 3);

% iterate over each subject
for i=1:5
    disp(strcat("#### Sample ", num2str(i), " ####"))
    labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));

    %------------------ I- T1 Scan -----------------------------
    structural_fns = struct();
    structural_fns.ch1 = fullfile(base_data_path, num2str(i), '/T1.nii,1'); % scan path
    vol = niftiread(structural_fns.ch1(1:end-2)); % reading the original scan
    
    % bias field correction parameters
    settings = struct();
    settings.biasreg = 0.0001;
    settings.biasfwhm = 40;
    settings.write = [1 1];
    
    % segement tissues
    out_fn = fullfile(['figs/results/seg_res_s', num2str(i), '_t1.nii']);
    [res_seg, brain_mask, skull_mask, skull_stripped, correct] = ...
        segment_brain_tissues(structural_fns, settings, out_fn);
%     res_seg = double(niftiread(out_fn));

    
    % calculate metrics 
    dice_score = dice(res_seg, labels); 
    results_t1(i, :) = dice_score(:);
    
    % Plotting segmentation figures
    fig_fn = strcat("figs/skull_strp/seg_res_s", num2str(i),"_t1.png");
    plot_segmentation_results(vol, labels, res_seg, fig_fn);

    % Plotting skull stripping figures
    slice_i = 24;
    fig_fn = strcat("figs/skull_strp/s", num2str(i),"_t1_ss.png");
    plot_ss_results(slice_i, fig_fn, vol, brain_mask, skull_mask, skull_stripped)

    
    % ------------------ II- T2 FLAIR Scan----------------------
    structural_fns = struct();
    structural_fns.ch1 = fullfile(base_data_path, num2str(i), '/T2_FLAIR.nii,1'); % scan path
    vol = niftiread(structural_fns.ch1(1:end-2)); % reading the original scan
    
    % bias field correction parameters
    settings = struct();
    settings.biasreg = 0.01;
    settings.biasfwhm = 60;
    settings.write = [1 1];

    % segement tissues
    out_fn = fullfile(['figs/results/seg_res_s', num2str(i), '_t2.nii']);
    [res_seg, brain_mask, skull_mask, skull_stripped, correct] = ...
        segment_brain_tissues(structural_fns, settings, out_fn);
%     res_seg = double(niftiread(out_fn));

    %calculate statistics
    dice_score = dice(res_seg, labels); 
    results_t2(i, :) = dice_score(:);
    
    % Plotting segmentation figures
    fig_fn = strcat("figs/skull_strp/seg_res_s", num2str(i),"_t2.png");
    plot_segmentation_results(vol, labels, res_seg, fig_fn);

    % Plotting figures for 1 slice
    slice_i = 24;
    fig_fn = strcat("figs/skull_strp/s", num2str(i),"_t2_ss.png");
    plot_ss_results(slice_i, fig_fn, vol, brain_mask, skull_mask, skull_stripped)


    % ------------------ II- T1 + T2 FLAIR Scan----------------------
    structural_fns = struct();
    structural_fns.ch1 = fullfile(base_data_path, num2str(i), '/T1.nii,1'); % scan path
    structural_fns.ch2 = fullfile(base_data_path, num2str(i), '/T2_FLAIR.nii,1'); % scan path
    vol = niftiread(structural_fns.ch1(1:end-2)); % reading the original scan
    
    % bias field correction parameters
    settings = struct();
    settings.biasreg = 0.01;
    settings.biasfwhm = 60;
    settings.write = [1 1];

    % segement tissues
    out_fn = fullfile(['figs/results/seg_res_s', num2str(i), '_t1_t2.nii']);
    [res_seg, brain_mask, skull_mask, skull_stripped, correct] = ...
        segment_brain_tissues(structural_fns, settings, out_fn);
%     res_seg = double(niftiread(out_fn));

    %calculate statistics
    dice_score = dice(res_seg, labels); 
    results_t1_t2(i, :) = dice_score(:);
    
    % Plotting segmentation figures
    fig_fn = strcat("figs/skull_strp/seg_res_s", num2str(i),"_t1_t2.png");
    plot_segmentation_results(vol, labels, res_seg, fig_fn);

    % Plotting figures for 1 slice
    slice_i = 24;
    fig_fn = strcat("figs/skull_strp/s", num2str(i),"_t1_t2_ss.png");
    plot_ss_results(slice_i, fig_fn, vol, brain_mask, skull_mask, skull_stripped)
end

save('results_t1.mat','results_t1');
save('results_t2.mat','results_t2');
save('results_t1_t2.mat','results_t1_t2');

%%  Get statistics and plots
means = zeros(3,3); % Columns will contain tissues, rows modalities
means(1, :) = mean(results_t1, 1);
means(2, :) = mean(results_t2, 1);
means(3, :) = mean(results_t1_t2, 1);
stds = zeros(3,3); % Columns will contain tissues, rows modalities
stds(1, :) = std(results_t1, 1);
stds(2, :) = std(results_t2, 1);
stds(3, :) = std(results_t1_t2, 1);
save('dice_means_tissue-cols_modalities-rows.mat','means');
save('dice_stds_tissue-cols_modalities-rows.mat','stds');

figure
boxplotGroup({results_t1, results_t2, results_t1_t2},'PrimaryLabels', {'T1', 'T2F','T1+T2F'}, ...
  'SecondaryLabels', {'CSF', 'GM','WM'}, 'InterGroupSpace', 1)
ylim([0.2,0.9])
grid('on')
title('Dice Scores per tissue type and processing type')
set(gcf,'Position',[100 100 700 400])
set(findobj(gca,'type','line'),'linew',1.5)
fig_fn = "figs/results/boxplots_comparison.png";
exportgraphics(gcf, fig_fn);
