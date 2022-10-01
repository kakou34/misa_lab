clear all; 
close all; 
clc;

%% Playing with the first sample's bias field => T1 

fid = fopen('log.txt','wt');

addpath('D:\Master\Girona\Segmentation\labs\lab1\spm12\spm12')
base_data_path = 'D:\Master\Girona\Segmentation\labs\lab1\';

structural_fn = fullfile(base_data_path, num2str(1), '/T1.nii,1');

vol = niftiread(fullfile(base_data_path, num2str(1), '/T1.nii'));
labels = double(niftiread(fullfile(base_data_path, num2str(1), '/LabelsForTesting.nii')));


% 1 - Modify regularization parameter
% bias_reg = [0, 0.0001, 0.01, 1];
% bias_fwhm = 60;
% 
% for i = 1:4
%     result = spm_seg(structural_fn, bias_reg(i), bias_fwhm); 
%     % Read tissue probability maps
%     gm = niftiread(result.gm_fn(1:end-2));
%     wm = niftiread(result.wm_fn(1:end-2));
%     csf = niftiread(result.csf_fn(1:end-2));
%     bone = niftiread(result.bone_fn(1:end-2));
%     soft = niftiread(result.soft_fn(1:end-2));
%     air = niftiread(result.air_fn(1:end-2));
%     corrected = niftiread(result.correct_fn(1:end-2));
%     field = niftiread(result.field_fn(1:end-2));
%     
%     % Classify the tissues according to maximum a posteriori probabilies
%     maps = zeros(240, 240, 48, 6); 
%     maps(:, :, :, 1) = csf; 
%     maps(:, :, :, 2) = gm;
%     maps(:, :, :, 3) = wm;
%     maps(:, :, :, 4) = bone;
%     maps(:, :, :, 5) = soft;
%     maps(:, :, :, 6) = air;
%     [~, res_seg] = max(maps, [], 4);
%     
%     % Ignore bone, soft tissue and air
%     res_seg(res_seg>3) = 0;
% 
%     % Dice 
%     dice_res = dice(res_seg, labels);
%     msg = strcat("Sample 1, T1 => reg: ", num2str(bias_reg(i)), " fwhm: ", num2str(bias_fwhm), " dice CSF: ", num2str(dice_res(1)), " dice GM: ", num2str(dice_res(2)), " dice WM: ", num2str(dice_res(3)));
%     disp(msg);
%     fprintf(fid, msg);
%     ttl = strcat("REG: ", num2str(bias_reg(i)), ", FWHM: ", num2str(bias_fwhm));
%     % Plotting figures for 1 slice
%     slice_i = 24;
%     figure;
%     t = tiledlayout(1,3); 
%     title(t,ttl);
% 
%     nexttile
%     imagesc(vol(:, :, slice_i));
%     colormap gray
%     axis square
%     axis off
%     nexttile
%     imagesc(corrected(:, :, slice_i));
%     colormap gray
%     axis square
%     axis off
%     nexttile
%     imagesc(field(:, :, slice_i));
%     colormap gray
%     axis square
%     axis off
% 
%     fig_fn = strcat("figs/bias_exp/s1_t1_", num2str(bias_reg(i)), "_",num2str(bias_fwhm),".png");
%     exportgraphics(t,fig_fn)
% end


% 2 - Modify fWHM parameter
bias_reg = 0.01;
bias_fwhm = [40, 60, 100, 130];

for i = 1:4
    result = spm_seg(structural_fn, bias_reg, bias_fwhm(i)); 
    % Read tissue probability maps
    gm = niftiread(result.gm_fn(1:end-2));
    wm = niftiread(result.wm_fn(1:end-2));
    csf = niftiread(result.csf_fn(1:end-2));
    bone = niftiread(result.bone_fn(1:end-2));
    soft = niftiread(result.soft_fn(1:end-2));
    air = niftiread(result.air_fn(1:end-2));
    corrected = niftiread(result.correct_fn(1:end-2));
    field = niftiread(result.field_fn(1:end-2));
    
    % Classify the tissues according to maximum a posteriori probabilies
    maps = zeros(240, 240, 48, 6); 
    maps(:, :, :, 1) = csf; 
    maps(:, :, :, 2) = gm;
    maps(:, :, :, 3) = wm;
    maps(:, :, :, 4) = bone;
    maps(:, :, :, 5) = soft;
    maps(:, :, :, 6) = air;
    [~, res_seg] = max(maps, [], 4);
    
    % Ignore bone, soft tissue and air
    res_seg(res_seg>3) = 0;

    % Dice 
    dice_res = dice(res_seg, labels);
    msg = strcat("\nSample 1, T1 => reg: ", num2str(bias_reg), " fwhm: ", num2str(bias_fwhm(i)), " dice CSF: ", num2str(dice_res(1)), " dice GM: ", num2str(dice_res(2)), " dice WM: ", num2str(dice_res(3)));
    disp(msg);
    fprintf(fid, msg);
    ttl = strcat("REG: ", num2str(bias_reg), ", FWHM: ", num2str(bias_fwhm(i)));
    % Plotting figures for 1 slice
    slice_i = 24;
    figure;
    t = tiledlayout(1,3); 
    title(t,ttl);

    nexttile
    imagesc(vol(:, :, slice_i));
    colormap gray
    axis square
    axis off
    nexttile
    imagesc(corrected(:, :, slice_i));
    colormap gray
    axis square
    axis off
    nexttile
    imagesc(field(:, :, slice_i));
    colormap gray
    axis square
    axis off

    fig_fn = strcat("figs/bias_exp/s1_t1_", num2str(bias_reg), "_",num2str(bias_fwhm(i)),".png");
    exportgraphics(t,fig_fn)
end

fclose(fid);
