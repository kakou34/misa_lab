clear all; 
close all; 
clc;

addpath('D:\Master\Girona\Segmentation\labs\lab1\spm12\spm12') %import SPM

%% Performing Skull Stripping
 
base_data_path = 'D:\Master\Girona\Segmentation\labs\lab1\'; % where the data is 

dice_score = zeros(2, 5); 

% iterate over each sample
for i=1:5
    disp(strcat("#### Sample ", num2str(i), " ####"))

    structural_fn = fullfile(base_data_path, num2str(i), '/T1.nii,1'); % scan path
    vol = niftiread(fullfile(base_data_path, num2str(i), '/T1.nii')); % reading the original scan
    labels = niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')); % reading the ground truth

    brain_mask_gt = int16(labels>0); % get the binary mask of the brain (0 for BG, 1 for Tissue)

    % I- T1 Scan
    
    % bias field correction parameters
    b_reg = 0.0001;
    b_fwhm = 40;
    
    result = spm_seg(structural_fn, b_reg, b_fwhm);  
    % Read tissue probability maps
    gm = niftiread(result.gm_fn(1:end-2));
    wm = niftiread(result.wm_fn(1:end-2));
    csf = niftiread(result.csf_fn(1:end-2));
    bone = niftiread(result.bone_fn(1:end-2));
    soft = niftiread(result.soft_fn(1:end-2));
    air = niftiread(result.air_fn(1:end-2));
        
    % Classify the tissues according to maximum a posteriori probabilies
    maps = zeros(240, 240, 48, 6); 
    maps(:, :, :, 1) = csf; 
    maps(:, :, :, 2) = gm;
    maps(:, :, :, 3) = wm;
    maps(:, :, :, 4) = bone;
    maps(:, :, :, 5) = soft;
    maps(:, :, :, 6) = air;
    [~, res_seg] = max(maps, [], 4);
        
    % Extract brain mask
    brain_mask = int16(res_seg<4); % Keep GM+WM+CSF
    skull_mask = int16(res_seg==4); % bone mask 
    
    % remove sull
    skull_stripped = vol .* brain_mask;

    % calculate dice
    dice_score(1, i) = dice(brain_mask, brain_mask_gt);
    
    % Plotting figures for 1 slice
    slice_i = 24;
    figure;
    t = tiledlayout(1,4);
    nexttile
    imshow(uint8(vol(:, :, slice_i)));
    title("Original T1 Slice");
    
    nexttile
    imagesc(skull_mask(:, :, slice_i));
    axis square off
    title("Skull Mask");
    
    nexttile
    imagesc(brain_mask(:, :, slice_i));
    axis square off
    title("Brain Mask");
    
    nexttile
    imshow(uint8(skull_stripped(:, :, slice_i)));
    axis square off
    title("Skull-stripped");
    colormap gray
    
    fig_fn = strcat("figs/skull_strp/s", num2str(i),"_t1_ss.png");
    exportgraphics(t,fig_fn)
    
    % I- T2 FLAIR Scan
    
    % bias field correction parameters
    b_reg = 1;
    b_fwhm = 120;
    
    result = spm_seg(structural_fn, b_reg, b_fwhm);  
    % Read tissue probability maps
    gm = niftiread(result.gm_fn(1:end-2));
    wm = niftiread(result.wm_fn(1:end-2));
    csf = niftiread(result.csf_fn(1:end-2));
    bone = niftiread(result.bone_fn(1:end-2));
    soft = niftiread(result.soft_fn(1:end-2));
    air = niftiread(result.air_fn(1:end-2));
        
    % Classify the tissues according to maximum a posteriori probabilies
    maps = zeros(240, 240, 48, 6); 
    maps(:, :, :, 1) = csf; 
    maps(:, :, :, 2) = gm;
    maps(:, :, :, 3) = wm;
    maps(:, :, :, 4) = bone;
    maps(:, :, :, 5) = soft;
    maps(:, :, :, 6) = air;
    [~, res_seg] = max(maps, [], 4);
        
    % Extract brain mask
    brain_mask = int16(res_seg<4); % Keep GM+WM+CSF
    skull_mask = int16(res_seg==4); % bone mask
    
    % remove sull
    skull_stripped = vol .* brain_mask;

    % calculate dice
    dice_score(2, i) = dice(brain_mask, brain_mask_gt);
    
    % Plotting figures for 1 slice
    slice_i = 24;
    figure;
    t = tiledlayout(1,4);
    
    nexttile
    imshow(uint8(vol(:, :, slice_i)));
    title("Original Slice");
    
    nexttile
    imagesc(skull_mask(:, :, slice_i));
    axis square off
    title("Skull Mask");
    
    nexttile
    imagesc(brain_mask(:, :, slice_i));
    axis square off
    title("Brain Mask");

    nexttile
    imshow(uint8(skull_stripped(:, :, slice_i)));
    title("Skull-stripped");
    colormap gray
    
    
    fig_fn = strcat("figs/skull_strp/s", num2str(i),"_t2_ss.png");
    exportgraphics(t,fig_fn)

end

