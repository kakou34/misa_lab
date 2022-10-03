clear all; 
close all; 
clc;

%% Finetuning the parameters for bias field correction

fid = fopen('log.txt','wt'); %log file

addpath('D:\Master\Girona\Segmentation\labs\lab1\spm12\spm12')
base_data_path = 'D:\Master\Girona\Segmentation\labs\lab1\';

samp = 2; % Sample
seq = 'T1'; % Modality to use

structural_fn = fullfile(base_data_path, num2str(samp), strcat('/', seq, '.nii,1'));
vol = niftiread(fullfile(base_data_path, num2str(samp), strcat('/', seq, '.nii')));
labels = double(niftiread(fullfile(base_data_path, num2str(1), '/LabelsForTesting.nii')));


% 1 - Modify regularization parameter
bias_reg = [0, 0.0001, 0.01, 0.1, 1]; 
bias_fwhm = 60;

dice_s = zeros(length(bias_reg), 3);

% iterate over each value of reg 
for i = 1:length(bias_reg)
    result = spm_seg(structural_fn, bias_reg(i), bias_fwhm);
    [res_seg, corrected, field] = generate_mask(result);

    % Dice 
    dice_res = dice(res_seg, labels);
    dice_s(i, :) = dice_res;

    %log
    msg = strcat("Sample ", num2str(samp), ", ", seq, " => reg: ", num2str(bias_reg(i)), " fwhm: ", num2str(bias_fwhm), " dice CSF: ", num2str(dice_res(1)), " dice GM: ", num2str(dice_res(2)), " dice WM: ", num2str(dice_res(3)), "\n");
    fprintf(fid, msg);

    % Qualitative results 
    ttl = strcat("REG: ", num2str(bias_reg(i)), ", FWHM: ", num2str(bias_fwhm));
    slice_i = 24;
    figure;
    t = tiledlayout(1,3); 
    title(t,ttl);
    nexttile
    imshow(uint8(vol(:, :, slice_i)));
    nexttile
    imshow(uint8(corrected(:, :, slice_i)));
    nexttile
    imagesc(field(:, :, slice_i));
    colormap gray
    axis square off
    
    fig_fn = strcat("figs/bias_exp/s", num2str(samp), "_", seq, "_", num2str(bias_reg(i)), "_", num2str(bias_fwhm),".png");
    exportgraphics(t,fig_fn)
end

% visualize dice
x = bias_reg;
y1 = dice_s(:, 1); % CSF Dice for each reg value
y2 = dice_s(:, 2); % GM Dice 
y3 = dice_s(:, 3); % WM Dice
y4 = mean(transpose(dice_s)); % Average dice over the 3 tissues

figure;
plot(x,y1,'g',x,y2,'b',x,y3,'r', x,y4, 'm--');
xlabel('Bias Regularization');
ylabel('Dice Score');
legend('CSF','GM', 'WM', 'Avg');
title(strcat('Dice VS Bias Regularisation - ', seq));
exportgraphics(gcf,strcat('figs\bias_exp\s', num2str(samp),'_dice_reg_', seq, '.png'))

% save dice
dice_fn = strcat('dice\dice_s', num2str(samp), '_', seq, '_reg.mat');
save(dice_fn,'dice_s');


% ########################
% 2 - Modify Bias Field FWHM

bias_reg = 0.01; 
bias_fwhm = [40, 60, 80, 100, 120, 140];

dice_s = zeros(length(bias_fwhm), 3);

% iterate over each value of reg 
for i = 1:length(bias_fwhm)
    result = spm_seg(structural_fn, bias_reg, bias_fwhm(i)); 
    [res_seg, corrected, field] = generate_mask(result);

    % Dice 
    dice_res = dice(res_seg, labels);
    dice_s(i, :) = dice_res;

    %log
    msg = strcat("Sample ", num2str(samp), ", ", seq, " => reg: ", num2str(bias_reg), " fwhm: ", num2str(bias_fwhm(i)), " dice CSF: ", num2str(dice_res(1)), " dice GM: ", num2str(dice_res(2)), " dice WM: ", num2str(dice_res(3)), "\n");
    fprintf(fid, msg);

    % Qualitative results 
    ttl = strcat("REG: ", num2str(bias_reg), ", FWHM: ", num2str(bias_fwhm(i)));
    slice_i = 24;
    figure;
    t = tiledlayout(1,3); 
    title(t,ttl);
    nexttile
    imshow(uint8(vol(:, :, slice_i)));
    nexttile
    imshow(uint8(corrected(:, :, slice_i)));
    nexttile
    imagesc(field(:, :, slice_i));
    colormap gray
    axis square off
    fig_fn = strcat("figs/bias_exp/s", num2str(samp), "_", seq, "_", num2str(bias_reg), "_", num2str(bias_fwhm(i)),".png");
    exportgraphics(t,fig_fn)
end

% visualize dice
x = bias_fwhm;
y1 = dice_s(:, 1); % CSF Dice for each reg value
y2 = dice_s(:, 2); % GM Dice 
y3 = dice_s(:, 3); % WM Dice
y4 = mean(transpose(dice_s)); % Average dice over the 3 tissues

figure;
plot(x,y1,'g',x,y2,'b',x,y3,'r', x,y4, 'm--');
xlabel('Bias FWHM');
ylabel('Dice Score');
legend('CSF','GM', 'WM', 'Avg');
title(strcat('Dice VS Bias FWHM - ', seq));
exportgraphics(gcf,strcat('figs\bias_exp\s', num2str(samp),'_dice_fwhm_', seq, '.png'))


% save dice
dice_fn = strcat('dice\dice_s', num2str(samp), '_', seq, '_fwhm.mat');
save(dice_fn,'dice_s');
