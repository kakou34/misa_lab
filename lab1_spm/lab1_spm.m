%% MISA LAB 1, Joaquin Oscar Seia - Kaouther Mouheb 
clear all; 
close all; 
clc;
% Modify paths to your device:
addpath('/home/jseia/Desktop/MATLAB/spm12')
base_data_path = '/home/jseia/Desktop/MAIA/Clases/spain/misa/misa_lab/lab1_spm/data/P2_data/';

% Run segmentations and metrics
dice_results = zeros(5, 3);
for i=1:5
    % Running everything for one case:
    res_seg = segment_brain_tissues(fullfile(base_data_path, num2str(i), '/T1.nii,1'));
    labels = double(niftiread(fullfile(base_data_path, num2str(i), '/LabelsForTesting.nii')));
    dice_results(i, :) = dice(res_seg , labels);

    % plotting an example segmentation of one slice
    figure;
    t = tiledlayout(1,3); 
    nexttile
    imagesc(res_seg(:, :, 24))
    nexttile
    imagesc(labels(:, :, 24))
    nexttile
    imagesc(labels(:, :, 24) - res_seg(:, :, 24))

end
save('dice.mat','dice_results');

% Get statistics
figure
csf = dice_results(:, 1);
gm = dice_results(:, 2);
wm = dice_results(:, 3);
boxplot([csf, gm, wm],'Labels',{'CSF', 'GM','WM'})
title('Dice Scores per tissue type')






