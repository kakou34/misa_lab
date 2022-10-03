function output = spm_seg(structural_fn, flair_fn, settings)
% Function to segment T1 MRI data from a single subject using Matlab/SPM12.

% Steps include coregistering structural image to first functional image,
% segmenting the coregistered structural image into tissue types.
% If spm12 batch parameters are not explicitly set, defaults are assumed. 
%
% INPUT:
% structural_fn      - filename of T1-weighted structural scan
% settings           - configuration settings for the segmentation pipeline
% 
% OUTPUT: 
% output            - structure with filenames and data

% Declare output structure
output = struct;

% Segmentation of coregistered structural image into GM, WM, CSF, etc
% (with implicit warping to MNI space, saving forward and inverse transformations)
disp('Segmentation of T1 volume starting...');
spm('defaults','fmri');
spm_jobman('initcfg');
segmentation = struct;
% Channel

%% Segmentation
segmentation.matlabbatch{1}.spm.spatial.preproc.channel.biasreg = settings.biasreg;
segmentation.matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = settings.biasfwhm;
segmentation.matlabbatch{1}.spm.spatial.preproc.channel.write = settings.write;
segmentation.matlabbatch{1}.spm.spatial.preproc.channel.vols = {fullfile(structural_fn)};
if ~strcmp(flair_fn , 'none')
    disp('multichannel mode')
    segmentation.matlabbatch{1}.spm.spatial.preproc.channel(1).biasreg = settings.biasreg;
    segmentation.matlabbatch{1}.spm.spatial.preproc.channel(1).biasfwhm = settings.biasfwhm;
    segmentation.matlabbatch{1}.spm.spatial.preproc.channel(1).write = settings.write;
    segmentation.matlabbatch{1}.spm.spatial.preproc.channel(1).vols = {fullfile(structural_fn)};
    segmentation.matlabbatch{1}.spm.spatial.preproc.channel(2).biasreg = settings.biasreg;
    segmentation.matlabbatch{1}.spm.spatial.preproc.channel(2).biasfwhm = settings.biasfwhm;
    segmentation.matlabbatch{1}.spm.spatial.preproc.channel(2).write = settings.write;
    segmentation.matlabbatch{1}.spm.spatial.preproc.channel(2).vols = {fullfile(flair_fn)};
end
% Tissue
ngaus  = [1 1 2 3 4 2];
native = [1 1 1 1 1 1];
for c = 1:6 % tissue class c
    segmentation.matlabbatch{1}.spm.spatial.preproc.tissue(c).tpm = {
        fullfile(spm('dir'), 'tpm', sprintf('TPM.nii,%d', c))};
    segmentation.matlabbatch{1}.spm.spatial.preproc.tissue(c).ngaus = ngaus(c);
    segmentation.matlabbatch{1}.spm.spatial.preproc.tissue(c).native = [native(c) 0];
    segmentation.matlabbatch{1}.spm.spatial.preproc.tissue(c).warped = [0 0];
end
% Warp
segmentation.matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;
segmentation.matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
segmentation.matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
segmentation.matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
segmentation.matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
segmentation.matlabbatch{1}.spm.spatial.preproc.warp.samp = 3;
segmentation.matlabbatch{1}.spm.spatial.preproc.warp.write=[0 0];
% Run
spm_jobman('run', segmentation.matlabbatch);
% Save filenames
[d, f, e] = fileparts(structural_fn);
output.forward_transformation = [d filesep 'y_' f e];
output.inverse_transformation = [d filesep 'iy_' f e];
output.gm_fn = [d filesep 'c1' f e];
output.wm_fn = [d filesep 'c2' f e];
output.csf_fn = [d filesep 'c3' f e];
output.bone_fn = [d filesep 'c4' f e];
output.soft_fn = [d filesep 'c5' f e];
output.air_fn = [d filesep 'c6' f e];
disp('SPM Probability maps done!');
end