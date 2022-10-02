function output = spm_seg(structural_fn, bias_reg, bias_fwhm)
% Function to segment brain MRI data from a single subject using Matlab/SPM12.

% Steps include coregistering structural image to first functional image,
% segmenting the coregistered structural image into tissue types.
% If spm12 batch parameters are not explicitly set, defaults are assumed. 
%
% INPUT:
% structural_fn      - filename of structural scan (T1 or FLAIR)
% 
% OUTPUT: 
% output            - structure with filenames and data

% Declare output structure
output = struct;

% Segmentation of coregistered structural image into GM, WM, CSF, etc
% (with implicit warping to MNI space, saving forward and inverse transformations)
disp('Segmentation of volume starting...');
spm('defaults','fmri');
spm_jobman('initcfg');
segmentation = struct;
% Channel

%% Segmentation
segmentation.matlabbatch{1}.spm.spatial.preproc.channel.biasreg = bias_reg; %bias regularization parameter
segmentation.matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = bias_fwhm; %bias fhwm parameter
segmentation.matlabbatch{1}.spm.spatial.preproc.channel.write = [1 1]; %save field & corrected
segmentation.matlabbatch{1}.spm.spatial.preproc.channel.vols = {fullfile(structural_fn)};
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
output.correct_fn = [d filesep 'm' f e];
output.field_fn = [d filesep 'BiasField_' f e];
disp('SPM Probability maps done!');
end