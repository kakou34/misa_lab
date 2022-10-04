function output = spm_seg(struct_fns, settings)
% Function to segment brain MRI data from a single subject using Matlab/SPM12.
% Steps include coregistering structural image to first functional image,
% segmenting the coregistered structural image into tissue types.
%
% INPUT:
% struct_fns         - structure of available channels
% settings           - structure with parameters for bias correction
%   biasreg           - regularization parameter for bias field correction
%   biasfwhm          - fwhm parameter for bias field correction
%   write             - writing mode
% 
% OUTPUT: 
% output            - structure with filenames of the generated volumes

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
if length(fieldnames(struct_fns)) == 1
    segmentation.matlabbatch{1}.spm.spatial.preproc.channel.biasreg = settings.biasreg; %bias regularization parameter
    segmentation.matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = settings.biasfwhm; %bias fhwm parameter
    segmentation.matlabbatch{1}.spm.spatial.preproc.channel.write = settings.write; %save field & corrected
    segmentation.matlabbatch{1}.spm.spatial.preproc.channel.vols = {fullfile(struct_fns.ch1)};
else
    for i=1:2
        segmentation.matlabbatch{1}.spm.spatial.preproc.channel(i).biasreg = settings.biasreg;
        segmentation.matlabbatch{1}.spm.spatial.preproc.channel(i).biasfwhm = settings.biasfwhm;
        segmentation.matlabbatch{1}.spm.spatial.preproc.channel(i).write = settings.write;
    end
    segmentation.matlabbatch{1}.spm.spatial.preproc.channel(1).vols = {fullfile(struct_fns.ch1)};
    segmentation.matlabbatch{1}.spm.spatial.preproc.channel(2).vols = {fullfile(struct_fns.ch2)};
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
[d, f, e] = fileparts(struct_fns.ch1);
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