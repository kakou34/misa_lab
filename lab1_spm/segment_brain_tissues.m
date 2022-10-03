function res_seg = segment_brain_tissues(structural_fn, flair_fn, settings)
% Function to segment T1 MRI data from a single subject using Matlab/SPM12.
% structural_fn      - filename of T1-weighted structural scan
% settings           - configuration settings for the segmentation pipeline
% 
% OUTPUT: 
% output            - structure with filenames and data

% Obtain segmentation posterior probability maps
result = spm_seg(structural_fn, flair_fn, settings);

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

% Ignore bone, soft tissue and air
res_seg(res_seg>3) = 0;

% Save results with correct spacing
template_fn = [structural_fn];
template_spm = spm_vol(template_fn);
new_nii = spm_create_vol(template_spm);
new_nii.fname = 'resulting_segmetation.nii';
spm_write_vol(new_nii, res_seg);
end

