function [res_seg, brain_mask, skull_mask, skull_stripped, corrected] = ...
                segment_brain_tissues(struct_fns, settings, out_fn)
    % Function to segment T1/T2 FLAIR MRI data from a single subject using Matlab/SPM12.
    % struct_fns         - structure of available channels
    % settings           - structure with parameters for bias correction
    %   biasreg           - regularization parameter for bias field correction
    %   biasfwhm          - fwhm parameter for bias field correction
    %   write             - writing mode
    % 
    % OUTPUT: 
    % res_seg            - segmentation result for CSF, GM, WM
    % brain_mask         - brain tissues only mask
    % skull_mask         - bone mask
    % skull_stripped     - original volume without the skull and soft tissue
    % corrected          - brain volume after bias field correction
    
    % segement tissues
    result = spm_seg(struct_fns, settings);
    
    % read tissue probability maps
    gm = niftiread(result.gm_fn(1:end-2));
    wm = niftiread(result.wm_fn(1:end-2));
    csf = niftiread(result.csf_fn(1:end-2));
    bone = niftiread(result.bone_fn(1:end-2));
    soft = niftiread(result.soft_fn(1:end-2));
    air = niftiread(result.air_fn(1:end-2));
    corrected = niftiread(result.correct_fn(1:end-2));
        
    % classify the tissues according to maximum posterior probabilies
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
    
    % remove skull from original
    struct_fn = struct_fns.ch1;
    vol = niftiread(struct_fn(1:end-2)); % reading the original scan
    skull_stripped = vol.* brain_mask;
    
    % Return clean segmentation of CSF, GM, WM
    res_seg(res_seg>3) = 0;
    
    % Save results with correct spacing
    template_fn = [struct_fns.ch1(1:end-2)];
    template_spm = spm_vol(template_fn);
    new_nii = spm_create_vol(template_spm);
    new_nii.fname = out_fn;
    spm_write_vol(new_nii, res_seg);
end

