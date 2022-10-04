function plot_ss_results(slice_i, fig_fn, vol, brain_mask, skull_mask, skull_stripped)
% Plot the required figures for 1 slice in skull stripping part
figure;
t = tiledlayout(1,4);

nexttile
imshow(uint8(vol(:, :, slice_i)));
title("Original Slice");

nexttile
imagesc(skull_mask(:, :, slice_i));
colormap gray
axis square
axis off
title("Skull Mask");

nexttile
imagesc(brain_mask(:, :, slice_i));
colormap gray
axis square
axis off
title("Brain Mask");

nexttile
imshow(uint8(skull_stripped(:, :, slice_i)));
colormap gray
axis square
axis off
title("Skull-stripped");

exportgraphics(t,fig_fn)
end
