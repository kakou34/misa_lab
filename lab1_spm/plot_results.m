function [] = plot_results(volume, labels, res_seg)
%plot some results
figure;
subplot(3,3,1)
imagesc(volume(:, :, 20));
colormap gray
axis square off
title("Slice 20");

subplot(3,3,2)
imagesc(labels(:, :, 20));
colormap summer
axis square off
title("Ground Truth");

subplot(3,3,3)
imagesc(res_seg(:, :, 20));
colormap gray
axis square off
title("Result");

subplot(3,3,4)
imagesc(rot90(reshape(volume(:, 100, :), [240, 48])));
colormap gray
axis square off
title("Slice 100");

subplot(3,3,5)
imagesc(rot90(reshape(labels(:, 100, :), [240, 48])));
colormap gray
axis square off
title("Ground Truth");

subplot(3,3,6)
imagesc(rot90(reshape(res_seg(:, 100, :), [240, 48])));
colormap gray
axis square off
title("Result");

subplot(3,3,7)
imagesc(rot90(reshape(volume(125, :, :), [240, 48])));
colormap gray
axis square off
title("Slice 58");

subplot(3,3,8)
imagesc(rot90(reshape(labels(125, :, :), [240, 48])));
colormap gray
axis square off
title("Ground Truth");

subplot(3,3,9)
imagesc(rot90(reshape(res_seg(125, :, :), [240, 48])));
colormap gray
axis square off
title("Ground Truth");
end

