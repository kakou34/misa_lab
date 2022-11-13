import time
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from src import ExpectationMaximization
from typing import List
from scipy.stats import mode


def min_max_norm(
    img: np.ndarray, max_val: int = None, mask: np.ndarray = None, dtype: str = None
):
    """
    Scales images to be in range [0, 2**bits]

    Args:
        img (np.ndarray): Image to be scaled.
        max_val (int, optional): Value to scale images
            to after normalization. Defaults to None.
        mask (np.ndarray, optional): Mask to use in the normalization process.
            Defaults to None which means no mask is used.
        dtype (str, optional): Output datatype

    Returns:
        np.ndarray: Scaled image with values from [0, max_val]
    """
    if mask is not None:
        img_ = img.copy()
        img = img[mask != 0].flatten()
    if max_val is None:
        max_val = np.iinfo(img.dtype).max
    img = (img - img.min()) / (img.max() - img.min()) * max_val
    if mask is not None:
        img_[mask != 0] = img.copy()
        img = img_.copy()
    if dtype is not None:
        return img.astype(dtype)
    else:
        return img


def match_pred_w_gt(
    prediction: np.ndarray, gt_mask: np.ndarray, t1_array: np.ndarray, t2_array: np.ndarray = None
) -> np.ndarray:
    """
    Relabels the prediction volume to match the labels names in the ground truth. This is done
    by comparing the mean features (intensities, single or multi modality) inside the masks of each
    label between ground truth and predictions.
    Args:
        prediction (np.ndarray): Categorical volume with prediction results. Background must be zero
        gt_mask (np.ndarray): Categorical volume with ground truth labels. Background must be zero
        t1_array (np.ndarray): T1 volume
        t2_array (np.ndarray, optional): T2 volume, is not available just ignore it.
            Defaults to None.
    Returns:
        np.ndarray: Categorical volume with corrected labels.
    """

    # Initialize the container
    matched_predictions = np.zeros_like(prediction)
    original_shape = prediction.shape

    # Reshape the images for cleaner code
    prediction = prediction.flatten()
    gt_mask = gt_mask.flatten()

    # Include of not t2 volume data
    if t2_array is not None:
        data = np.array([t1_array.flatten(), t2_array.flatten()]).T
        n_features = 2
    else:
        n_features = 1
        data = t1_array.flatten()
        data = data.reshape(data.shape[0], -1)

    # Get the available labels
    predicted_labels = np.unique(prediction[prediction != 0])
    n_components = len(predicted_labels)

    # Get means of tissue intensities in each class under gt or prediction masks
    means_pred = np.zeros((n_components, n_features))
    means_gt = np.zeros((n_components, n_features))
    for label in range(n_components):
        means_pred[label, :] = np.meadian(data[prediction == (label+1), :], axis=0)
        means_gt[label, :] = np.meadian(data[gt_mask == (label+1), :], axis=0)

    # Compare the mean intensity of each class (euclidean distance)
    dists = np.zeros((n_components, n_components))
    for i in range(n_components):
        for j in range(n_components):
            dists[i, j] = np.linalg.norm(means_pred[i, :] - means_gt[j, :])

    # Match classes according to closer mean intensity value
    gt_match = np.argmin(dists, axis=1)
    prediction = prediction.reshape(original_shape)

    # Relabel image
    for i in range(n_components):
        matched_label = gt_match[i] + 1
        matched_predictions[prediction == i+1] = matched_label

    return matched_predictions


def match_labels(seg: np.ndarray, gt: np.ndarray, prob_array: np.ndarray = None):
    """
    Matches the labels numbers based on the counts of voxels inside the masks
        deifned by gt labels
    Args:
        seg (np.ndarray): segmentation results from em
        gt (np.ndarray): gt labels
        prob_array (np.ndarray, optional): posterior probs array used inside em.
            Defaults to None.
    """
    shape = seg.shape
    seg = seg.flatten()
    gt = gt.flatten()
    order = {}
    for label in [0, 2, 3]:
        labels, counts = np.unique(seg[gt == label], return_counts=True)
        order[label] = labels[np.argmax(counts)]
    order[1] = [i for i in [0, 1, 2, 3] if i not in list(order.values())][0]
    seg_ = seg.copy()
    prob_array_ = prob_array.copy() if prob_array is not None else None
    for des_val, seg_val in order.items():
        seg[seg_ == seg_val] = des_val
        if des_val in [1, 2, 3]:
            if prob_array_ is not None:
                prob_array[:, des_val-1] = prob_array_[:, seg_val-1]
    if prob_array is not None:
        return seg.reshape(shape), prob_array
    else:
        return seg.reshape(shape)


def brain_tissue_segmentation_em(
    t1: np.ndarray, brain_mask: np.ndarray, tissue_models: np.ndarray = None,
    mean_init: str = 'kmeans', atlas_use: str = None, atlas_map: np.ndarray = None,
    previous_result: np.ndarray = None
):
    """Computes brain tissue segmentation in single approach
    Args:
        t1 (np.ndarray): T1 volume
        brain_mask (np.ndarray): bask of brain tissues (region to classify)
        mean_init (str, optional): How to initialize the means in EM.
            Defaults to 'kmeans'.
        tissue_models (np.ndarray, optional): tissue models to use for initialization.
            Defaults to None, if used, mean_init should be equal to 'tissue_models'
        atlas_use (str, optional): If atlas provided, how to use it in EM. Defaults to None.
        atlas_map (np.ndarray, optional): Atlas. Defaults to None.
        previous_result (np.ndarray, optional): Result from EM part for "after" use of atlases.
            If provided just the last multiplication is done. Defaults to None.
    """
    # Define data
    t1_vector = t1[brain_mask == 255].flatten()
    if atlas_map is not None:
        atlas_map_vector = atlas_map[:, brain_mask == 255].reshape(atlas_map.shape[0], -1)
        atlas_map_vector = atlas_map_vector[1:, :]
    else:
        atlas_map_vector = None
    data = np.array(t1_vector)[:, np.newaxis]

    # Define model
    model = ExpectationMaximization(
        n_components=3,
        mean_init=mean_init,
        verbose=False,
        plot_rate=None,
        tissue_models=tissue_models,
        atlas_use=atlas_use,
        atlas_map=atlas_map_vector,
        previous_result=previous_result
    )

    # Run
    start = time.time()
    preds_prob, preds_categorical = model.fit_predict(data)
    t1_time = time.time() - start
    predictions_categorical = brain_mask.flatten()
    predictions_categorical[predictions_categorical == 255] = preds_categorical + 1
    t1_seg_res_cat = predictions_categorical.reshape(t1.shape)
    t1_iters = model.n_iter_
    return preds_prob, t1_seg_res_cat, t1_time, t1_iters


def brain_tissue_segmentation_tm(
    t1: np.ndarray, brain_mask: np.ndarray, tissue_models: np.ndarray
):
    """Computes brain tissue segmentation in single approach
    Args:
        t1 (np.ndarray): T1 volume
        brain_mask (np.ndarray): bask of brain tissues (region to classify)
        tissue_models (np.ndarray, optional): tissue models to use for segmentation.
    """
    # Define data
    t1_vector = t1[brain_mask == 255].flatten()
    n_classes = tissue_models.shape[0]
    preds = np.zeros((n_classes, len(t1_vector)))
    for c in range(n_classes):
        preds[c, :] = tissue_models[c, t1_vector]
    preds = np.argmax(preds, axis=0)
    predictions = brain_mask.flatten()
    predictions[predictions == 255] = preds + 1
    t1_seg_res = predictions.reshape(t1.shape)
    return t1_seg_res


def brain_tissue_segmentation_prob_map(
    brain_mask: np.ndarray, tissue_prob_maps: np.ndarray
):
    """Computes brain tissue segmentation in single approach
    Args:
        t1 (np.ndarray): T1 volume
        brain_mask (np.ndarray): bask of brain tissues (region to classify)
        tissue_prob_maps (np.ndarray, optional): size -> [n_class, [volume_shape]]
    """
    # Define data
    pred = np.argmax(tissue_prob_maps, axis=0)
    t1_seg_res = np.where(brain_mask != 255, 0, pred)
    return t1_seg_res


def brain_tissue_segmentation_tm_prob_map(
    t1: np.ndarray, brain_mask: np.ndarray,
    tissue_models: np.ndarray, tissue_prob_maps: np.ndarray
):
    """Computes brain tissue segmentation in single approach
    Args:
        t1 (np.ndarray): T1 volume
        brain_mask (np.ndarray): bask of brain tissues (region to classify)
        tissue_models (np.ndarray, optional): tissue models to use for segmentation.
        tissue_prob_maps (np.ndarray, optional): size -> [n_class, [volume_shape]]
    """
    # Define datas
    brain_mask = brain_mask.flatten()
    t1_vector = t1.flatten()[brain_mask == 255]
    prob_vects = tissue_prob_maps.reshape((tissue_prob_maps.shape[0], -1))
    prob_vects = prob_vects[1:, :]
    prob_vects = prob_vects[:, brain_mask == 255]

    n_classes = tissue_models.shape[0]
    preds = np.zeros((n_classes, len(t1_vector)))
    for c in range(n_classes):
        preds[c, :] = tissue_models[c, :][t1_vector]
    preds *= prob_vects
    preds = np.argmax(preds, axis=0)

    predictions = brain_mask.copy()
    predictions[brain_mask == 255] = preds + 1
    t1_seg_res = predictions.reshape(t1.shape)
    return t1_seg_res


def dice_score(gt: np.ndarray, pred: np.ndarray):
    """Compute dice across classes. The corresponding labels should be
    previously matched.
    Args:
        gt (np.ndarray): Grounth truth
        pred (np.ndarray): Labels
    Returns:
        list: Dice scores per tissue [CSF, GM, WM]
    """
    dice = np.zeros((3))
    for i in [1, 2, 3]:
        bin_pred = np.where(pred == i, 1, 0)
        bin_gt = np.where(gt == i, 1, 0)
        dice[i-1] = np.sum(bin_pred[bin_gt == 1]) * 2.0 / (np.sum(bin_pred) + np.sum(bin_gt))
    return dice.tolist()


def plots_debug(volumes: List[np.ndarray], names: List[str], slice_n: int = 20):
    """Generates plots of all volumes with the corresponding names of the volume plotted
    Args:
        volumes (List[np.ndarray]): List on np.ndarrays to plot
        names (List[str]): List of string with the volumes names
        slice_n (int, optional): Axial slice N° to plot. Defaults to 20.
    """
    _, ax = plt.subplots(1, 6, figsize=(18, 8))
    for j, (vol, name) in enumerate(zip(volumes[0:2], ['T1', 'GT'])):
        cmap = 'gray' if (j == 0) else 'viridis'
        ax[j].set_title(name)
        ax[j].imshow(vol[slice_n, :, :], cmap=cmap)
        ax[j].set_xticks([])
        ax[j].set_yticks([])
    for j, (vol, name) in enumerate(zip(volumes[2:], names[2:]), start=2):
        ax[j].set_title(name)
        ax[j].imshow(vol[slice_n, :, :], cmap='viridis', vmin=0, vmax=3)
        ax[j].set_xticks([])
        ax[j].set_yticks([])
    plt.show()


def plots_by_case(volumes: List[np.ndarray], names: List[str], slice_n: int = 20):
    """Generates plots of all volumes with the corresponding names of the volume plotted
    Args:
        volumes (List[np.ndarray]): List on np.ndarrays to plot
        names (List[str]): List of string with the volumes names
        slice_n (int, optional): Axial slice N° to plot. Defaults to 20.
    """
    n = len(names[2:])
    n_rows = int(np.ceil((n)/7))
    n_cols = int(np.ceil((n)/n_rows) + 2)
    _, ax = plt.subplots(n_rows, n_cols, figsize=(18, 8))
    for i in range(n_rows):
        for j, (vol, name) in enumerate(zip(volumes[0:2], ['T1', 'GT'])):
            cmap = 'gray' if (j == 0) else 'viridis'
            ax[i][j].set_title(name)
            ax[i][j].imshow(vol[slice_n, :, :], cmap=cmap)
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
    for j, (vol, name) in enumerate(zip(volumes[2:], names[2:])):
        i = j // 7
        j = 2 + j % 7
        ax[i][j].set_title(name)
        ax[i][j].imshow(vol[slice_n, :, :], cmap='viridis')
        ax[i][j].set_xticks([])
        ax[i][j].set_yticks([])
    plt.show()


def save_segementations(
    volume: np.ndarray, reference: sitk.Image, filepath: Path
):
    """Stores the volume in nifty format using the spatial parameters coming
        from a reference image
    Args:
        volume (np.ndarray): Volume to store as in Nifty format
        reference (sitk.Image): Reference image to get the spatial parameters from.
        filepath (Path): Where to save the volume.
    """
    # Save image
    if (type(volume) == list) or (len(volume.shape) > 3):
        if type(volume[0]) == sitk.Image:
            vol_list = [vol for vol in volume]
        else:
            vol_list = [sitk.GetImageFromArray(vol) for vol in volume]
        joiner = sitk.JoinSeriesImageFilter()
        img = joiner.Execute(*vol_list)
    else:
        img = sitk.GetImageFromArray(volume)
    img.SetDirection(reference.GetDirection())
    img.SetOrigin(reference.GetOrigin())
    img.SetSpacing(reference.GetSpacing())
    for key in reference.GetMetaDataKeys():
        img.SetMetaData(key, reference.GetMetaData(key))
    sitk.WriteImage(img, filepath)


def complete_figure(
    img_path: Path, labels_path: Path, segs_path: Path,
    cases: List[str], algo_list: List[str], slice_n: int = 25
):
    """
    Plot all segementations in plots of 7xn_cases. First row is t1 and second
    ground truth
    Args:
        img_path (Path): t1 path
        labels_path (Path): ground_truth path
        segs_path (Path): directory containing all segementations
        cases (List[str]): list of case names to use in the plot
        algo_list (List[str]): list of segementation algo_names to include
        slice_n (int, optional): Axial slice to plot. Defaults to 25.
    """
    n_figures = np.ceil(algo_list / 5)
    for n in range(n_figures):
        n_rows, n_cols = 7, len(cases)
        _, ax = plt.subplots(n_rows, n_cols, figsize=(18, 24))
        for i, path, name in enumerate(zip([img_path, labels_path], ['T1', 'GT'])):
            for j, case in enumerate(cases):
                cmap = 'gray' if (j == 0) else 'viridis'
                img_name = f'{case}.nii.gz' if i == 0 else f'{case}_3C.nii.gz'
                img = sitk.ReadImage(str(path / img_name))
                img_array = sitk.GetArrayFromImage(img)
                ax[i][j].set_title(name)
                ax[i][j].imshow(img_array[slice_n, :, :], cmap=cmap)
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])
        for i, name in enumerate(2, algo_list[n*5:(n+1)*5]):
            for j, case in enumerate(cases):
                img_name = f'{case}_{name}.nii.gz'
                img = sitk.ReadImage(str(segs_path / img_name))
                img_array = sitk.GetArrayFromImage(img)
                title = (name.upper()).replace('_', '-')
                ax[i][j].set_title(title)
                ax[i][j].imshow(img_array[slice_n, :, :], cmap='viridis')
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])
        plt.show()
