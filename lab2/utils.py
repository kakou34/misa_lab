import time
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from sklearn.cluster import KMeans
from src import ExpectationMaximization
from typing import List


def min_max_norm(img: np.ndarray, max_val: int = None, dtype: str = None):
    """
    Scales images to be in range [0, 2**bits]

    Args:
        img (np.ndarray): Image to be scaled.
        max_val (int, optional): Value to scale images
            to after normalization. Defaults to None.
        dtype (str, optional): Output datatype

    Returns:
        np.ndarray: Scaled image with values from [0, max_val]
    """
    if max_val is None:
        max_val = np.iinfo(img.dtype).max
    img = (img - img.min()) / (img.max() - img.min()) * max_val
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
        data = t1_array.flatten()[:, np.newaxis]

    # Get the available labels
    predicted_labels = np.unique(prediction[prediction != 0])
    n_components = len(predicted_labels)

    # Get means of tissue intensities in each class under gt or prediction masks
    means_pred = np.zeros((n_components, n_features))
    means_gt = np.zeros((n_components, n_features))
    for label in range(n_components):
        means_pred[label, :] = np.mean(data[prediction == label+1, :], axis=0)
        means_gt[label, :] = np.mean(data[gt_mask == label+1, :], axis=0)

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


def brain_tissue_segmentation(
    t1: np.ndarray, t2: np.ndarray, brain_mask: np.ndarray,
    mode: str, mean_init='kmeans', seed=420
):
    """Compute brain tissue segmentation in single and multichannel approach
    Args:
        t1 (np.ndarray): T1 volume
        t2 (np.ndarray): T2_FLAIR volume
        brain_mask (np.ndarray): bask of brain tissues (region to classify)
        mode (str): 'em' or 'km' denoting Exp. Maximization or K-means
        mean_init (str, optional): How to initialize the means in EM.
            Defaults to 'kmeans'.
        seed (int, optional): To guarantee reproducibility. Defaults to 420.
    """
    # T1+T2
    t1_vector = t1[brain_mask == 255].flatten()
    t2_vector = t2[brain_mask == 255].flatten()
    data = np.array([t1_vector, t2_vector]).T
    if mode == 'em':
        model = ExpectationMaximization(
            n_components=3, mean_init=mean_init, priors='non_informative',
            verbose=False, plot_rate=None, seed=seed
        )
    else:
        model = KMeans(n_clusters=3, random_state=420)
    start = time.time()
    preds = model.fit_predict(data)
    t_t1_t2 = time.time() - start
    predictions = brain_mask.flatten()
    predictions[predictions == 255] = preds + 1
    t1_t2_seg_res = predictions.reshape(t1.shape)
    t1_t2_iters = model.n_iter_

    # T1
    data = np.array(t1_vector)[:, np.newaxis]
    if mode == 'em':
        model = ExpectationMaximization(
            n_components=3, mean_init=mean_init, priors='non_informative',
            verbose=False, plot_rate=None, seed=seed
        )
    else:
        model = KMeans(n_clusters=3, random_state=420)
    start = time.time()
    preds = model.fit_predict(data)
    t_t1 = time.time() - start
    predictions = brain_mask.flatten()
    predictions[predictions == 255] = preds + 1
    t1_seg_res = predictions.reshape(t1.shape)
    t1_iters = model.n_iter_
    return t1_seg_res, t1_t2_seg_res, t_t1, t_t1_t2, t1_iters, t1_t2_iters


def dice_score(gt: np.ndarray, pred: np.ndarray):
    """Compute dice across classes. The corresponding labels should be
    previously matched.
    Args:
        gt (np.ndarray): Grounth truth
        pred (np.ndarray): Labels
    Returns:
        list: Dice scores per tissue [CSF, GM, WM]
    """
    classes = np.unique(gt[gt != 0])
    dice = np.zeros((len(classes)))
    for i in classes:
        bin_pred = np.where(pred == i, 1, 0)
        bin_gt = np.where(gt == i, 1, 0)
        dice[i-1] = np.sum(bin_pred[bin_gt == 1]) * 2.0 / (np.sum(bin_pred) + np.sum(bin_gt))
    return dice.tolist()


def plots(volumes: List[np.ndarray], names: List[str], slice_n: int = 20):
    """Generates plots of all volumes with the corresponding names of the volume plotted
    Args:
        volumes (List[np.ndarray]): List on np.ndarrays to plot
        names (List[str]): List of string with the volumes names
        slice_n (int, optional): Axial slice N° to plot. Defaults to 20.
    """
    n = len(volumes)
    fig, ax = plt.subplots(1, n, figsize=(20, 5))
    for i in range(n):
        cmap = 'gray' if len(np.unique(volumes[i])) > 4 else 'viridis'
        ax[i].set_title(names[i])
        ax[i].imshow(volumes[i][slice_n, :, :], cmap=cmap)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
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
    img = sitk.GetImageFromArray(volume)
    img.SetDirection(reference.GetDirection())
    img.SetOrigin(reference.GetOrigin())
    img.SetSpacing(reference.GetSpacing())
    sitk.WriteImage(img, filepath)


def complete_figure(data_path: Path, img_names: List[Path], ylabels: List[str], slice_n: int = 25):
    """Plots a huge figure with all image names from all subjects.
    Args:
        data_path (Path): Path to P2_data folder.
        img_names (List[Path]): file paths to read the images from
        ylabels (List[str]): Modality / Algorithm
        slice_n (int, optional): Axial slice to plot. Defaults to 25.
    """
    fig, ax = plt.subplots(len(img_names), 5, figsize=(10, 14))
    for i in range(5):
        for j, img_name in enumerate(img_names):
            img = sitk.ReadImage(data_path/f'{i+1}/{img_name}')
            img_array = sitk.GetArrayFromImage(img)
            if j == len(img_names)-1:
                ax[j, i].set_xlabel(f'Subject {i+1}')
            if j in [0, 1]:
                ax[j, i].imshow(img_array[slice_n, :, :], cmap='gray')
            else:
                ax[j, i].imshow(img_array[slice_n, :, :])
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
            if i == 0:
                ax[j, i].set_ylabel(ylabels[j])
    plt.show()
