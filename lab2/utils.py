import numpy as np


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
        t2_array (np.ndarray, optional): T2 volume, is not available just ignore it. Defaults to None.
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
            print(np.linalg.norm(means_pred[i, :] - means_gt[j, :]))
            dists[i, j] = np.linalg.norm(means_pred[i, :] - means_gt[j, :])

    # Match classes according to closer mean intensity value
    gt_match = np.argmin(dists, axis=1)
    prediction = prediction.reshape(original_shape)

    # Relabel image
    for i in range(n_components):
        matched_label = gt_match[i] + 1
        matched_predictions[prediction == i+1] = matched_label

    return matched_predictions
