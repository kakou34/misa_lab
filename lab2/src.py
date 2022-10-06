from statistics import covariance
import numpy as np
from typing import Union
from psutil import net_connections
from sklearn.cluster import KMeans
import scipy


def expectation(data: np.ndarray, mean: np.ndarray, cov_mat: np.ndarray, priors: np.ndarray):
    glh = gaussian_likelihood(data, mean, cov_mat)
    num = glh * np.repeat(priors.T, 20, 0)
    denom = np.sum(num, 1)
    membership_weight = num / np.repeat(np.expand_dims(denom,1), num.shape[1], 1)
    return membership_weight


def gaussian_likelihood(
    x: np.ndarray, mean: np.ndarray, cov_mat: np.ndarray
):
    n_components, n_feats = mean.shape
    n_samples = x.shape[0]
    constant = 1/(((2*np.pi) ** n_feats/2) * (np.linalg.det(cov_mat) ** 1/2))

    likelihood = np.zeros((n_samples, n_components))
    for i in range(mean.shape[0]):
        dif = x - mean[i]
        if (np.linalg.det(cov_mat[i, :, :]) != 0):
            mahalanobis_dist = np.diagonal(np.dot(np.dot(dif, np.linalg.inv(cov_mat[i, :, :])), dif.T))
        else:
            mahalanobis_dist = np.diagonal(np.dot(np.dot(dif, np.linalg.pinv(cov_mat[i, :, :])), dif.T))
        likelihood[:, i] = constant[i] * np.exp(-(1/2)*mahalanobis_dist)
    return likelihood


def maximization(x: np.ndarray, weights: np.ndarray):
    predictions = np.argmax(weights, 1)
    n_components = weights.shape[1]
    
    counts = np.zeros((n_components, 1))
    for i in range(n_components):
        counts[i] = np.count_nonzero(predictions == i)
    new_priors = counts / len(x)
    new_mean = np.dot(weights.T, x) / np.repeat(counts, 2, 1)
    
    for i in range(n_components):
        centered_x = x - new_mean[i]
        sigma = np.dot(centered_x.T, centered_x)

    return


def expectation_maximization(
    data: np.ndarray,
    n_components: int,
    mean_init: Union[str, np.ndarray] = 'random',  # array, 'k-means'
    priors: Union[str, np.ndarray] = 'non_informative',
    max_iter: int = 10000,
    change_tol: float = 0.001,
    seed: float = 420
):
    n_samples, n_feat = data.shape
    if isinstance(priors, str) and (priors == 'non_informative'):
        priors = np.ones((n_components, n_feat)) / n_components

    if isinstance(mean_init, str) and (mean_init == 'radom'):
        rng = np.random.default_rng(seed=seed)
        mean = rng.integers(0, 255, (n_components, n_feat))
    elif isinstance(mean_init, str) and (mean_init == 'kmeans'):
        kmeans = KMeans(n_clusters=n_components, random_state=seed).fit(data)
        mean = kmeans.cluster_centers_
    # TODO: Add mean shiftss

    n_samples, n_feat = data.shape
    # labels = np.zeros((n_samples, 1))
    # covariance = np.zeros((n_components, 1))
    cov_mat = np.cov(data)
    cov_mat = np.repeat(np.expand_dims(cov_mat, 0), n_components)
    for _ in range(max_iter):
        expectation(mean, cov_mat)
        maximization()
        tol = abs(prev_loglh - loglh)
        if tol < change_tol: 
    return

#TODO:
# benchmark: K-means
# innitialization: random vs k-means
# modalities: T1-w vs T1-w+FLAIR
# metrics: DSC, num iterations, comp.time

