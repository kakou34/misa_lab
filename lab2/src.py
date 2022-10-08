import logging
from os import replace
import numpy as np
from typing import Union, Tuple
from sklearn.cluster import KMeans, MeanShift
# from numba import njit


# logging.basicConfig(level=print, format='%(message)s')


def our_multivariate_normal_pdf(x, means, sigmas):
    n_feats = len(means)
    constant = 1/(((2*np.pi) ** n_feats/2) * (np.linalg.det(sigmas) ** 1/2))
    dif = x - means
    if (np.linalg.det(sigmas) != 0):
        mahalanobis_dist = np.sum(
            np.dot(dif, np.linalg.inv(sigmas)) * dif[:, ::-1], 1)
    else:
        mahalanobis_dist = np.sum(
            np.dot(dif, np.linalg.pinv(sigmas)) * dif[:, ::-1], 1)
    return constant * np.exp(-(1/2)*mahalanobis_dist)



# @njit()
def expectation(
    x: np.ndarray, mean: np.ndarray, sigmas: np.ndarray, priors: np.ndarray
) -> np.ndarray:
    glh = gaussian_likelihood(x, mean, sigmas)
    num = glh * np.repeat(priors.T, 20, 0)
    denom = np.sum(num, 1)
    weights = num / np.repeat(np.expand_dims(denom, 1), num.shape[1], 1)
    return weights, gaussian_likelihood


# @njit()
def gaussian_likelihood(
    x: np.ndarray, mean: np.ndarray, sigmas: np.ndarray
) -> np.ndarray:
    n_components, n_feats = mean.shape
    n_samples = x.shape[0]
    likelihood = np.zeros((n_samples, n_components))
    for i in range(mean.shape[0]):
        constant = 1/(((2*np.pi) ** n_feats/2) * (np.linalg.det(sigmas[i, :, :]) ** 1/2))
        dif = x - mean[i]
        if (np.linalg.det(sigmas[i, :, :]) != 0):
            mahalanobis_dist = np.diagonal(
                np.dot(np.dot(dif, np.linalg.inv(sigmas[i, :, :])), dif.T))
        else:
            mahalanobis_dist = np.diagonal(
                np.dot(np.dot(dif, np.linalg.pinv(sigmas[i, :, :])), dif.T))
        likelihood[:, i] = constant * np.exp(-(1/2)*mahalanobis_dist)
    return likelihood


# @njit()
def maximization(
    x: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples, n_feat = x.shape
    predictions = np.argmax(weights, 1)
    n_components = weights.shape[1]

    counts = np.zeros((n_components, 1))
    for i in range(n_components):
        counts[i] = np.count_nonzero(predictions == i)
    new_priors = counts / len(x)
    new_means = np.dot(weights.T, x) / np.repeat(counts, 2, 1)

    new_sigmas = np.zeros(n_components, n_feat, n_feat)
    for i in range(n_components):
        centered_x = x - new_means[i]
        new_sigmas[i, :, :] = np.dot((weights * centered_x).T, centered_x) / counts[i]

    return new_means, new_sigmas, new_priors


# @njit()
def expectation_maximization(x: np.ndarray, means, sigmas, priors, max_iter, change_tol):
    prev_log_lkh = 0
    for _ in range(max_iter):
        # E-step
        weights, likelihood = expectation(x, means, sigmas, priors)

        # Stop criteria: tolerance
        log_lkh = np.sum(np.log(likelihood))
        tol = abs(prev_log_lkh - log_lkh)
        if tol < change_tol:
            break

        # M-Step
        means, sigmas, priors = maximization(x, weights)
    return means, sigmas, priors


class ExpectationMaximization():
    def __init__(
        self,
        n_components: int = 3,
        mean_init: Union[str, np.ndarray] = 'random',  # array, 'k-means'
        priors: Union[str, np.ndarray] = 'non_informative',
        max_iter: int = 10000,
        change_tol: float = 0.001,
        seed: float = 420,
        verbose: bool = False
    ) -> None:

        self.n_components = n_components
        self.mean_init = mean_init
        self.priors = priors
        self.change_tol = change_tol
        self.max_iter = max_iter
        self.seed = seed
        self.verbose = verbose

        # Check kind of priors to be used
        condition_one = isinstance(priors, str) and (priors not in ['non_informative'])
        condition_two = isinstance(priors, np.ndarray) and (priors.size != n_components)
        if condition_one or condition_two:
            raise Exception(
                "Priors must be either 'non_informative' or an array of "
                "'n_components' elements"
            )

        # Check kind of means to be used
        mean_options = ['radom', 'kmeans', 'mean_shifts']
        condition_one = isinstance(mean_init, str) and (mean_init not in mean_options)
        condition_two = isinstance(priors, np.ndarray) and (priors.size != n_components)
        if condition_one or condition_two:
            raise Exception(
                "Priors must be either 'random', 'kmeans', ' mean_shifts', or "
                "an array of 'n_components' rows, and n_features number of columns"
            )

    def fit(self, x: np.ndarray):
        self.x = x
        _, self.n_feat = x.shape

        self.priors_type = 'Provided array'
        if isinstance(self.priors, str) and (self.priors == 'non_informative'):
            self.priors = np.ones((self.n_components, self.n_feat)) / self.n_components
            self.priors_type = 'Non Informative'

        # Define kind of means to be used
        self.mean_type = 'Passed array'
        if isinstance(self.mean_init, str):
            if self.mean_init == 'radom':
                rng = np.random.default_rng(seed=self.seed)
                self.means = rng.integers(0, 255, (self.n_components, self.n_feat))
                self.mean_type = 'Random Init'
            elif self.mean_init == 'kmeans':
                kmeans = KMeans(n_clusters=self.n_components, random_state=self.seed).fit(self.x)
                self.means = kmeans.cluster_centers_
                self.mean_type = 'K-Means'
            else:
                mean_shift = MeanShift().fit(self.x)
                self.means = mean_shift.cluster_centers_
                self.mean_type = 'Mean Shifts'

        # Define initial covariance matrix
        rng = np.random.default_rng(seed=self.seed)
        self.sigmas = np.cov(rng.choice(self.x, int((0.01)*len(x)), replace=False))
        self.sigmas = np.repeat(np.expand_dims(self.sigmas, 0), self.n_components)
        print(self.sigmas.shape)

        # Log initial info
        if self.verbose:
            print('Starting Expectation Maximization Algorithm')
            print(f'Priors type:{self.priors_type} ---- Priors: {self.priors}')
            print(f'Mean type: {self.mean_type} --- Means: {self.means}')

        # Expectation Maximization process
        self.means, self.sigmas, self.priors = expectation_maximization(
            self.x, self.means, self.sigmas, self.priors, self.max_iter, self.change_tol)
        self.cluster_centroids = self.means

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.weights, _ = expectation(x, self.mean, self.sigmas, self.priors)
        self.predictions = np.argmax(self.weights, 1)
        return self.predictions

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        self.predictions = self.predict(self.x)
        return self.predictions


"""
PERFORMANCE TESTS:

modify remaining fors
    try without njit
    try with njit

use fors
    try without njit
    try with njit


TODO:
benchmark: K-means
innitialization: random vs k-means
modalities: T1-w vs T1-w+FLAIR
metrics: DSC, num iterations, comp.time
"""
