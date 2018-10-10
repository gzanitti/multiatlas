import numpy as np

from .voting import voting_filter
from warnings import warn

from collections import Counter

from logpar.utils import cifti_utils

def initialize_confusion_matrix_from_atlases(atlases, undetermined=-1):
    atlases = np.atleast_2d(atlases)

    voted_segmentation = voting_filter(atlases)

    msize = atlases.max() + 1
    confusion_matrix = np.zeros((atlases.shape[0], msize, msize + 1),
                                dtype=float)

    for k, a in enumerate(atlases):
        pairs = Counter(zip(a, voted_segmentation))

        pairs_i, pairs_j = np.array(pairs.keys()).T

        confusion_matrix[k, pairs_i, pairs_j] = pairs.values()

    normaliser = confusion_matrix.sum(2)[:, :, None]
    normaliser[normaliser == 0] = 1
    confusion_matrix /= normaliser
    return confusion_matrix


def initialize_prior_probabilities_from_atlases(atlases, undetermined=-1):
    atlases = np.atleast_2d(atlases)

    labels = np.r_[np.unique(atlases), undetermined]
    msize = atlases.max()+1+1  # max + 1 (undeterminated)
    prior_probabilities = np.zeros(msize, dtype=float)

    for label in labels:
        prior_probabilities[label] = (atlases == label).sum()

    prior_probabilities /= prior_probabilities.sum()
    return prior_probabilities


def multi_label_segmentation(atlases, confusion_matrix=None,
                             prior_probabilities=None, iters=100, tol=1e-4):
    # We flatten the atlases and remove the voxels which are zero
    # in all of them
    atlases = np.atleast_2d(atlases)
    labels = np.r_[np.unique(atlases), -1]

    flat_atlases = np.array([a.ravel() for a in atlases])
    all_nzrs = flat_atlases.sum(0).nonzero()[0]
    flat_atlases_filtered = flat_atlases[:, all_nzrs]

    # Compute the confusion and prior_probabilities matrices
    if confusion_matrix is None:
        print("Computing confusion matrix")
        confusion_matrix = initialize_confusion_matrix_from_atlases(atlases)

    if prior_probabilities is None:
        print("Computing prior probabilities")
        prior_probabilities = initialize_prior_probabilities_from_atlases(atlases)

    print("EM Algorithm")
    # We define the EM algorithm and iterate trought the atlases
    # in chunks, to be able to fit them in memory
    chunk_size = 50000
    updated_confusion_matrix = np.zeros_like(confusion_matrix)

    def expectation_maximization(flat_atlases):
        # E Step per voxel
        weights_per_voxel = np.tile(prior_probabilities,
                                    (flat_atlases.shape[1], 1))

        for k, atlas in enumerate(flat_atlases):
            weights_per_voxel *= confusion_matrix[k, atlas, :]

        sum_weights = weights_per_voxel.sum(1)
        weights_per_voxel[sum_weights > 0, :] /= np.c_[sum_weights[sum_weights > 0]]

        # M Step per label
        for label in labels:
            for k, atlas in enumerate(flat_atlases):
                pos_label = (atlas == label)
                updated_confusion_matrix[:, label, :] += weights_per_voxel[pos_label].sum(0)

    for _ in range(iters):
        # Iterate the EM algorithm
        updated_confusion_matrix[:] = 0

        for s in range(0, flat_atlases_filtered.shape[1], chunk_size):
            expectation_maximization(flat_atlases_filtered[:, s:s+chunk_size])

        updated_confusion_matrix_normaliser = updated_confusion_matrix.sum(2)[:, :, None]
        updated_confusion_matrix_normaliser[updated_confusion_matrix_normaliser == 0] = 1
        updated_confusion_matrix /= updated_confusion_matrix_normaliser

        max_update = np.abs(confusion_matrix - updated_confusion_matrix).max()
        confusion_matrix[:] = updated_confusion_matrix

        if max_update < tol:
            break
    else:
        warn("Leaving due to maxiters")

    print("Computing Segmentation")
    def compute_output_segmentation(flat_atlases):

        weights_per_voxel = np.tile(prior_probabilities, (flat_atlases.shape[1], 1))

        # Recompute the normalized weights cause we need them
        for k, atlas in enumerate(flat_atlases):
            weights_per_voxel *= confusion_matrix[k, atlas, :]

        sum_weights = weights_per_voxel.sum(1)  # Normalize
        weights_per_voxel[sum_weights > 0, :] /= np.c_[sum_weights[sum_weights > 0]]

        # The output segmentation possess the labels with highest weight
        output_segmentation = np.zeros(flat_atlases.shape[1], dtype=int)

        dim0 = weights_per_voxel.shape[0]

        max_weight_indices = np.argmax(weights_per_voxel, axis=1)
        output_segmentation = max_weight_indices

        max_weights = weights_per_voxel[range(dim0), max_weight_indices]

        weights_per_voxel[range(dim0), max_weight_indices] = 0

        max2_weight_indices = np.argmax(weights_per_voxel, axis=1)
        max_2_weights = weights_per_voxel[range(dim0), max2_weight_indices]

        undetermined = max_weights == max_2_weights
        output_segmentation[undetermined] = -1

        return output_segmentation

    res = []

    for s in range(0, flat_atlases_filtered.shape[1], chunk_size):
        res += compute_output_segmentation(
            flat_atlases_filtered[:, s:s+chunk_size]
            ).tolist()

    output_segmentation = np.zeros(flat_atlases.shape[1:], dtype=int)
    output_segmentation[all_nzrs] = np.array(res)

    return output_segmentation, confusion_matrix
