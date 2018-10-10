import numpy as np

from scipy.sparse import lil_matrix, csr_matrix


def voting_filter(atlases, undetermined=-1):
    atlases = np.atleast_2d(atlases)

    # Get the labels
    labels = np.unique(atlases)

    # The shape of the matrix is labels x nvoxels
    shape = (labels.max()+1,) + atlases.shape[1:]

    if atlases.shape[0] < 256:
        votes_per_label = np.zeros(shape, dtype=np.uint8)
    else:
        votes_per_label = np.zeros(shape, dtype=np.uint16)

    # Count votes per label/voxel
    voxindices = np.arange(shape[1])
    for s, a in enumerate(atlases):
        print("Counting votes of subject {}".format(s))
        votes_per_label[a, voxindices] += 1

    # The final segmentation contains the most-voted labels
    max_indices = np.argmax(votes_per_label, axis=0)
    segmentation = max_indices

    # But if two labels where voted the same amount of times
    # on a voxel, then the voxel is undefined
    max_votes = votes_per_label[max_indices, voxindices]
    votes_per_label[max_indices, voxindices] = 0

    max_2nd_indices = np.argmax(votes_per_label, axis=0)
    max_2nd_votes = votes_per_label[max_2nd_indices, voxindices]

    nzr = (max_votes == max_2nd_votes).nonzero()[0]
    segmentation[nzr] = undetermined

    return segmentation
