import json
import os

import nibabel
import numpy as np

from dipy.io import read_bvals_bvecs

from logpar.utils import cifti_utils
from logpar.utils import streamline_utils as sutils

from ..diffusion_inner import multi_label_segmentation


def segmentation(tract_files, dwi_file, outfile):

    subjects = [os.path.basename(atlas_file).split('.')[0]
                for atlas_file in tract_files]
    subjects = list(set(subjects))

    nsubjects = len(subjects)

    # DWI data
    dwi_image = nibabel.load(dwi_file)
    dwi_affine = dwi_image.affine

    # Lets first number each subject and recover the map between label's name
    # and label's number
    names = np.unique([os.path.basename(f).split('.')[1] for f in tract_files])
    name2label = {n:i for i, n in enumerate(names, 1)}

    votes = np.zeros([len(names)+1] + list(dwi_image.shape[:3]), dtype=np.int8)
    votes[0, :] = nsubjects

    for f in tract_files:
        tt = os.path.basename(f).split('.')[1]
        lab = name2label[tt]

        streamlines = sutils.load_stream(f, dwi_affine)

        visit = np.zeros(dwi_image.shape[:3], bool)

        for s in streamlines:
            pos_in_vox = np.round(s).astype(int)
            # if you vote for a tract, then you don't vote for a cortical label
            visit[tuple(np.transpose(pos_in_vox))] = True

        votes[0][visit] -= 1
        votes[lab][visit] += 1

    out_segmentation = np.argmax(votes, axis=0)

    cifti_utils.save_nifti(outfile, out_segmentation, affine=dwi_affine,
                           header=dwi_image.header, version=1)
