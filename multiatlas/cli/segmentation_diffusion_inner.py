import json
import os

import nibabel
import numpy as np

from dipy.io import read_bvals_bvecs

from logpar.utils import cifti_utils
from logpar.utils import streamline_utils as sutils

from ..diffusion_inner import multi_label_segmentation


def segmentation(atlases_files, tract_files, dwi_file,
                 bvals_file, bvecs_file, outfile):
    """Computes a segmentation voting weighted by diffusion information.
       This script assumes some name conventions, particularly:
           - The atlas files shuold be of the form: <subject>_*.nii
           - The streamlines are in the form: <subject>_<tract>_*.trk

       Parameters
       ----------
       atlases_files: list
           A list of volumes with a fsl parcellation per subject
       tract_files: list
           A list of tract files
       dwi_file: string
           The dwi of the target subject
       bvals: string
           bvals file
       bvecs: string
           bvecs file
       outfile: string
           name of the outfile"""
    
    if atlases_files:
        subjects = [os.path.basename(atlas_file).split('_')[0]
                    for atlas_file in atlases_files]

        # Load atlases
        atlases = [nibabel.load(atlas_file).get_data()
                   for atlas_file in atlases_files]

        atlases = np.array(atlases)
        max_atlas_label = atlases.max()
    else:
        atlases = []
        subjects = [os.path.basename(atlas_file).split('.')[0]
                    for atlas_file in tract_files]
        subjects = list(set(subjects))
        max_atlas_label = 0

    subject2id = {s:i for i, s in enumerate(subjects)}
    # DWI data
    dwi_image = nibabel.load(dwi_file)
    dwi_affine = dwi_image.affine
    test_dwi = dwi_image.get_data()

    bvals, bvecs = read_bvals_bvecs(bvals_file, bvecs_file)

    # Tracts
    # Lets first number each subject and recover the map between label's name
    # and label's number
    names = np.unique([os.path.basename(f).split('.')[1] for f in tract_files])
    name2label = {n:i+max_atlas_label for i, n in enumerate(names, 1)}

    # Lets create a structure such that tracts[s, l] are the tracts of the
    # label l for the subject s
    tracts = {}
    for f in tract_files:
        ss = os.path.basename(f).split('.')[0]
        tt = os.path.basename(f).split('.')[1]
        sid, lab = subject2id[ss], name2label[tt]

        streamlines = sutils.load_stream(f, dwi_affine)
        tracts[(sid, lab)] = streamlines

        # if you vote for a tract, then you don't vote for a cortical label
        #for s in streamlines:
        #    pos_in_vox = np.round(s).astype(int)
        #    atlases[sid][tuple(np.transpose(pos_in_vox))] = 0
   
    segmentation = multi_label_segmentation(atlases, tracts, test_dwi,
                                            bvecs, bvals, dwi_affine)

    cifti_utils.save_nifti(outfile, segmentation, affine=dwi_affine,
                           header=dwi_image.header, version=1)
