import nibabel

from logpar.utils import cifti_utils

from ..rohlfing import multi_label_segmentation


def segmentation_rohlfing(atlases, segout, confout=None):
    """Segments a set of atlases using the voting filter"""

    nibs = [nibabel.load(a) for a in atlases]
    atlases = [n.get_data().astype(int).ravel() for n in nibs]
    atlas0 = nibs[0]

    segmentation, conf_matrix = multi_label_segmentation(atlases)
    segmentation.resize(atlas0.shape)

    cifti_utils.save_nifti(segout, segmentation, atlas0.header, atlas0.affine,
                           version=1)
    if confout:
        cifti_utils.save_nifti(confout, conf_matrix, version=1)
