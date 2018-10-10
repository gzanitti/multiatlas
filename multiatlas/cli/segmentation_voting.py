import nibabel

from logpar.utils import cifti_utils

from ..voting import voting_filter


def segmentation_voting(atlases, outfile):
    """Segments a set of atlases using the voting filter"""
    
    nibs = [nibabel.load(a) for a in atlases]
    atlases = [n.get_data().astype(int).ravel() for n in nibs]
    atlas0 = nibs[0]
    

    segmentation = voting_filter(atlases)
    segmentation.resize(atlas0.shape)

    cifti_utils.save_nifti(outfile, segmentation, atlas0.header, atlas0.affine,
                           version=1) 
