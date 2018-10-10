import nibabel
import numpy as np

from ..utils.tractography import tractography_from_vtk_file

def vtk_to_trk_converter(tract_in, tract_out, reference_volume=None):
    
    if not tract_in.endswith('vtk'):
        raise ValueError("Sorry, we only work with vtk files as input")

    if not tract_out.endswith('trk'):
        raise ValueError("Sorry, we only work with trk files as output")

    tractography = tractography_from_vtk_file(tract_in)

    # ADD metadata to the TRK format
    hdr_dict = None

    if not (reference_volume is None):
        ref = nibabel.load(reference_volume)

        hdr_dict = {'dimensions': ref.shape[:3],
                    'voxel_sizes': ref.header.get_zooms()[:3],
                    'voxel_to_rasmm': ref.affine,
                    'voxel_order': "".join(nibabel.aff2axcodes(ref.affine))}

    tract = nibabel.streamlines.Tractogram(tractography.tracts(),
                                           affine_to_rasmm=np.eye(4))
    trk_file = nibabel.streamlines.TrkFile(tract, hdr_dict)
    trk_file.save(tract_out)
