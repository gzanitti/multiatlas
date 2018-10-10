import numpy
import nibabel

from scipy.interpolate import RegularGridInterpolator


def load_stream(streamfile, affine_to_rasmm=None):
    ''' Loads streamlines from a file. By default, nibabel.streamlines.load
        loads the streamlines in the ras+mm space, this function automatically
        transforms the loaded streamlines if an affine_to_rasmm is given.

        Parameters
        ----------
        streamfile: string
            Route to the file
        affine_to_rasm: array_like
            Transformation from some space to ras+mm. This matrix will
            be inverted and applied to the loaded streamlines
        Returns
        ------
        array_like
            List of streamlines
    '''
    # When loading, the streamlines are in RAS+ mm
    tractogram = nibabel.streamlines.load(streamfile)
    streamlines = tractogram.streamlines
    if affine_to_rasmm is not None:
        inv_affine = numpy.linalg.inv(affine_to_rasmm)
        streamlines = [nibabel.affines.apply_affine(inv_affine, s)
                       for s in streamlines]
    return streamlines


def save_stream(outfile, streamlist, affine_to_rasmm=None,
                dimention=(1, 1, 1), voxel_size=(1, 1, 1)):
    ''' Function to save a set of streamlines. The streamlines can be in
        any space. If the streamlines are in rasmm space, then it's not
        necessary to specify a transformation. Otherwhise, the transformation
        from the current space to rasmm MUST be specified.

        Parameters
        ----------
        streamlist: list of array-like
            List of streamlines to save
        affine_to_rasmm: array-like
            Transformation from the current space of the streamlines to the
            rasmm space

        Returns
        -------
        None '''
    if affine_to_rasmm is None:
        affine_to_rasmm = numpy.eye(4)

    tract = nibabel.streamlines.Tractogram(streamlist,
                                           affine_to_rasmm=affine_to_rasmm)
    hdr_dict = {'dimensions': dimention,
                'voxel_sizes': voxel_size,
                'voxel_to_rasmm': affine_to_rasmm,
                'voxel_order': "".join(nibabel.aff2axcodes(affine_to_rasmm))}
    trk_file = nibabel.streamlines.TrkFile(tract, hdr_dict)
    trk_file.save(outfile)


def warp_points(src_affine, warp_field, streamlines_vox):
    """Affine is the transform from src_vox to src_mm
       Warp transforms from vox to vox
       streamlines are in voxels
       transformed = aff_inv_dst*(warp*(aff_inv_warp*(aff_src * pts)))"""
    # Interpolate the warp field
    rx, ry, rz = list(map(range, warp_field.shape[:3]))

    warp_interpolators = [RegularGridInterpolator((rx, ry, rz),
                                                  warp_field[:, :, :, i])
                          for i in range(3)]

    # warp each streamline
    streamlines_warped = []
    for stream in streamlines_vox:
        # Compute the position of the streamline in mm
        stream_mm = nibabel.affines.apply_affine(src_affine, stream)

        # Get the warp at (x,y,z )
        vox_interpol = [interpol(stream) for interpol in warp_interpolators]
        warp_offset = numpy.transpose(vox_interpol)

        # Apply the warp to the streamlines
        flip = [-1, -1, 1]
        stream_warped = stream_mm + warp_offset*flip
        
        # Save
        streamlines_warped.append(stream_warped)

    return streamlines_warped


def apply_warp_streamlines(fstreamlines, img_src, img_warp, img_dst, outfile):
    """Applies warp to a set of streamlines, to non-linear transform"""

    # Load streamlines in vox-src space
    src_affine = nibabel.load(img_src).affine
    streamlines_vox = load_stream(fstreamlines, src_affine)

    # Load warp
    warp_nib = nibabel.load(img_warp)
    warp_field = warp_nib.get_data()
    warp_field = warp_field[:,:,:,0,:]
    
    # move to vox space in destination
    streamlines_warped = warp_points(src_affine, warp_field, streamlines_vox)

    warp_affine = warp_nib.affine
    warp_affine_inv = numpy.linalg.inv(warp_affine)

    streamlines_transformed = [nibabel.affines.apply_affine(warp_affine_inv,
                                                            sw)
                               for sw in streamlines_warped]

    save_stream(outfile, streamlines_transformed, warp_affine,
                warp_nib.shape[:3], warp_nib.header.get_zooms()[:3])
