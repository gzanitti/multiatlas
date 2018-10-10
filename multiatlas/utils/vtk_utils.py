import vtk
from vtk.util import numpy_support as ns
import numpy as np

from . import tractography

def read_vtkPolyData(filename):
    r'''
    Reads a VTKPolyData file and outputs a tracts/tracts_data pair

    Parameters
    ----------
    filename : str
        VTKPolyData filename

    Returns
    -------
    tracts : list of float array N_ix3
        Each element of the list is a tract represented as point array,
        the length of the i-th tract is N_i
    tract_data : dict of <data name>= list of float array of N_ixM
        Each element in the list corresponds to a tract,
        N_i is the length of the i-th tract and M is the
        number of components of that data type.
    '''

    if filename.endswith('xml') or filename.endswith('vtp'):
        polydata_reader = vtk.vtkXMLPolyDataReader()
    else:
        polydata_reader = vtk.vtkPolyDataReader()

    polydata_reader.SetFileName(filename)
    polydata_reader.Update()

    polydata = polydata_reader.GetOutput()

    return vtkPolyData_to_tracts(polydata)


def vtkPolyData_to_tracts(polydata, return_tractography_object=True):
    r'''
    Reads a VTKPolyData object and outputs a tracts/tracts_data pair

    Parameters
    ----------
    polydata : vtkPolyData
        VTKPolyData Object

    Returns
    -------
    tracts : list of float array N_ix3
        Each element of the list is a tract represented as point array,
        the length of the i-th tract is N_i
    tract_data : dict of <data name>= list of float array of N_ixM
        Each element in the list corresponds to a tract,
        N_i is the length of the i-th tract and M is the
        number of components of that data type.
    '''

    result = {}
    result['lines'] = ns.vtk_to_numpy(polydata.GetLines().GetData())
    result['points'] = ns.vtk_to_numpy(polydata.GetPoints().GetData())
    result['numberOfLines'] = polydata.GetNumberOfLines()

    data = {}
    if polydata.GetPointData().GetScalars():
        data['ActiveScalars'] = polydata.GetPointData().GetScalars().GetName()
    if polydata.GetPointData().GetVectors():
        data['ActiveVectors'] = polydata.GetPointData().GetVectors().GetName()
    if polydata.GetPointData().GetTensors():
        data['ActiveTensors'] = polydata.GetPointData().GetTensors().GetName()

    for i in range(polydata.GetPointData().GetNumberOfArrays()):
        array = polydata.GetPointData().GetArray(i)
        np_array = ns.vtk_to_numpy(array)
        if np_array.ndim == 1:
            np_array = np_array.reshape(len(np_array), 1)
        data[polydata.GetPointData().GetArrayName(i)] = np_array

    result['pointData'] = data

    tracts, data = vtkPolyData_dictionary_to_tracts_and_data(result)
    if return_tractography_object:
        tr = tractography.Tractography()
        tr.append(tracts, data)
        return tr
    else:
        return tracts, data


def vtkPolyData_dictionary_to_tracts_and_data(dictionary):
    r'''
    Create a tractography from a dictionary
    organized as a VTK poly data.

    Parameters
    ----------
    dictionary : dict
                Dictionary containing the elements for a tractography
                points : array Nx3 of float
                    each element is a point in RAS space
                lines : Mx1 of int
                    The array is organized as: K, ix_1, ..., ix_k, L, ix_1, ..., ix_L
                    For instance the array [4, 0, 1, 2, 3] means that that line
                    is formed by the sequence of points 0, 1, 2 and 3 on the
                    points array.
                'numberOfLines' : int
                    The total number of lines in the array.

    Returns
    -------
    tracts : list of float array N_ix3
        Each element of the list is a tract represented as point array,
        the length of the i-th tract is N_i
    tract_data : dict of <data name>= list of float array of N_ixM
        Each element in the list corresponds to a tract,
        N_i is the length of the i-th tract and M is the
        number of components of that data type.
    '''
    dictionary_keys = set(('lines', 'points', 'numberOfLines'))
    if not dictionary_keys.issubset(dictionary.keys()):
        raise ValueError("Dictionary must have the keys lines and points" + repr(
            dictionary.keys()))

    # Tracts and Lines are the same thing
    tract_data = {}
    tracts = []

    lines = np.asarray(dictionary['lines']).squeeze()
    points = dictionary['points']

    actual_line_index = 0
    number_of_tracts = dictionary['numberOfLines']
    original_lines = []
    for l in range(number_of_tracts):
        tracts.append(
            points[
                lines[
                    actual_line_index + 1:
                    actual_line_index + lines[actual_line_index] + 1
                ]
            ]
        )
        original_lines.append(
            np.array(
                lines[
                    actual_line_index + 1:
                    actual_line_index + lines[actual_line_index] + 1],
                copy=True
            ))
        actual_line_index += lines[actual_line_index] + 1

    if 'pointData' in dictionary:
        point_data_keys = [
            it[0] for it in dictionary['pointData'].items()
            if isinstance(it[1], np.ndarray)
        ]

        for k in point_data_keys:
            array_data = dictionary['pointData'][k]
            if not k in tract_data:
                tract_data[k] = [
                    array_data[f]
                    for f in original_lines
                ]
            else:
                np.vstack(tract_data[k])
                tract_data[k].extend(
                    [
                        array_data[f]
                        for f in original_lines[-number_of_tracts:]
                    ]
                )

    return tracts, tract_data
