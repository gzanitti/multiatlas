''' Setup file '''
from os import path
from distutils import extension

from setuptools import setup
import numpy

package_name = 'multiatlas'
cli_module = package_name + '.cli'
utils_module = package_name + '.utils'

setup(name=package_name,
      version='0.2',
      description='Tools to create multiatlas segmentation',
      url='https://github.com/AthenaEPI/NeuroRef/multiatlas',
      author='Gallardo Diez, Guillermo Alejandro',
      author_email='guillermo-gallardo.diez@inria.fr',
      include_package_data=True,
      packages=[package_name, cli_module, utils_module],
      scripts=['scripts/apply_warp_streamlines',
               'scripts/segmentation_voting',
               'scripts/segmentation_rohlfing',
               'scripts/segmentation_diffusion_inner',
               'scripts/vtk_to_trk_converter'],
      zip_safe=False)
