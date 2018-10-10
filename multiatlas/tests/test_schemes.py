import unittest

import numpy as np

from .. import generative_model
from ..schemes import label_votes, intensity_id, diffusion_id

class TestScheme(unittest.TestCase):

    def test_voting_methodology(self):
        """Tests the voting scheme and the voting as a whole method """
        # 3 trains subjects, 1 test subject, 10 voxels
        test_image = np.arange(0, 1, 0.1)
        train_images = np.tile(test_image, (3,1))

        train_labels = [[0,1,0,1,1,0,1,2,3,3],
                        [1,1,0,1,1,0,1,2,2,3],
                        [1,1,1,2,2,2,3,3,2,3]]

        test_diffusion = [[0]*3]*10

        train_diffusion = [test_diffusion,
                           test_diffusion,
                           test_diffusion]

        # Voting skeme
        segmentation = generative_model(label_votes, intensity_id, diffusion_id,
                                        train_labels, test_image, train_images,
                                        test_diffusion, train_diffusion)
        gt = np.array([1, 1, 0, 1, 1, 0, 1, 2, 2, 3])
        np.testing.assert_array_equal(segmentation, gt)
