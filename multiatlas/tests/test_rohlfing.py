import unittest

import numpy as np

from multiatlas.rohlfing import multi_label_segmentation

class TestRohlfing(unittest.TestCase):

    def test_multi_label_segmentation(self):
        """Tests the implementation of rohlfing (2004) """

        train_labels = [[0,1,0,1,1,0,1,2,3,3],
                        [1,1,0,1,1,0,1,2,2,3],
                        [1,1,1,2,2,2,3,3,2,3]]

        # Voting skeme
        import ipdb; ipdb.set_trace()
        segmentation, cmatrix = multi_label_segmentation(train_labels)

        gt = np.array([1, 1, 0, 1, 1, 0, 1, 2, 2, 3])

        np.testing.assert_array_equal(segmentation, gt)
 

if __name__ == '__main__':
    train_labels = [[0,1,0,1,1,0,1,2,3,3],
                    [1,1,0,1,1,0,1,2,2,3],
                    [1,1,1,2,2,2,3,3,2,3]]

    # Voting skeme
    #import ipdb; ipdb.set_trace()
    #segmentation, cmatrix = multi_label_segmentation(train_labels)

    dataImageA = np.r_[0, 1, 3, 3, 0, 4, 13, 13, 0, 0]
    dataImageB = np.r_[1, 1, 2, 4, 0, 4, 5, 12, 1, 0]
    dataImageC = np.r_[0, 2, 2, 3, 0, 5, 5, 13, 8, 0]

    combinationABC = np.r_[0, 1, 2, 3, 0, 4, 5, 13, -1, 0]
    combinationAB = np.r_[-1, 1, -1, -1, 0, 4, -1, -1, -1, 0]

    segmentation, confusion_matrix = multi_label_segmentation([dataImageA,
                                                               dataImageB])
    np.testing.assert_equal(combinationAB, segmentation)

    segmentation, confusion_matrix = multi_label_segmentation(
        np.array([dataImageA, dataImageB, dataImageC])
        )
    np.testing.assert_equal(combinationABC, segmentation)
