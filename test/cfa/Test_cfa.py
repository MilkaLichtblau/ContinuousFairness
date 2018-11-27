'''
Created on Sep 28, 2018

@author: mzehlike
'''
import unittest
import pandas as pd
import numpy as np
from cfa import cfa


class TestContinuousFairnessAlgorithm(unittest.TestCase):

    def setUp(self):
        self.qualityAttribute = "score"
        self.protectedAttributes = ["gender", "ethnicity"]
        self.groups = pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], columns=["gender", "ethnicity"])
        datapoints = np.array([[int(0), int(0), int(9), ""],
                               [int(0), int(0), int(8), ""],
                               [int(0), int(1), int(7), ""],
                               [int(0), int(1), int(6), ""],
                               [int(1), int(0), int(5), ""],
                               [int(1), int(0), int(4), ""],
                               [int(1), int(1), int(3), ""],
                               [int(1), int(1), int(2), ""]], dtype=object)
        self.scoreRanges = [2, 9]
        self.scoreStepsize = 1
        self.regForOT = 5e-3
        self.smallRawData = pd.DataFrame(datapoints, columns=["gender", "ethnicity", "score", "uuid"])

    def tearDown(self):
        pass

    def test_getScoresByGroup_SameGroupSizes(self):
        cfa_object = cfa.ContinuousFairnessAlgorithm(self.smallRawData,
                                                     self.groups,
                                                     self.protectedAttributes,
                                                     self.qualityAttribute,
                                                     self.scoreRanges,
                                                     self.scoreStepsize,
                                                     [1, 1, 1, 1],
                                                     self.regForOT)
        expectedData = np.array([[9, 7, 5, 3], [8, 6, 4, 2]], dtype=object)
        expected = pd.DataFrame(expectedData, columns=["[0 0]", "[0 1]", "[1 0]", "[1 1]"])
        actual = cfa_object._getScoresByGroup(self.smallRawData)
        pd.testing.assert_frame_equal(expected, actual)

    def test_getScoresByGroup_DifferentGroupSizes(self):
        extraRow = pd.DataFrame([[int(1), int(1), int(1), ""]], columns=["gender", "ethnicity", "score", "uuid"])
        self.smallRawData = self.smallRawData.append(extraRow)
        cfa_object = cfa.ContinuousFairnessAlgorithm(self.smallRawData,
                                                     self.groups,
                                                     self.protectedAttributes,
                                                     self.qualityAttribute,
                                                     self.scoreRanges,
                                                     self.scoreStepsize,
                                                     [1, 1, 1, 1],
                                                     self.regForOT)
        expectedData = np.array([[9, 7, 5, 3], [8, 6, 4, 2], [np.NaN, np.NaN, np.NaN, 1]], dtype=object)
        expected = pd.DataFrame(expectedData, columns=["[0 0]", "[0 1]", "[1 0]", "[1 1]"])
        actual = cfa_object._getScoresByGroup(self.smallRawData)
        pd.testing.assert_frame_equal(expected, actual)
        
        
    def test__replaceRawByFairScores(self):


if __name__ == "__main__":
    unittest.main()
