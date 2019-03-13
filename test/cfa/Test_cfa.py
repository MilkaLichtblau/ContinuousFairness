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

    def test_replaceRawByFairScores_VeryDifferentScores(self):
        groupFairScoresData = np.array([[12, 13, 14, 15, 16, 17, 18, 19],
                                        [22, 23, 24, 25, 26, 27, 28, 29],
                                        [32, 33, 34, 35, 36, 37, 38, 39],
                                        [42, 43, 44, 45, 46, 47, 48, 49]], dtype=object)
        groupFairScores = pd.DataFrame(groupFairScoresData.T, columns=["[0 0]", "[0 1]", "[1 0]", "[1 1]"])

        cfa_object = cfa.ContinuousFairnessAlgorithm(self.smallRawData,
                                                     self.groups,
                                                     self.protectedAttributes,
                                                     self.qualityAttribute,
                                                     self.scoreRanges,
                                                     self.scoreStepsize,
                                                     [1, 1, 1, 1],
                                                     self.regForOT)

        expectedData = np.array([[int(0), int(0), int(19), ""],
                                 [int(0), int(0), int(18), ""],
                                 [int(0), int(1), int(27), ""],
                                 [int(0), int(1), int(26), ""],
                                 [int(1), int(0), int(35), ""],
                                 [int(1), int(0), int(34), ""],
                                 [int(1), int(1), int(43), ""],
                                 [int(1), int(1), int(42), ""]], dtype=object)
        expected = pd.DataFrame(expectedData, columns=["gender", "ethnicity", "score", "uuid"])
        actual = cfa_object._replaceRawByFairScores(groupFairScores)
        pd.testing.assert_frame_equal(expected, actual)


if __name__ == "__main__":
    unittest.main()
