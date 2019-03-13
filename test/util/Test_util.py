'''
Created on Sep 28, 2018

@author: mzehlike
'''
import unittest
import pandas as pd
import numpy as np
from util import util


class Test(unittest.TestCase):


    def setUp(self):
        cols = ['gender', 'score']
        self.dataset = pd.DataFrame([[0, 10],
                                     [0, 10],
                                     [0, 10],
                                     [0, 10],
                                     [0, 10],
                                     [1, 2],
                                     [1, 2]], columns=cols)
        self.groups = pd.DataFrame([0, 1], columns=['gender'])


    def tearDown(self):
        pass


    def test_scoresByGroups(self):
        scoresPerGroup = util.scoresByGroups(self.dataset, self.groups, 'score')
        expected = pd.DataFrame([[10, 2],
                                 [10, 2],
                                 [10, np.NAN],
                                 [10, np.NAN],
                                 [10, np.NAN]], columns=['[0]', '[1]'])
        pd.testing.assert_frame_equal(expected, scoresPerGroup)


