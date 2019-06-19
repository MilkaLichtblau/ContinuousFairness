'''
Created on Jun 19, 2019

@author: meike
'''
import unittest
import pandas as pd
import numpy as np
import uuid
from evaluation.relevanceMeasures import pak


class Test(unittest.TestCase):

    def setUp(self):
        dataSize = 20
        uuidCol = []
        for _ in np.arange(dataSize):
            uuidCol.append(uuid.uuid4())
        origScoreCol = np.arange(dataSize)
        fairScoreCol = np.arange(10, dataSize + 10)
        self._origData = pd.DataFrame([origScoreCol, uuidCol], columns=["score", "uuid"])
        self._fairData = pd.DataFrame([fairScoreCol, uuidCol], columns=["score", "uuid"])

    def test_pak(self):

        pass


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_pak']
    unittest.main()
