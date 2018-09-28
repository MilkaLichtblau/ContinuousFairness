'''
Created on Sep 27, 2018

@author: meike.zehlike
'''

import ot

def continuousFairnessAlgorithm(data, thetas):
    """
    @param data:     pandas dataframe with scores per group, groups do not necessarily have same
                     amount of scores, i.e. data might contain NaNs
    @param thetas:   vector of parameters that determine how close a distribution is to be moved to
                     the general barycenter of all score distributions. One theta per group.
                     theta of 1 means that a group distribution is totally moved into the general barycenter
                     theta of 0 means that a group distribution stays exactly as it is
    """

    n = 100
    a2 = ot.datasets.get_1D_gauss(n, m=60, s=8)


    # loss matrix + normalization
    loss_matrix = ot.utils.dist0(n)
    loss_matrix /= loss_matrix.max()
