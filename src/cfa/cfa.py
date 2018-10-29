'''
Created on Sep 27, 2018

@author: meike.zehlike
'''

import ot
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import stats


def continuousFairnessAlgorithm(data, thetas, regForOT, shape_bins, path='.', plot=False):
    """
    @param data:         pandas dataframe with scores per group, groups do not necessarily have same
                         amount of scores, i.e. data might contain NaNs
    @param thetas:       vector of parameters that determine how close a distribution is to be moved to
                         the general barycenter. One theta per group.
                         theta of 1 means that a group distribution is totally moved into the general
                         barycenter
                         theta of 0 means that a group distribution stays exactly as it is
    @param regForOT:     regularization parameter for optimal transport, see ot docs for details
    @param shape_bins:   divisor for the bin size in the histogram that is calculated for each data column
    """

    print(data.shape)

    # calculate z score per column
    norm_data = stats.zscore(data[~np.isnan(data)], axis=1)
    normalizedData = pd.DataFrame(norm_data, columns=data.columns)
    print(normalizedData.shape)

    if plot:
        normalizedData.plot(kind='line')
        plt.savefig(path + 'rawData_normalized', dpi=100, bbox_inches='tight')

    # calculate numbers of bins by largest group and devide by shape_bins, round up to have at least one
    num_bins = math.ceil(data.shape[0] / shape_bins)

    # calculate general edges for column histograms
    dataIn1D = normalizedData.values.flatten()
    oneDNoNans = dataIn1D[~np.isnan(dataIn1D)]
    bin_edges = np.histogram_bin_edges(oneDNoNans, bins=num_bins)
    dataAsHistograms = pd.DataFrame()

    # get normalized histogram from each column and save to new dataframe
    for colName in data.columns:
        colNoNans = pd.DataFrame(normalizedData[colName][~np.isnan(normalizedData[colName])])
        colAsHist = np.histogram(colNoNans[colName], bins=bin_edges, density=True)[0]
        dataAsHistograms[colName] = colAsHist

    if plot:
        ax = dataAsHistograms.plot(kind='line', use_index=False)
        ax.set_xticklabels(np.around(bin_edges[1:], decimals=2))
        plt.savefig(path + 'dataAsHistograms.png', dpi=100, bbox_inches='tight')

    # loss matrix + normalization
    loss_matrix = ot.utils.dist0(num_bins)
    loss_matrix /= loss_matrix.max()

    # compute general barycenter of all score distributions
    total_bary = ot.bregman.barycenter(dataAsHistograms, loss_matrix, regForOT, verbose=True, log=True)[0]
    if plot:
        baryFrame = pd.DataFrame(total_bary)
        baryFrame.plot()
        plt.savefig(path + 'totalBarycenter.png', dpi=100, bbox_inches='tight')

    # compute barycenters between general barycenter and each score distribution (i.e. each social group)
    group_barycenters = pd.DataFrame(columns=dataAsHistograms.columns.values.tolist())
    for groupName in dataAsHistograms:
        # build 2-column matrix from group data and general barycenter
        groupMatrix = pd.concat([dataAsHistograms[groupName], pd.Series(total_bary)], axis=1)
        # get corresponding theta
        theta = thetas[dataAsHistograms.columns.get_loc(groupName)]
        # calculate barycenters
        weights = np.array([1 - theta, theta])
        group_barycenters[groupName] = ot.bregman.barycenter(groupMatrix, loss_matrix, regForOT,
                                                             weights=weights, verbose=True, log=True)[0]

    if plot:
        ax = group_barycenters.plot(kind='line', use_index=False)
        # ax.set_xticklabels(np.around(bin_edges[1:], decimals=2))
        plt.savefig(path + 'groupBarycenters.png', dpi=100, bbox_inches='tight')

