'''
Created on Sep 27, 2018

@author: meike.zehlike
'''

import ot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def continuousFairnessAlgorithm(data, score_ranges, thetas, regForOT, path='.', plot=False):
    """
    TODO: write that algorithm assumes finite integer scores, so no floating scores are allowed
    and they have to be represented somehow as integers, otherwise the algorithm doesn't work

    @param data:         pandas dataframe with scores per group, groups do not necessarily have same
                         amount of scores, i.e. data might contain NaNs
    @param score_ranges: tuple that contains the lower and upper most possible score
    @param thetas:       vector of parameters that determine how close a distribution is to be moved to
                         the general barycenter. One theta per group.
                         theta of 1 means that a group distribution is totally moved into the general
                         barycenter
                         theta of 0 means that a group distribution stays exactly as it is
    @param regForOT:     regularization parameter for optimal transport, see ot docs for details
    """

    # calculate group sizes in total and percent
    groupSizes = data.count()
    groupSizesPercent = data.count().divide(groupSizes.sum())

    # calculate general edges for column histograms
    score_values = np.arange(score_ranges[0], score_ranges[1] + 1)
    num_bins = int(score_values.max())
    bin_edges = np.arange(num_bins + 1)  # len(bin_edges[1:])
    dataAsHistograms = pd.DataFrame()

    # get histogram from each column and save to new dataframe
    for colName in data.columns:
        colNoNans = pd.DataFrame(data[colName][~np.isnan(data[colName])])
        colAsHist = np.histogram(colNoNans[colName], bins=bin_edges, density=True)[0]
        dataAsHistograms[colName] = colAsHist

    if dataAsHistograms.isnull().values.any():
        raise ValueError("Histogram data contains nans")

    if plot:
        dataAsHistograms.plot(kind='line', use_index=False)
        plt.savefig(path + 'dataAsHistograms.png', dpi=100, bbox_inches='tight')

    # loss matrix + normalization
    loss_matrix = ot.utils.dist0(num_bins)
    loss_matrix /= loss_matrix.max()

    # compute general barycenter of all score distributions
    weights = groupSizesPercent.values
    total_bary = ot.bregman.barycenter(dataAsHistograms, loss_matrix, regForOT,
                                       weights=groupSizesPercent.values,
                                       verbose=True, log=True)[0]
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
        group_barycenters.plot(kind='line', use_index=False)
        plt.savefig(path + 'groupBarycenters.png', dpi=100, bbox_inches='tight')

    # calculate new scores from group barycenters
    group_fair_scores = pd.DataFrame(columns=dataAsHistograms.columns.values.tolist())
    for groupName in dataAsHistograms:
        ot_matrix = ot.emd(dataAsHistograms[groupName],
                           group_barycenters[groupName],
                           loss_matrix)
        # TODO: landet man damit auf jeden Fall im gleichen Score Range?
        group_fair_scores[groupName] = np.matmul(ot_matrix, score_values.T)

#         group_fair_scores[groupName] =
    if plot:
        group_fair_scores.plot(kind='line', use_index=False)
        plt.savefig(path + 'fairScoresPerGroup.png', dpi=100, bbox_inches='tight')

    data_fair_scores = pd.DataFrame(columns=data.columns.values.tolist())
    for colName in data.columns:
        colNoNans = pd.DataFrame(data[colName][~np.isnan(data[colName])])

    # use stats.percentileofscore um herauszufinden, an welcher Stelle im Gruppenranking einer prozentual war
    # packe ihn in das gleiche Percentile im Barycenter wieder hinein...
