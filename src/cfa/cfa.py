'''
Created on Sep 27, 2018

@author: meike.zehlike
'''

import ot
import numpy as np
import pandas as pd
import visualization.plots as plots
import matplotlib.pyplot as plt


class ContinuousFairnessAlgorithm():
    """
    TODO: write doc
    """

    def __init__(self, rawData, score_ranges, score_stepsize, thetas, regForOT, path='.', plot=False):
        """
        @param rawData:          pandas dataframe with scores per group, groups do not necessarily have same
                                 amount of scores, i.e. data might contain NaNs
        @param score_ranges:     tuple that contains the lower and upper most possible score
        @param score_stepsize:   stepsize between two scores
        @param thetas:           vector of parameters that determine how close a distribution is to be moved to
                                 the general barycenter. One theta per group.
                                 theta of 1 means that a group distribution is totally moved into the general
                                 barycenter
                                 theta of 0 means that a group distribution stays exactly as it is
        @param regForOT:         regularization parameter for optimal transport, see ot docs for details
        """

        self.__rawData = rawData
        self.__groupNames = self.__rawData.columns.values.tolist()
        self.__score_values = np.arange(score_ranges[0], score_ranges[1] + score_stepsize, score_stepsize)
        # calculate bin number for histograms and loss matrix size
        self.__num_bins = int(len(self.__score_values))
        # calculate loss matrix
        self.__lossMatrix = ot.utils.dist0(self.__num_bins)
        self.__lossMatrix /= self.__lossMatrix.max()
        self.__thetas = thetas
        self.__regForOT = regForOT

        self.__fairData = pd.DataFrame(columns=self.__groupNames)
        self.__rawDataAsHistograms = pd.DataFrame(columns=self.__groupNames)
        self.__fairDataAsHistograms = pd.DataFrame(columns=self.__groupNames)

        self.__plotPath = path
        self.__plot = plot

    def __getGroupHistograms(self, data, histograms, bin_edges):
        # get histogram from each column and save to new dataframe
        # gets as parameters pointers to raw data and new dataframe where to save the histograms

        for groupName in data.columns:
            groupColNoNans = pd.DataFrame(data[groupName][~np.isnan(data[groupName])])
            colAsHist = np.histogram(groupColNoNans[groupName], bins=bin_edges, density=True)[0]
            histograms[groupName] = colAsHist

        if histograms.isnull().values.any():
            raise ValueError("Histogram data contains nans")

    def __getTotalBarycenter(self):
        # calculate group sizes in total and percent
        groupSizes = self.__rawData.count()
        groupSizesPercent = self.__rawData.count().divide(groupSizes.sum())

        # compute general barycenter of all score distributions
        total_bary = ot.bregman.barycenter(self.__rawDataAsHistograms,
                                           self.__lossMatrix,
                                           self.__regForOT,
                                           weights=groupSizesPercent.values,
                                           verbose=True,
                                           log=True)[0]
        if self.__plot:
            self.__plott(pd.DataFrame(total_bary), 'totalBarycenter.png')

        return total_bary

    def __get_group_barycenters(self, total_bary):
        # compute barycenters between general barycenter and each score distribution (i.e. each social group)
        group_barycenters = pd.DataFrame(columns=self.__groupNames)
        for groupName in self.__rawDataAsHistograms:  # build 2-column matrix from group data and general barycenter
            groupMatrix = pd.concat([self.__rawDataAsHistograms[groupName], pd.Series(total_bary)], axis=1)  # get corresponding theta
            theta = self.__thetas[self.__rawDataAsHistograms.columns.get_loc(groupName)]  # calculate barycenters
            weights = np.array([1 - theta, theta])
            group_barycenters[groupName] = ot.bregman.barycenter(groupMatrix,
                                                                 self.__lossMatrix,
                                                                 self.__regForOT,
                                                                 weights=weights,
                                                                 verbose=True,
                                                                 log=True)[0]

        if self.__plot:
            self.__plott(group_barycenters, 'groupBarycenters.png')
        return group_barycenters

    def __calculateFairScores(self, group_barycenters):
        # calculate new scores from group barycenters
        groupFairScores = pd.DataFrame(columns=self.__groupNames)
        for groupName in self.__rawDataAsHistograms:
            ot_matrix = ot.emd(self.__rawDataAsHistograms[groupName],
                               group_barycenters[groupName],
                               self.__lossMatrix)
            # TODO: landet man damit auf jeden Fall im gleichen Score Range?
            groupFairScores[groupName] = np.matmul(ot_matrix, self.__score_values.T)

        if self.__plot:
            self.__plott(groupFairScores, 'fairScoreReplacementStrategy.png')

        for groupName in self.__rawData.columns:
            rawScores = self.__rawData[groupName][~np.isnan(self.__rawData[groupName])]
            fairScores = groupFairScores[groupName]

            for index, fairScore in fairScores.iteritems():
                # fairScore = fairScore.at[rawScore, groupName]
            #             fairScore = fairScore.at[0]
                rawScores[rawScores == index] = fairScore
            self.__fairData[groupName] = rawScores

        if self.__plot:
            self.__fairData.plot.kde()
            plt.savefig(self.__plotPath + 'fairScoreDistributionPerGroup.png', dpi=100, bbox_inches='tight')
            bin_edges = np.linspace(groupFairScores.min().min(), groupFairScores.max().max(), int(self.__num_bins / 4))
            self.__getGroupHistograms(self.__fairData, self.__fairDataAsHistograms, bin_edges)
            self.__plott(self.__fairDataAsHistograms, 'fairScoresAsHistograms.png')

    def __plott(self, dataframe, filename):
        dataframe.plot(kind='line', use_index=False)
        plt.savefig(self.__plotPath + filename, dpi=100, bbox_inches='tight')

    def continuousFairnessAlgorithm(self):
        """
        TODO: write that algorithm assumes finite integer scores, so no floating scores are allowed
        and they have to be represented somehow as integers, otherwise the algorithm doesn't work


        """

        leftMostEdge = self.__score_values[0] - (self.__score_values[1] - self.__score_values[0])
        bin_edges = np.insert(self.__score_values, 0, leftMostEdge)

        self.__getGroupHistograms(self.__rawData, self.__rawDataAsHistograms, bin_edges)
        if self.__plot:
            self.__plott(self.__rawDataAsHistograms, 'rawScoresAsHistograms.png')

        total_bary = self.__getTotalBarycenter()
        group_barycenters = self.__get_group_barycenters(total_bary)

        self.__calculateFairScores(group_barycenters)

