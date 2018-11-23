'''
Created on Sep 27, 2018

@author: meike.zehlike
'''

import ot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpmath import norm


class ContinuousFairnessAlgorithm():
    """
    TODO: write doc
    """

    def __init__(self, rawData, groups, prot_attr, qual_attr, score_ranges, score_stepsize, thetas, regForOT, path='.', plot=False):
        """
        @param rawData:          pandas dataframe with scores per group, groups do not necessarily have same
                                 amount of scores, i.e. data might contain NaNs
        @param groups:           all possible groups in data as dataframe. One row contains manifestations
                                 of protected attributes, hence represents a group
                                 example: [[white, male], [white, female], [hispanic, male], [hispanic, female]]
        @param prot_attr:        column names that are protected attributes
        @param qual_attr:        name of column that contains the scores
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
        self.__qualityAtribute = qual_attr
        self.__protectedAttributes = prot_attr
        self.__rawDataByGroup = self.__getScoresByGroup(rawData, groups, qual_attr)
        self.__groupNames = self.__rawDataByGroup.columns.values.tolist()
        self.__groups = groups  # TODO: das könnte man elefanter lösen, hier ist jetzt einiges doppelt und dreifach...nochmal refactoring machen
        self.__score_values = np.arange(score_ranges[0], score_ranges[1] + score_stepsize, score_stepsize)
        # calculate bin number for histograms and loss matrix size
        self.__num_bins = int(len(self.__score_values))
        leftMostEdge = self.__score_values[0] - (self.__score_values[1] - self.__score_values[0])
        self.__bin_edges = np.insert(self.__score_values, 0, leftMostEdge)
        # calculate loss matrix
        self.__lossMatrix = ot.utils.dist0(self.__num_bins)
        self.__lossMatrix /= self.__lossMatrix.max()
        self.__thetas = thetas
        self.__regForOT = regForOT

        self.__fairData = pd.DataFrame()
        self.__rawDataPerGroupAsHistograms = pd.DataFrame(columns=self.__groupNames)
        self.__fairDataAsHistograms = pd.DataFrame(columns=self.__groupNames)

        self.__plotPath = path
        self.__plot = plot

    def __getScoresByGroup(self, data, groups, qual_attr):
        """
        takes a dataset with one item per row and each item has qualifying as well as sensitive
        attributes.
        takes all values from column qual_attr and resorts data such that result contains all scores from
        qual_attr in one column per group of sensitive attributes.

            @param groups:                 all possible groups in data as dataframe. One row contains manifestations
                                           of protected attributes, hence represents a group
                                           example: [[white, male], [white, female], [hispanic, male], [hispanic, female]]
        @param qual_attr:              name of column that contains the quality attribute (only one possible)

        @return: dataframe with group labels as column names and scores per group as column values
        """
        protectedAttributes = groups.columns.values
        result = pd.DataFrame(dtype=float)
        # select all rows that belong to one group
        for _, group in groups.iterrows():
            colName = str(group.values)
            scoresPerGroup = pd.DataFrame(data)
            for prot_attr in protectedAttributes:
                scoresPerGroup = scoresPerGroup.loc[(scoresPerGroup[prot_attr] == group.get(prot_attr))]
            resultCol = pd.DataFrame(data=scoresPerGroup[qual_attr].values, columns=[colName])
            # needs concat to avoid data loss in case new resultCol is longer than already existing result
            # dataframe
            result = pd.concat([result, resultCol], axis=1)
        return result

    def __getDataAsHistograms(self, data, histograms, bin_edges):
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
        groupSizes = self.__rawDataByGroup.count()
        groupSizesPercent = self.__rawDataByGroup.count().divide(groupSizes.sum())

        # compute general barycenter of all score distributions
        total_bary = ot.bregman.barycenter(self.__rawDataPerGroupAsHistograms,
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
        for groupName in self.__rawDataPerGroupAsHistograms:  # build 2-column matrix from group data and general barycenter
            groupMatrix = pd.concat([self.__rawDataPerGroupAsHistograms[groupName], pd.Series(total_bary)], axis=1)  # get corresponding theta
            theta = self.__thetas[self.__rawDataPerGroupAsHistograms.columns.get_loc(groupName)]  # calculate barycenters
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

    def __calculateFairReplacementStrategy(self, group_barycenters):
        # calculate new scores from group barycenters
        groupFairScores = pd.DataFrame(columns=self.__groupNames)
        for groupName in self.__groupNames:
            ot_matrix = ot.emd(self.__rawDataPerGroupAsHistograms[groupName],
                               group_barycenters[groupName],
                               self.__lossMatrix)
#             plt.imshow(ot_matrix)
            # normalize OT matrix such that each row sums up to 1
            norm_vec = np.matmul(ot_matrix, np.ones(ot_matrix.shape[0]))
            inverse_norm_vec = np.reciprocal(norm_vec)
            inverse_norm_vec = np.nan_to_num(inverse_norm_vec, copy=False)
            norm_matrix = np.diag(inverse_norm_vec)
            normalized_ot_matrix = np.matmul(norm_matrix, ot_matrix)

            groupFairScores[groupName] = np.matmul(normalized_ot_matrix, self.__score_values.T)

        if self.__plot:
            self.__plott(groupFairScores, 'fairScoreReplacementStrategy.png')

        return groupFairScores

    def __replaceRawByFairScores(self, groupFairScores):

        def replace(rawData, colName, groupName):
            rawScores = rawData[colName]
            replaced = rawData.copy()
            fairScores = groupFairScores[groupName]
            for index, fairScore in fairScores.iteritems():
                range_left = self.__bin_edges[index]
                range_right = self.__bin_edges[index + 1]
                replaceAtIndex = rawScores.between(range_left, range_right)
                replaced.at[replaceAtIndex, colName] = fairScore
                # TODO: fair scores stehen nicht an der richtigen Stelle, d.h. für Gruppe 00 stehen die
                # fair scores in den vorderen Indexen, aber die Scores fangen erst höher an
            return replaced

        for groupName in groupFairScores.columns:
            self.__fairData = self.__rawData.copy()
            self.__fairData = self.__fairData.groupby(list(self.__protectedAttributes), as_index=False,
                                                    sort=False).apply(replace,
                                                                      colName=self.__qualityAtribute,
                                                                      groupName=groupName)

        if self.__plot:
            fairDataPerGroup = self.__getScoresByGroup(self.__fairData, self.__groups, self.__qualityAtribute)
#             fairDataPerGroup.plot.kde()
#             plt.savefig(self.__plotPath + 'fairScoreDistributionPerGroup.png', dpi=100, bbox_inches='tight')
#             bin_edges = np.linspace(groupFairScores.min().min(), groupFairScores.max().max(), int(self.__num_bins))
            self.__getDataAsHistograms(fairDataPerGroup, self.__fairDataAsHistograms, self.__bin_edges)
            self.__plott(self.__fairDataAsHistograms, 'fairScoresAsHistograms.png')

    def __plott(self, dataframe, filename):
        dataframe.plot(kind='line', use_index=False)
        plt.savefig(self.__plotPath + filename, dpi=100, bbox_inches='tight')

    def continuousFairnessAlgorithm(self):
        """
        TODO: write that algorithm assumes finite float scores, so no floating scores are allowed
        and they have to be represented somehow as integers, otherwise the algorithm doesn't work
        """

        self.__getDataAsHistograms(self.__rawDataByGroup, self.__rawDataPerGroupAsHistograms, self.__bin_edges)
        print(self.__rawDataPerGroupAsHistograms.idxmax())
        if self.__plot:
            self.__plott(self.__rawDataPerGroupAsHistograms, 'rawScoresAsHistograms.png')

        total_bary = self.__getTotalBarycenter()
        group_barycenters = self.__get_group_barycenters(total_bary)

        fairScoreReplacementStrategy = self.__calculateFairReplacementStrategy(group_barycenters)
        self.__replaceRawByFairScores(fairScoreReplacementStrategy)

