'''
Created on Sep 27, 2018

@author: meike.zehlike
'''

import ot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


class ContinuousFairnessAlgorithm():
    """
    TODO: write doc
    TODO: rewrite algorithm with float scores only!!
    """

    def __init__(self, rawData, groups, prot_attr, qual_attr, score_ranges, score_stepsize, thetas, regForOT, path='.', plot=False):
        """
        @param rawData:          pandas dataframe with scores per group, groups are not necessarily
                                 of the same size, i.e. data might contain NaNs
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
        self.__groups = groups  # TODO: das könnte man eleganter lösen, hier ist jetzt einiges doppelt und dreifach...nochmal refactoring machen
        self.__rawDataByGroup = self._getScoresByGroup(self.__rawData)
        self.__groupColumnNames = self.__rawDataByGroup.columns.values.tolist()
        self.__groupNamesForPlots = ['Group ' + f'{index}' for index, _ in enumerate(self.__groupColumnNames)]
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
        self.__rawDataPerGroupAsHistograms = pd.DataFrame(columns=self.__groupColumnNames)
        self.__fairDataAsHistograms = pd.DataFrame(columns=self.__groupColumnNames)

        self.__plotPath = path
        self.__plot = plot

    def _getScoresByGroup(self, dataset):
        """
        takes a dataset with one item per row and each item has qualifying as well as sensitive
        attributes.
        takes all values from column qual_attr and resorts data such that result contains all scores from
        qual_attr in one column per group of sensitive attributes.

        TODO: correct doc

        @param groups:                 all possible groups in data as dataframe. One row contains manifestations
                                           of protected attributes, hence represents a group
                                           example: [[white, male], [white, female], [hispanic, male], [hispanic, female]]
        @param qual_attr:              name of column that contains the quality attribute (only one possible)

        @return: dataframe with group labels as column names and scores per group as column values
        """
        protectedAttributes = self.__groups.columns.values
        result = pd.DataFrame(dtype=float)
        # select all rows that belong to one group
        for _, group in self.__groups.iterrows():
            colName = str(group.values)
            copy = dataset.copy()
            for prot_attr in protectedAttributes:
                copy = copy.loc[(copy[prot_attr] == group.get(prot_attr))]
            resultCol = pd.DataFrame(data=copy[self.__qualityAtribute].values, columns=[colName])
            # needs concat to avoid data loss in case new resultCol is longer than already existing result
            # dataframe
            result = pd.concat([result, resultCol], axis=1)
        return result

    def _getDataAsHistograms(self, data, histograms, bin_edges):
        # get histogram from each column and save to new dataframe
        # gets as parameters pointers to raw data and new dataframe where to save the histograms
        for colName in data.columns:
            colNoNans = pd.DataFrame(data[colName][~np.isnan(data[colName])])
            colAsHist = np.histogram(colNoNans[colName], bins=bin_edges, density=True)[0]
            histograms[colName] = colAsHist

        if histograms.isnull().values.any():
            raise ValueError("Histogram data contains nans")

    def _getTotalBarycenter(self):
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
            self.__plott(pd.DataFrame(total_bary),
                         'totalBarycenter.png',
                         xLabel="raw score")

        return total_bary

    def _get_group_barycenters(self, total_bary):
        # compute barycenters between general barycenter and each score distribution (i.e. each social group)
        group_barycenters = pd.DataFrame(columns=self.__groupColumnNames)
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
            self.__plott(group_barycenters,
                         'groupBarycenters.png',
                         xLabel="raw score")
        return group_barycenters

    def _calculateFairReplacementStrategy(self, group_barycenters):
        # calculate new scores from group barycenters
        groupFairScores = pd.DataFrame(columns=self.__groupColumnNames)
        for groupName in self.__groupColumnNames:
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

            # this contains a vector per group with len(score_values) entries (e.g. a score range from 1 to 100
            # results into a group fair score vector of length 100

            groupFairScores[groupName] = np.matmul(normalized_ot_matrix, self.__score_values.T)

        if self.__plot:
            self.__plott(groupFairScores,
                         'fairScoreReplacementStrategy.png',
                         xLabel="raw score",
                         yLabel="fair replacement")

        return groupFairScores

    def _replaceRawByFairScores(self, groupFairScores):
        '''
        TODO: write how raw scores are replaced by fair scores from groupFairScores matrix
        '''

        def buildGroupNameFromValues(dataRow):
            name = "["
            firstIter = True
            for attr in self.__protectedAttributes:
                if firstIter:
                    name += str(int(dataRow.iloc[0][attr]))
                    firstIter = False
                else:
                    name += " " + str(int(dataRow.iloc[0][attr]))
            name += "]"
            return name

        def replace(rawData, colName):
            rawScores = rawData[colName]
            groupName = buildGroupNameFromValues(rawData.head(1))
            replaced = rawData.copy()
            fairScores = groupFairScores[groupName]
            for index, fairScore in fairScores.iteritems():
                range_left = self.__bin_edges[index]
                range_right = self.__bin_edges[index + 1]
                replaceAtIndex = (rawScores > range_left) & (rawScores <= range_right)
                replaced.at[replaceAtIndex, colName] = fairScore
            return replaced

        self.__fairData = self.__rawData.copy()
        self.__fairData = self.__fairData.groupby(list(self.__protectedAttributes),
                                                  as_index=False,
                                                  sort=False).apply(replace,
                                                                    colName=self.__qualityAtribute)
        self.__fairData = self.__fairData.reset_index(drop=True)

        if self.__plot:
            mpl.rcParams.update({'font.size': 24, 'lines.linewidth': 3, 'lines.markersize': 15, 'font.family':'Times New Roman'})
            # avoid type 3 (i.e. bitmap) fonts in figures
            mpl.rcParams['ps.useafm'] = True
            mpl.rcParams['pdf.use14corefonts'] = True
            mpl.rcParams['text.usetex'] = True

            fairDataPerGroup = self._getScoresByGroup(self.__fairData)
            ax = fairDataPerGroup.plot.kde()
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., labels=self.__groupNamesForPlots)
            ax.set_xlabel("fair score")
            plt.savefig(self.__plotPath + 'fairScoreDistributionPerGroup.png', dpi=100, bbox_inches='tight')

            bin_edges = np.linspace(groupFairScores.min().min(), groupFairScores.max().max(), int(self.__num_bins))
            self._getDataAsHistograms(fairDataPerGroup, self.__fairDataAsHistograms, bin_edges)
            self.__plott(self.__fairDataAsHistograms,
                         'fairScoresAsHistograms.png',
                         xLabel="fair score",
                         yLabel="Density")

        return self.__fairData

    def __plott(self, dataframe, filename, xLabel="", yLabel=""):
        mpl.rcParams.update({'font.size': 24, 'lines.linewidth': 3, 'lines.markersize': 15, 'font.family':'Times New Roman'})
    # avoid type 3 (i.e. bitmap) fonts in figures
        mpl.rcParams['ps.useafm'] = True
        mpl.rcParams['pdf.use14corefonts'] = True
        mpl.rcParams['text.usetex'] = True

        ax = dataframe.plot(kind='line', use_index=False)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., labels=self.__groupNamesForPlots)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        plt.savefig(self.__plotPath + filename, dpi=100, bbox_inches='tight')

    def continuousFairnessAlgorithm(self):
        """
        TODO: write that algorithm assumes finite float scores, otherwise the algorithm doesn't work
        """

        self._getDataAsHistograms(self.__rawDataByGroup, self.__rawDataPerGroupAsHistograms, self.__bin_edges)
        print(self.__rawDataPerGroupAsHistograms.idxmax())
        if self.__plot:
            self.__plott(self.__rawDataPerGroupAsHistograms,
                         'rawScoresAsHistograms.png',
                         xLabel="raw score",
                         yLabel="Density")

        total_bary = self._getTotalBarycenter()
        group_barycenters = self._get_group_barycenters(total_bary)

        fairScoreReplacementStrategy = self._calculateFairReplacementStrategy(group_barycenters)
        self._replaceRawByFairScores(fairScoreReplacementStrategy)

