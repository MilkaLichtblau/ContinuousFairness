'''
Created on Sep 25, 2018

@author: meike.zehlike
'''

import pandas as pd

from data_preparation import *
from visualization.plots import plotKDEPerGroup
from util.util import scoresByGroups
from cfa import cfa


def createSyntheticData(size):
    nonProtectedAttributes = ['score']
    protectedAttributes = {"gender": 2, "ethnicity": 3}
    creator = synthetic.SyntheticDatasetCreator(size, protectedAttributes, nonProtectedAttributes)
    creator.createScoresNormalDistribution(nonProtectedAttributes)
    creator.writeToCSV('../data/synthetic/dataset.csv', '../data/synthetic/groups.csv')
    plotKDEPerGroup(creator.dataset, creator.groups, 'score', '../data/synthetic/scoreDistributionPerGroup', '')


def createLSATDatasets():
    creator = LSAT.LSATCreator('../data/LSAT/law_data.csv.xlsx')
    # gender and ethnicity in one dataset
    creator.prepareAllInOneData()
    creator.writeToCSV('../data/LSAT/all/allInOneLSAT.csv', '../data/LSAT/all/allInOneGroups.csv')
    plotKDEPerGroup(creator.dataset, creator.groups, 'LSAT', '../data/LSAT/all/scoreDistributionPerGroup_All_LSAT', '')
    plotKDEPerGroup(creator.dataset, creator.groups, 'ZFYA', '../data/LSAT/all/scoreDistributionPerGroup_All_ZFYA', '')

    # all ethnicity in one dataset
    creator.prepareAllRaceData()
    creator.writeToCSV('../data/LSAT/allRace/allEthnicityLSAT.csv', '../data/LSAT/allRace/allEthnicityGroups.csv')
    plotKDEPerGroup(creator.dataset, creator.groups, 'LSAT', '../data/LSAT/allRace/scoreDistributionPerGroup_AllRace_LSAT', '')
    plotKDEPerGroup(creator.dataset, creator.groups, 'ZFYA', '../data/LSAT/allRace/scoreDistributionPerGroup_AllRace_ZFYA', '')

    # gender dataset
    creator.prepareGenderData()
    creator.writeToCSV('../data/LSAT/gender/genderLSAT.csv', '../data/LSAT/gender/genderGroups.csv')
    plotKDEPerGroup(creator.dataset, creator.groups, 'LSAT', '../data/LSAT/gender/scoreDistributionPerGroup_Gender_LSAT', '')
    plotKDEPerGroup(creator.dataset, creator.groups, 'ZFYA', '../data/LSAT/gender/scoreDistributionPerGroup_Gender_ZFYA', '')


def rerank_with_cfa(thetas, pathToData, pathToGroups, qual_attr):
    data = pd.read_csv(pathToData, sep=',')
    groups = pd.read_csv(pathToGroups, sep=',')
    regForOT = 1e-4

    scoresPerGroup = scoresByGroups(data, groups, qual_attr)
    groupSizes = scoresPerGroup.count().divide(data.shape[0])
    cfa.continuousFairnessAlgorithm(scoresPerGroup, groupSizes, thetas, regForOT, 100, path='../data/synthetic/', plot=True)


def main():
#     createSyntheticData(50000)
#     createLSATDatasets()
    # TODO: make thetas and paths command line arguments

    thetas = [1, 1, 1, 1, 1, 1]
    rerank_with_cfa(thetas, '../data/synthetic/dataset.csv', '../data/synthetic/groups.csv', 'score')

    # TODO: make paths command line argument, also make protected and non-protected attributes command line arguments


if __name__ == '__main__':
    main()
