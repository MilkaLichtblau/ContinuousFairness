'''
Created on Sep 25, 2018

@author: meike.zehlike
'''

from data_preparation import *
from visualization.plots import plotKDEPerGroup


def createSyntheticData(size):
    nonProtectedAttributes = ['score']
    protectedAttributes = {"gender": 2, "ethnicity": 3}
    creator = synthetic.SyntheticDatasetCreator(size, protectedAttributes, nonProtectedAttributes)
    creator.createScoresNormalDistribution(nonProtectedAttributes)
    creator.writeToCSV('../data/synthetic/dataset.csv')
    plotKDEPerGroup(creator.dataset, creator.groups, 'score', '../data/synthetic/scoreDistributionPerGroup', '')


def createLSATDatasets():
    creator = LSAT.LSATCreator('../data/LSAT/law_data.csv.xlsx')
    # gender and ethnicity in one dataset
    creator.prepareAllInOneData()
    creator.writeToCSV('../data/LSAT/allInOneLSAT.csv')
    plotKDEPerGroup(creator.dataset, creator.groups, 'LSAT', '../data/LSAT/scoreDistributionPerGroup_All_LSAT', '')
    plotKDEPerGroup(creator.dataset, creator.groups, 'ZFYA', '../data/LSAT/scoreDistributionPerGroup_All_ZFYA', '')

    # all ethnicity in one dataset
    creator.prepareAllRaceData()
    creator.writeToCSV('../data/LSAT/allEthnicityLSAT.csv')
    plotKDEPerGroup(creator.dataset, creator.groups, 'LSAT', '../data/LSAT/scoreDistributionPerGroup_AllRace_LSAT', '')
    plotKDEPerGroup(creator.dataset, creator.groups, 'ZFYA', '../data/LSAT/scoreDistributionPerGroup_AllRace_ZFYA', '')

    # gender dataset
    creator.prepareGenderData()
    creator.writeToCSV('../data/LSAT/genderLSAT.csv')
    plotKDEPerGroup(creator.dataset, creator.groups, 'LSAT', '../data/LSAT/scoreDistributionPerGroup_Gender_LSAT', '')
    plotKDEPerGroup(creator.dataset, creator.groups, 'ZFYA', '../data/LSAT/scoreDistributionPerGroup_Gender_ZFYA', '')


def main():
    createSyntheticData(10000)
    createLSATDatasets()

if __name__ == '__main__':
    main()
