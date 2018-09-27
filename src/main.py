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


def main():
    createSyntheticData(50000)
    createLSATDatasets()

if __name__ == '__main__':
    main()
