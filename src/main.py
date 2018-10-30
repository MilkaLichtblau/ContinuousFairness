'''
Created on Sep 25, 2018

@author: meike.zehlike
'''

import pandas as pd
import numpy as np
import argparse

from data_preparation import *
from visualization.plots import plotKDEPerGroup
from util.util import scoresByGroups
from cfa import cfa
from argparse import ArgumentError


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


def rerank_with_cfa(thetas, result_dir, pathToData, pathToGroups, qual_attr):
    data = pd.read_csv(pathToData, sep=',')
    groups = pd.read_csv(pathToGroups, sep=',')

    # check that we have a theta for each group
    if groups.shape[0] != len(thetas):
        raise ArgumentError("invalid number of thetas. Specify one theta per group.")

    regForOT = 1e-3

    scoresPerGroup = scoresByGroups(data, groups, qual_attr)
    groupSizes = scoresPerGroup.count().divide(data.shape[0])
    cfa.continuousFairnessAlgorithm(scoresPerGroup, groupSizes, thetas, regForOT, path=result_dir, plot=True)


def parseThetas(thetaString):
    thetas = np.array(thetaString.split(","))
    floatThetas = [float(i) for i in thetas]
    return floatThetas


def main():
    # parse command line options
    parser = argparse.ArgumentParser(prog='Continuous Fairness Algorithm',
                                     epilog="=== === === end === === ===")

    parser.add_argument("--create",
                        nargs=1,
                        choices=['synthetic', 'lsat'],
                        help="creates datasets from raw data and writes them to disk")
    parser.add_argument("--run",
                        nargs=3,
                        metavar=('DATASET', 'THETAS', 'DIRECTORY'),
                        help="runs continuous fairness algorithm for given DATASET and THETAS \n \
                              and stores results into DIRECTORY")
    parser.add_argument("--dir",
                        nargs=1,
                        type=str,
                        default="../data/",
                        metavar='DIRECTORY',
                        help="specifies directory to store the results")

    args = parser.parse_args()

    if args.create == ['synthetic']:
        createSyntheticData(50000)
    elif args.create == ['lsat']:
        createLSATDatasets()
    elif args.run:
        thetas = parseThetas(args.run[1])
        result_dir = args.run[2]
        if args.run[0] == 'synthetic':
            rerank_with_cfa(thetas,
                            result_dir,
                            '../data/synthetic/dataset.csv',
                            '../data/synthetic/groups.csv',
                            'score')
        elif args.run[0] == 'LSAT_gender':
            # TODO: run experiments also with ZFYA
            rerank_with_cfa(thetas,
                            result_dir,
                            '../data/LSAT/gender/genderLSAT.csv',
                            '../data/LSAT/gender/genderGroups.csv',
                            'LSAT')
        elif args.run[0] == 'LSAT_race':
            rerank_with_cfa(thetas,
                            result_dir,
                            '../data/LSAT/allRace/allEthnicityLSAT.csv',
                            '../data/LSAT/allRace/allEthnicityGroups.csv',
                            'LSAT')
        elif args.run[0] == 'LSAT_all':
            rerank_with_cfa(thetas,
                            result_dir,
                            '../data/LSAT/all/allInOneLSAT.csv',
                            '../data/LSAT/all/allInOneGroups.csv',
                            'LSAT')
        else:
            parser.error("unknown dataset. Options are 'synthetic', 'LSAT_gender', 'LSAT_race, 'LSAT_all'")
    else:
        parser.error("choose one command line option")


if __name__ == '__main__':
    main()
