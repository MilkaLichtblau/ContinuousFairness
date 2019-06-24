import pandas as pd
import numpy as np
import argparse, os, math
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import process_time

from data_preparation import synthetic, LSAT
from visualization.plots import plotKDEPerGroup
from cfa.cfa import ContinuousFairnessAlgorithm
from evaluation.fairnessMeasures import groupPercentageAtK
from evaluation.relevanceMeasures import pak, ndcg_score


def createSyntheticData(size):
    nonProtectedAttributes = ['score']
    protectedAttributes = {"gender": 2, "ethnicity": 3}
    creator = synthetic.SyntheticDatasetCreator(
        size, protectedAttributes, nonProtectedAttributes)
    creator.createTruncatedIntegerScoresNormallyDistributed(1, 101)
    creator.sortByColumn('score')
    creator.writeToCSV('../data/synthetic/dataset.csv',
                       '../data/synthetic/groups.csv')
    groupNames = {"[0 0]":"Group 1",
                  "[0 1]":"Group 2",
                  "[0 2]":"Group 3",
                  "[1 0]":"Group 4",
                  "[1 1]":"Group 5",
                  "[1 2]":"Group 6"}
    plotKDEPerGroup(creator.dataset, creator.groups, 'score',
                    '../data/synthetic/scoreDistributionPerGroup.png', groupNames)


def createLSATDatasets():
    creator = LSAT.LSATCreator('../data/LSAT/law_data.csv.xlsx')

    # all ethnicity in one dataset
    creator.prepareAllRaceData()
    creator.writeToCSV('../data/LSAT/allRace/allEthnicityLSAT.csv',
                       '../data/LSAT/allRace/allEthnicityGroups.csv')
    groupNames = {"[0]":"White",
                  "[1]":"Amerindian",
                  "[2]":"Asian",
                  "[3]":"Black",
                  "[4]":"Hispanic",
                  "[5]":"Mexican",
                  "[6]":"Other",
                  "[7]":"Puertorican"}
    plotKDEPerGroup(creator.dataset, creator.groups, 'LSAT',
                    '../data/LSAT/allRace/scoreDistributionPerGroup_AllRace_LSAT', groupNames)
    plotKDEPerGroup(creator.dataset, creator.groups, 'ZFYA',
                    '../data/LSAT/allRace/scoreDistributionPerGroup_AllRace_ZFYA', groupNames)

    # gender dataset
    creator.prepareGenderData()
    creator.writeToCSV('../data/LSAT/gender/genderLSAT.csv',
                       '../data/LSAT/gender/genderGroups.csv')
    groupNames = {"[0]":"Male",
                  "[1]":"Female"}
    plotKDEPerGroup(creator.dataset, creator.groups, 'LSAT',
                    '../data/LSAT/gender/scoreDistributionPerGroup_Gender_LSAT', groupNames)
    plotKDEPerGroup(creator.dataset, creator.groups, 'ZFYA',
                    '../data/LSAT/gender/scoreDistributionPerGroup_Gender_ZFYA', groupNames)


def rerank_with_cfa(score_stepsize, thetas, result_dir, pathToData, pathToGroups, qual_attr, group_names):
    data = pd.read_csv(pathToData, sep=',')
    groups = pd.read_csv(pathToGroups, sep=',')

    # check that we have a theta for each group
    if groups.shape[0] != len(thetas):
        raise ValueError(
            "invalid number of thetas, should be {numThetas} Specify one theta per group.".format(numThetas=groups.shape[0]))

    regForOT = 5e-3
    t = process_time()
    cfa = ContinuousFairnessAlgorithm(data,
                                      groups,
                                      group_names,
                                      qual_attr,
                                      score_stepsize,
                                      thetas,
                                      regForOT,
                                      path=result_dir,
                                      plot=True)
    result = cfa.run()
    elapsed_time = process_time() - t
    result.to_csv(result_dir + "resultData.csv")

    print('running time: ' + str(elapsed_time), file=open(result_dir + "runtime.txt", "a"))


def parseThetas(thetaString):
    thetas = np.array(thetaString.split(","))
    floatThetas = [float(i) for i in thetas]
    return floatThetas


def precisionAtKPrep(origData, fairData, qualAttr):
    origSorting = origData.sort_values(by=[qualAttr, 'uuid'], ascending=[False, True])
    origSorting = origSorting.reset_index(drop=True)

    fairSorting = fairData.sort_values(by=['fairScore', 'uuid'], ascending=[False, True])
    fairSorting = fairSorting.reset_index(drop=True)

    return origSorting['uuid'].values, fairSorting['uuid'].values


def ndcgPrep(fairData):
    ndcgData = fairData.sort_values(by=['fairScore', 'uuid'], ascending=[False, True])
    ndcgData = ndcgData.reset_index(drop=True)
    return ndcgData


def evaluateRelevance(origData, fairData, result_dir, qualAttr, stepsize, calcResult=0):

    ndcgAtK = np.empty(int(math.ceil(fairData.shape[0] / stepsize)))
    precisionAtK = np.empty(int(math.ceil(fairData.shape[0] / stepsize)))
    kAtK = np.empty(int(math.ceil(fairData.shape[0] / stepsize)))
    index = 0

    ndcgData = ndcgPrep(fairData)
    pakOrigData, pakFairData = precisionAtKPrep(origData, fairData, qualAttr)
    for k in range(0, fairData.shape[0], stepsize):
        print(k)
        # relevance measures
        np.put(ndcgAtK,
               index,
               ndcg_score(ndcgData[qualAttr].values, ndcgData['fairScore'].values, k, gains="linear"))
        np.put(precisionAtK,
               index,
               pak(k + 1, pakOrigData, pakFairData))
        np.put(kAtK,
               index,
               k)
        index += 1

    # save result to disk if wanna change plots later
    performanceData = np.stack((kAtK, ndcgAtK, precisionAtK), axis=-1)
    performanceDataframe = pd.DataFrame(performanceData, columns=['pos', 'ndcg', 'P$@$k'])
    performanceDataframe = performanceDataframe.set_index('pos')
    performanceDataframe.to_csv(result_dir + "relevanceEvaluation_stepsize=" + str(stepsize) + ".csv")
    performanceDataframe = pd.read_csv(result_dir + "relevanceEvaluation_stepsize=" + str(stepsize) + ".csv")
    performanceDataframe = performanceDataframe.set_index('pos')

    # plot results
    mpl.rcParams.update({'font.size': 24, 'lines.linewidth': 3,
                         'lines.markersize': 15, 'font.family': 'Times New Roman'})
    mpl.rcParams['ps.useafm'] = True
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = True
    ax = performanceDataframe.plot(y=['ndcg', 'P$@$k'],
                                   kind='line',
                                   use_index=True,
                                   yticks=np.arange(0.0, 1.1, 0.2),
                                   rot=45)
    ax.legend(bbox_to_anchor=(1.05, 1),
              loc=2,
              borderaxespad=0.)  # , labels=self.__groupNamesForPlots)
    ax.set_xlabel("ranking position")
    ax.set_ylabel("relevance score")
    plt.savefig(result_dir + "relevanceEvaluation_stepsize=" + str(stepsize) + ".png", dpi=100, bbox_inches='tight')


def evaluateFairness(data, groups, groupNames, result_dir, stepsize):
    """
    evaluates fairness of rankings resulting from cfa algorithm
    """

    index = 0
    percAtK = np.empty(shape=(int(math.ceil(data.shape[0] / stepsize)), len(groups)))
    kAtK = np.empty(int(math.ceil(data.shape[0] / stepsize)))
    data = ndcgPrep(data)
    for k in range(0, data.shape[0], stepsize):
        print(k)
        percAtK[index] = groupPercentageAtK(data.head(k + 1), groups)
        kAtK[index] = k
        index += 1

    # save result to disk if wanna change plots later
    fairnessData = np.c_[kAtK.T, percAtK]
    colNames = ['pos'] + groupNames
    fairnessDataframe = pd.DataFrame(fairnessData, columns=colNames)
    fairnessDataframe = fairnessDataframe.set_index('pos')
    fairnessDataframe.to_csv(result_dir + "fairnessEvaluation_stepsize=" + str(stepsize) + ".csv")
    fairnessDataframe = pd.read_csv(result_dir + "fairnessEvaluation_stepsize=" + str(stepsize) + ".csv")
    fairnessDataframe = fairnessDataframe.set_index('pos')

    # plot results
    mpl.rcParams.update({'font.size': 24, 'lines.linewidth': 3,
                         'lines.markersize': 15, 'font.family': 'Times New Roman'})
    mpl.rcParams['ps.useafm'] = True
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = True
    ax = fairnessDataframe.plot(y=groupNames,
                                kind='line',
                                use_index=True,
                                rot=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_xlabel("ranking position")
    ax.set_ylabel("percentage")
    plt.savefig(result_dir + "fairnessEvaluation_stepsize=" + str(stepsize) + ".png", dpi=100, bbox_inches='tight')


def main():
    # parse command line options
    parser = argparse.ArgumentParser(prog='Continuous Fairness Algorithm',
                                     epilog="=== === === end === === ===")

    parser.add_argument("--create",
                        nargs=1,
                        choices=['synthetic', 'lsat'],
                        help="creates datasets from raw data and writes them to disk")
    parser.add_argument("--run",
                        nargs=4,
                        metavar=('DATASET', 'STEPSIZE', 'THETAS', 'DIRECTORY'),
                        help="runs continuous fairness algorithm for given DATASET with \
                              STEPSIZE and THETAS and stores results into DIRECTORY")
    parser.add_argument("--evaluate",
                        nargs=3,
                        metavar=('DATASET', 'PATH TO ORIG DATASET', 'RESULT DATASET'),
                        help="evaluates all experiments for respective dataset and \
                              stores results into RESULT DIRECTORY")

    args = parser.parse_args()

    if args.create == ['synthetic']:
        createSyntheticData(100000)
    elif args.create == ['lsat']:
        createLSATDatasets()
    elif args.run:
        score_stepsize = float(args.run[1])
        thetas = parseThetas(args.run[2])
        result_dir = args.run[3]
        if args.run[0] == 'synthetic':
            groupNames = {"[0 0]": "Group 1",
                          "[0 1]": "Group 2",
                          "[0 2]": "Group 3",
                          "[1 0]": "Group 4",
                          "[1 1]": "Group 5",
                          "[1 2]": "Group 6"}
            rerank_with_cfa(score_stepsize,
                            thetas,
                            result_dir,
                            '../data/synthetic/dataset.csv',
                            '../data/synthetic/groups.csv',
                            'score',
                            groupNames)
        elif args.run[0] == 'lsat_gender':
            # TODO: run experiments also with ZFYA
            groupNames = {"[0]": "Male",
                          "[1]": "Female"}
            rerank_with_cfa(score_stepsize,
                            thetas,
                            result_dir,
                            '../data/LSAT/gender/genderLSAT.csv',
                            '../data/LSAT/gender/genderGroups.csv',
                            'LSAT',
                            groupNames)
        elif args.run[0] == 'lsat_race':
            groupNames = {"[0]":"White",
                          "[1]":"Amerindian",
                          "[2]":"Asian",
                          "[3]":"Black",
                          "[4]":"Hispanic",
                          "[5]":"Mexican",
                          "[6]":"Other",
                          "[7]":"Puertorican"}
            rerank_with_cfa(score_stepsize,
                            thetas,
                            result_dir,
                            '../data/LSAT/allRace/allEthnicityLSAT.csv',
                            '../data/LSAT/allRace/allEthnicityGroups.csv',
                            'LSAT',
                            groupNames)
        else:
            parser.error("unknown dataset. Options are 'synthetic', 'lsat_gender', 'lsat_race'")
    elif args.evaluate:
        pathToOrigData = args.evaluate[1]
        pathToCFAResult = args.evaluate[2]
        result_dir = os.path.dirname(pathToCFAResult) + '/'
        if args.evaluate[0] == 'synthetic':
            qualAttr = 'score'
            groups = pd.read_csv('../data/synthetic/groups.csv', sep=',')
            groupNames = ["Group 1", "Group 2", "Group 3", "Group 4", "Group 5", "Group 6"]

        if args.evaluate[0] == 'lsat_race':
            qualAttr = 'LSAT'
            groups = pd.read_csv('../data/LSAT/allRace/allEthnicityGroups.csv', sep=',')
            groupNames = ["White", "Amerindian", "Asian", "Black", "Hispanic", "Mexican", "Other", "Puertorican"]

        if args.evaluate[0] == 'lsat_gender':
            qualAttr = 'LSAT'
            groups = pd.read_csv('../data/LSAT/gender/genderGroups.csv', sep=',')
            groupNames = ["Male", "Female"]

        origData = pd.read_csv(pathToOrigData, sep=',')
        fairData = pd.read_csv(pathToCFAResult, sep=',')

        score_stepsize = 1000

        evaluateRelevance(origData, fairData, result_dir, qualAttr, score_stepsize)
        evaluateFairness(fairData, groups, groupNames, result_dir, score_stepsize)
    else:
        parser.error("choose one command line option")


if __name__ == '__main__':
    main()
