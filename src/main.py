import pandas as pd
import numpy as np
import argparse, os, math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from data_preparation import synthetic, LSAT
from visualization.plots import plotKDEPerGroup
from cfa.cfa import ContinuousFairnessAlgorithm
from evaluation.fairnessMeasures import statisticalParity
from evaluation.relevanceMeasures import pak, ndcg_score
from numpy import dtype


def createSyntheticData(size):
    nonProtectedAttributes = ['score']
    protectedAttributes = {"gender": 2, "ethnicity": 3}
    creator = synthetic.SyntheticDatasetCreator(
        size, protectedAttributes, nonProtectedAttributes)
    creator.createTruncatedIntegerScoresNormallyDistributed(1, 101)
    creator.sortByColumn('score')
    creator.writeToCSV('../data/synthetic/dataset.csv',
                       '../data/synthetic/groups.csv')
    plotKDEPerGroup(creator.dataset, creator.groups, 'score',
                    '../data/synthetic/scoreDistributionPerGroup.png', '')


def createLSATDatasets():
    creator = LSAT.LSATCreator('../data/LSAT/law_data.csv.xlsx')
    # gender and ethnicity in one dataset
    creator.prepareAllInOneData()
    creator.writeToCSV('../data/LSAT/all/allInOneLSAT.csv',
                       '../data/LSAT/all/allInOneGroups.csv')
    plotKDEPerGroup(creator.dataset, creator.groups, 'LSAT',
                    '../data/LSAT/all/scoreDistributionPerGroup_All_LSAT', '')
    plotKDEPerGroup(creator.dataset, creator.groups, 'ZFYA',
                    '../data/LSAT/all/scoreDistributionPerGroup_All_ZFYA', '')

    # all ethnicity in one dataset
    creator.prepareAllRaceData()
    creator.writeToCSV('../data/LSAT/allRace/allEthnicityLSAT.csv',
                       '../data/LSAT/allRace/allEthnicityGroups.csv')
    plotKDEPerGroup(creator.dataset, creator.groups, 'LSAT',
                    '../data/LSAT/allRace/scoreDistributionPerGroup_AllRace_LSAT', '')
    plotKDEPerGroup(creator.dataset, creator.groups, 'ZFYA',
                    '../data/LSAT/allRace/scoreDistributionPerGroup_AllRace_ZFYA', '')

    # gender dataset
    creator.prepareGenderData()
    creator.writeToCSV('../data/LSAT/gender/genderLSAT.csv',
                       '../data/LSAT/gender/genderGroups.csv')
    plotKDEPerGroup(creator.dataset, creator.groups, 'LSAT',
                    '../data/LSAT/gender/scoreDistributionPerGroup_Gender_LSAT', '')
    plotKDEPerGroup(creator.dataset, creator.groups, 'ZFYA',
                    '../data/LSAT/gender/scoreDistributionPerGroup_Gender_ZFYA', '')


def rerank_with_cfa(score_stepsize, thetas, result_dir, pathToData,
                    pathToGroups, qual_attr):
    data = pd.read_csv(pathToData, sep=',')
    groups = pd.read_csv(pathToGroups, sep=',')

    # check that we have a theta for each group
    if groups.shape[0] != len(thetas):
        raise ValueError(
            "invalid number of thetas, should be {numThetas} Specify one theta per group.".format(numThetas=groups.shape[0]))

    regForOT = 5e-3

    cfa = ContinuousFairnessAlgorithm(data,
                                      groups,
                                      qual_attr,
                                      score_stepsize,
                                      thetas,
                                      regForOT,
                                      path=result_dir,
                                      plot=True)
    result = cfa.run()
    result.to_csv(result_dir + "resultData.csv")


def parseThetas(thetaString):
    thetas = np.array(thetaString.split(","))
    floatThetas = [float(i) for i in thetas]
    return floatThetas


def evaluate(data, groupsWithMinProb, topk, result_dir, qualAttr):
    protCols = list(groupsWithMinProb)
    protCols.remove('minProps')
    statisticalParity(data.head(topk),
                      groupsWithMinProb,
                      protCols)
    # relevance measures
    stepsize = 5
    ndcgAtK = np.empty(int(math.ceil(data.shape[0] / stepsize)))
    precisionAtK = np.empty(int(math.ceil(data.shape[0] / stepsize)))
    index = 0
    for k in range(0, data.shape[0], stepsize):
        print(k)
        np.put(ndcgAtK,
               index,
               ndcg_score(data[qualAttr].values, data['fairScore'].values, k, gains="linear"))
        np.put(precisionAtK,
               index,
               pak(k + 1, data['newPos'].values, data['oldPos'].values))
        index += 1

    # save result to disk if wanna change plots later
    performanceData = np.stack((ndcgAtK, precisionAtK), axis=-1)
    performanceDataframe = pd.DataFrame(performanceData, columns=['ndcg', 'P$@$k'])
    performanceDataframe.to_csv(result_dir + "performanceEvaluation.csv")

    # plot results
    mpl.rcParams.update({'font.size': 24, 'lines.linewidth': 3,
                         'lines.markersize': 15, 'font.family': 'Times New Roman'})
    mpl.rcParams['ps.useafm'] = True
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = True
    ax = performanceDataframe.plot(kind='line', use_index=False)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  # , labels=self.__groupNamesForPlots)

    tickCount = 5
    tickLabels = np.linspace(1, data.shape[0], tickCount + 1, dtype=int)
    positions = np.arange(0, performanceData.shape[0] + 1, performanceData.shape[0] / tickCount, dtype=int)

    ax.xaxis.set_major_locator(ticker.FixedLocator((positions)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter((tickLabels)))
    ax.set_xticklabels(tickLabels)
    ax.set_xlabel("ranking position")
    ax.set_ylabel("relevance score")
    plt.savefig(result_dir + "relevanceEvaluation.png", dpi=100, bbox_inches='tight')


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
                        metavar=('DATASET', 'TOP-K', 'RESULT DIRECTORY'),
                        help="evaluates all experiments for respective dataset with rankings of top-k length")

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
            rerank_with_cfa(score_stepsize,
                            thetas,
                            result_dir,
                            '../data/synthetic/dataset.csv',
                            '../data/synthetic/groups.csv',
                            'score')
        elif args.run[0] == 'lsat_gender':
            # TODO: run experiments also with ZFYA
            rerank_with_cfa(score_stepsize,
                            thetas,
                            result_dir,
                            '../data/LSAT/gender/genderLSAT.csv',
                            '../data/LSAT/gender/genderGroups.csv',
                            'LSAT')
        elif args.run[0] == 'lsat_race':
            rerank_with_cfa(score_stepsize,
                            thetas,
                            result_dir,
                            '../data/LSAT/allRace/allEthnicityLSAT.csv',
                            '../data/LSAT/allRace/allEthnicityGroups.csv',
                            'LSAT')
        elif args.run[0] == 'lsat_all':
            rerank_with_cfa(score_stepsize,
                            thetas,
                            result_dir,
                            '../data/LSAT/all/allInOneLSAT.csv',
                            '../data/LSAT/all/allInOneGroups.csv',
                            'LSAT')
        else:
            parser.error(
                "unknown dataset. Options are 'synthetic', 'lsat_gender', 'lsat_race, 'lsat_all'")
    elif args.evaluate:
        topk = int(args.evaluate[1])
        pathToCFAResult = args.evaluate[2]
        result_dir = os.path.dirname(pathToCFAResult) + '/'
        if args.evaluate[0] == 'synthetic':
            qualAttr = 'score'
            groups = pd.read_csv('../data/synthetic/groups.csv', sep=',')
            groups['minProps'] = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]

        if args.evaluate[0] == 'lsat_race':
            qualAttr = 'LSAT'
            groups = pd.read_csv('../data/LSAT/allRace/allEthnicityGroups.csv', sep=',')
            # FIXME: adjust min probs
            groups['minProps'] = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 8, 1 / 8]

        if args.evaluate[0] == 'lsat_gender':
            qualAttr = 'LSAT'
            groups = pd.read_csv('../data/LSAT/gender/genderGroups.csv', sep=',')
            # FIXME: adjust min probs
            groups['minProps'] = [1 / 6, 1 / 6]

        data = pd.read_csv(pathToCFAResult, sep=',')
        oldPosColumn = data.index.values
#             plt.plot(oldPosColumn[:1000])
#             plt.show()
        fairSorting = data.rename_axis('idx').sort_values(by=['fairScore', 'idx'], ascending=[False, True])
        fairSorting['newPos'] = fairSorting.index
        fairSorting['oldPos'] = oldPosColumn
#             plt.plot(fairSorting['oldPos'].head(1000).values)
#             plt.plot(fairSorting['newPos'].head(1000).values)
#             plt.show()
        fairSorting = fairSorting.reset_index(drop=True)
        evaluate(fairSorting, groups, topk, result_dir, qualAttr)
    else:
        parser.error("choose one command line option")


if __name__ == '__main__':
    main()
