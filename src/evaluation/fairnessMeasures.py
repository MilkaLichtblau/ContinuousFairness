import numpy as np
import pandas as pd


def groupPercentageAtK(ranking, groups):
    """
    checks if a ranking is fair according to the definition of statistical parity (share of a certain
    protected group in the considered ranking should not be lower than the share of that group in dataset

    @param ranking {dataframe}:              each row represents an item with features in columns. Sorted
                                             by column 'score'
    @param groups {dataframe}:               group names and their values

    @return: a dict with group names as keys and boolean values, True if ranking contains enough items
             of respective group, False otherwise
    """

    rankingLength = ranking.shape[0]
    numGroups = groups.shape[0]
    groupCounts = ranking.groupby(list(groups)).size().reset_index(name='counts')
    while groupCounts.shape[0] != numGroups:
        rowWithZeros = pd.DataFrame([np.zeros(groupCounts.shape[1])], columns=list(groupCounts))
        groupCounts = groupCounts.append(rowWithZeros, ignore_index=True)
    return (groupCounts['counts'] / rankingLength).values.transpose()
