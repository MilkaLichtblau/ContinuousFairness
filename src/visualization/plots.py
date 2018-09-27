'''
Created on Sep 25, 2018

@author: meike.zehlike
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd


def scoresByGroups(data, groups, qual_attr):
    """
    @param groups:                 all possible groups in data as list of tuples. One tuple contains manifestations
                                   of protected attributes, hence represents a group
                                   example: [(white, male), (white, female), (hispanic, male), (hispanic, female)]
    @param qual_attr:              name of column that contains the quality attribute or score
    @return: dataframe with group labels as column names and scores per group as column values
    """
    protectedAttributes = groups.columns.values
    result = pd.DataFrame()
    # select all rows that belong to one group
    for idx, group in groups.iterrows():
        colName = str(group.values)
        scoresPerGroup = pd.DataFrame(data)
        for prot_attr in protectedAttributes:
            scoresPerGroup = scoresPerGroup.loc[(scoresPerGroup[prot_attr] == group.get(prot_attr))]
        scoresPerGroup = scoresPerGroup.reset_index(drop=True)
        result[colName] = scoresPerGroup[qual_attr]
    return result


def plotKDEPerGroup(data, groups, score_attr, filename, labels):

    mpl.rcParams.update({'font.size': 24, 'lines.linewidth': 3, 'lines.markersize': 15, 'font.family':'Times New Roman'})
    # avoid type 3 (i.e. bitmap) fonts in figures
    mpl.rcParams['ps.useafm'] = True
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = True

    scoresPerGroup = scoresByGroups(data, groups, score_attr)
    scoresPerGroup.plot.kde()
    score_attr = score_attr.replace('_', '\_')

    plt.xlabel(score_attr)
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.savefig(filename, dpi=100, bbox_inches='tight')

