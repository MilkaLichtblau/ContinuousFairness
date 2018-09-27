'''
Created on Sep 25, 2018

@author: meike.zehlike
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
from util import util


def plotKDEPerGroup(data, groups, score_attr, filename, labels):

    mpl.rcParams.update({'font.size': 24, 'lines.linewidth': 3, 'lines.markersize': 15, 'font.family':'Times New Roman'})
    # avoid type 3 (i.e. bitmap) fonts in figures
    mpl.rcParams['ps.useafm'] = True
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = True

    scoresPerGroup = util.scoresByGroups(data, groups, score_attr)
    scoresPerGroup.plot.kde()
    score_attr = score_attr.replace('_', '\_')

    plt.xlabel(score_attr)
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.savefig(filename, dpi=100, bbox_inches='tight')

