'''
Created on Sep 27, 2018

@author: meike.zehlike
'''
import pandas as pd

def scoresByGroups(data, groups, qual_attr):
    """
    takes a dataset with one item each row and each item has qualifying as well as sensitive
    attributes.
    takes all values from column qual_attr and resorts data such that result contains all scores from
    qual_attr in one column per group of sensitive attributes.

    @param groups:                 all possible groups in data as list of tuples. One tuple contains manifestations
                                   of protected attributes, hence represents a group
                                   example: [(white, male), (white, female), (hispanic, male), (hispanic, female)]
    @param qual_attr:              name of column that contains the quality attribute (only one possible)
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
