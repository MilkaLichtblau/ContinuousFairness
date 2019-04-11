

def statisticalParity(ranking, groupsWithMinprops, protCols):
    """
    checks if a ranking is fair according to the definition of statistical parity (share of a certain
    protected group in the considered ranking should not be lower than the share of that group in dataset

    @param ranking {dataframe}:              each row represents an item with features in columns. Sorted
                                             by column 'score'
    @param groupsWithMinprops {dataframe}:   contains group descriptions in protCols and percentage of
                                             how many items from each group should be in the ranking
                                             in column 'minProps'
    @param protCols {string array}:          names of the columns that contain the protected attributes

    @return: a dict with group names as keys and boolean values, True if ranking contains enough items
             of respective group, False otherwise
    """

    rankingLength = ranking.shape[0]
    groupCounts = ranking.groupby(protCols).size().reset_index(name='counts')
    groupCounts['percentage'] = groupCounts['counts'] / rankingLength
    print(groupCounts)

