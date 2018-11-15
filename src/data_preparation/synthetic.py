'''
Created on Oct 3, 2017

@author: meike.zehlike

'''
import numpy as np
import pandas as pd
import random
import itertools
import uuid
from scipy.stats import truncnorm


class SyntheticDatasetCreator(object):

    """
    a dataframe that contains protected and non-protected features in columns. Each row represents
    a candidate with their feature values
    """

    @property
    def dataset(self):
        return self.__dataset

    """
    dataframe with all possible combinations of protected attributes. Each group is an element of the Cartesian
    product of the element set per protected attribute.
    example:   attribute gender has two possible elements {0, 1}, attribute ethnicity has three
               possible elements {0, 1, 2} --> there are six groups
               a group is determined by one of the tuples (0, 0), (0,1), (1, 0), ..., (2, 1)
    the non-protected group is always represented by the tuple with only zeros
    """

    @property
    def groups(self):
        return self.__groups

    """
    list of strings containing all names of protected attributes
    """

    @property
    def protectedAttributes(self):
        return self.__protectedAttributes

    """
    list of strings containing all names of non-protected attributes
    """

    @property
    def nonProtectedAttributes(self):
        return self.__nonProtectedAttributes

    def __init__(self, size, attributeNamesAndCategories, nonProtectedAttributes):
        """
        @param size:                            total number of data points to be created
        @param attributeNamesAndCategories:     dictionary with name of protected attribute as key
                                                and number of possible manifestations as values
                                                e.g. {'gender': 2, 'ethnicity': 5}
        @param nonProtectedAttributes:          list of strings with names of non-protected attributes
        """

        self.__dataset = pd.DataFrame()
        self.__protectedAttributes = attributeNamesAndCategories.keys()
        self.__nonProtectedAttributes = nonProtectedAttributes

        # determine groups of candidates
        self.__determineGroups(attributeNamesAndCategories)

        # generate distribution of protected attributes
        self.__createCategoricalProtectedAttributes(attributeNamesAndCategories, size)

        # generate ID column
        self.__dataset['uuid'] = [uuid.uuid4() for _ in range(len(self.__dataset.index))]

    def createScoresNormalDistribution(self):
        """
        @param nonProtectedAttributes:     a string array that contains the names of the non-protected
                                           features
        """

#         if len(mu_diff) != len(self.groups) or len(sigma) != len(self.groups):
#             raise ValueError("not enough mean and standard deviation values for all groups. Check \
#                               size of self.groups for correct number")
        def score(x, colName):
            mu = 10 * np.random.uniform()
            sigma = np.random.uniform()
            x[colName] = np.random.normal(mu, sigma, size=len(x))
            return x

        for attr in self.nonProtectedAttributes:
            self.__dataset = self.__dataset.groupby(list(self.protectedAttributes), as_index=False,
                                                    sort=False).apply(score, colName=attr)

    def createTruncatedIntegerScoresNormallyDistributed(self, lower, upper):
        """
        creates Integer scores for each social group and each name in self.nonProtectedAttributes in maximum ranges
        of lower and upper

        @param lower:                      total lower bound for generated scores
        @param upper:                      total upper bound for generated scores
        """

        def get_truncated_normal(mean=0, sd=1, low=0, upp=10, size=100):
            return truncnorm.rvs((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd, size=size)

        def score(x, colName):
            low = np.random.randint(lower, upper)
            upp = np.random.randint(lower, upper)
            # check ranges are in right order:
            if low > upp:
                temp = low
                low = upp
                upp = temp
            elif (low == upp):
                if upp != upper:
                    upp += 1
                else:
                    low -= 1
            mu = np.random.randint(low, upp)
            sigma = np.random.randint(low, upp)

            x[colName] = get_truncated_normal(mean=mu, sd=sigma, low=low, upp=upp, size=len(x))
            x[colName] = x[colName].round().astype(int)
            return x

        for attr in self.nonProtectedAttributes:
            self.__dataset = self.__dataset.groupby(list(self.protectedAttributes), as_index=False,
                                                    sort=False).apply(score, colName=attr)

    def createScoresUniformDistribution(self):
        """
        creates uniformly distributed scores for each group in self.dataset
        done for all non-protected attributes (i.e. for all score columns) listed in self.nonProtectedAttributes
        """

        def score(x, colName):
            highest = np.random.uniform()
            x[colName] = np.random.uniform(high=highest, size=x.size)
            return x

        for attr in self.nonProtectedAttributes:
            self.__dataset = self.__dataset.groupby(list(self.protectedAttributes), as_index=False,
                                                    sort=False).apply(score, (attr))

    def writeToCSV(self, pathToDataset, pathToGroups):
        self.__dataset.to_csv(pathToDataset, index=False, header=True)
        self.__groups.to_csv(pathToGroups, index=False, header=True)

    def __determineGroups(self, protectedAttributeNamesAndCategories):
        """
        creates a list with all tuples that represent protected groups, parameters are described in
        protectedAttributeNamesAndCategories

        example:   attribute gender has two possible elements {0, 1}, attribute ethnicity has three
            possible elements {0, 1, 2} --> there are six groups
            a group is determined by one of the tuples (0, 0), (0,1), (1, 0), ..., (2, 1)
        """
        elementSets = []
        for cardinality in protectedAttributeNamesAndCategories.values():
            elementSets.append(list(range(0, cardinality)))

        allGroups = list(itertools.product(*elementSets))
        self.__groups = pd.DataFrame(allGroups, columns=protectedAttributeNamesAndCategories.keys())

    def __createScoresNormalDistributionGroupsSeparated(self, size):
            """
            @param size: expected size of the dataset
            """
            prot_data = pd.DataFrame()
            prot_data['gender'] = np.ones(int(size / 2)).astype(int)
            prot_data['score'] = np.random.normal(0.2, 0.3, size=int(size / 2))

            nonprot_data = pd.DataFrame()
            nonprot_data['gender'] = np.zeros(int(size / 2)).astype(int)
            nonprot_data['score'] = np.random.normal(0.8, 0.3, size=int(size / 2))

            self.__dataset = pd.concat([prot_data, nonprot_data])

            # normalize data
            mini = self.__dataset['score'].min()
            maxi = self.__dataset['score'].max()
            self.__dataset['score'] = (self.__dataset['score'] - mini) / (maxi - mini)

    def __createScoresUniformDistributionGroupsSeparated(self, size):
            """
            @param size:     expected size of the dataset
            """

            prot_data = pd.DataFrame()
            prot_data['gender'] = np.ones(int(size / 2)).astype(int)
            prot_data['score'] = np.random.uniform(high=0.5, low=0.0, size=int(size / 2))

            nonprot_data = pd.DataFrame()
            nonprot_data['gender'] = np.zeros(int(size / 2)).astype(int)
            nonprot_data['score'] = np.random.uniform(high=1.0, low=0.5, size=int(size / 2))

            self.__dataset = pd.concat([prot_data, nonprot_data])

    def __createCategoricalProtectedAttributes(self, attributeNamesAndCategories, size):
        """
        creates columns with manifestations of protected attributes from attributeNamesAndCategories
        e.g. creates a column "gender" containing 0s and 1s for each item in the dataset

        @param attributeNamesAndCategories:         a dictionary that contains the names of the
                                                    protected attributes as keys and the number of
                                                    categories as values
                                                    (e.g. {('ethnicity'; 5), ('gender'; 2)})
        @param size:                                number of items in entire created dataset (all
                                                    protection status)

        @return category zero is assumed to be the non-protected
        """
        newData = pd.DataFrame(columns=self.__protectedAttributes)

        for attributeName in self.__protectedAttributes:
            col = []
            categories = range(0, attributeNamesAndCategories[attributeName])
            for _ in range(0, size):
                col.append(random.choice(categories))
            newData[attributeName] = col

        # add protected columns to dataset
        self.__dataset = self.__dataset.append(newData)

