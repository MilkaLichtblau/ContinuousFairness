'''
Created on May 28, 2018

@author: mzehlike

protected attributes: sex, race
features: Law School Admission Test (LSAT), grade point average (UGPA)

scores: first year average grade (ZFYA)

excluding for now: region-first, sander-index, first_pf


höchste ID: 27476

'''

import pandas as pd
from scipy.stats import stats

class LSATCreator():

    @property
    def dataset(self):
        return self.__dataset


    @property
    def groups(self):
        return self.__groups


    def __init__(self, pathToDataFile):
        self.__origDataset = pd.read_excel(pathToDataFile)


    def prepareGenderData(self):
        data = self.__origDataset.drop(columns=['region_first', 'sander_index', 'first_pf', 'race'])

        data['sex'] = data['sex'].replace([2], 0)

        print(data['sex'].value_counts())

        data['LSAT'] = stats.zscore(data['LSAT'])
        data['UGPA'] = stats.zscore(data['UGPA'])

        data = data[['sex', 'LSAT', 'UGPA', 'ZFYA']]
        self.__groups = pd.DataFrame({"sex": [0, 1]})
        self.__dataset = data


    def prepareOneRaceData(self, protGroup, nonprotGroup):
        data = self.__origDataset.drop(columns=['region_first', 'sander_index', 'first_pf', 'sex'])

        data['race'] = data['race'].replace(to_replace=protGroup, value=1)
        data['race'] = data['race'].replace(to_replace=nonprotGroup, value=0)

        data = data[data['race'].isin([0, 1])]

        print(data['race'].value_counts())


        data['LSAT'] = stats.zscore(data['LSAT'])
        data['UGPA'] = stats.zscore(data['UGPA'])

        data = data[['race', 'LSAT', 'UGPA', 'ZFYA']]

        data = data.sort_values(by=['ZFYA'], ascending=False)
        self.__groups = pd.DataFrame({"race": [0, 1]})
        self.__dataset = data


    def prepareAllRaceData(self):
        data = self.__origDataset.drop(columns=['region_first', 'sander_index', 'first_pf', 'sex'])

        data['race'] = data['race'].replace(to_replace="White", value=0)
        data['race'] = data['race'].replace(to_replace="Amerindian", value=1)
        data['race'] = data['race'].replace(to_replace="Asian", value=2)
        data['race'] = data['race'].replace(to_replace="Black", value=3)
        data['race'] = data['race'].replace(to_replace="Hispanic", value=4)
        data['race'] = data['race'].replace(to_replace="Mexican", value=5)
        data['race'] = data['race'].replace(to_replace="Other", value=6)
        data['race'] = data['race'].replace(to_replace="Puertorican", value=7)

        data['LSAT'] = stats.zscore(data['LSAT'])
        data['UGPA'] = stats.zscore(data['UGPA'])

        data = data[['race', 'LSAT', 'UGPA', 'ZFYA']]
        self.__groups = pd.DataFrame({"race": [0, 1, 2, 3, 4, 5, 6, 7]})
        self.__dataset = data


    def prepareAllInOneData(self):
        data = self.__origDataset.drop(columns=['region_first', 'sander_index', 'first_pf'])

        data['sex'] = data['sex'].replace([2], 0)

        data['race'] = data['race'].replace(to_replace="White", value=0)
        data['race'] = data['race'].replace(to_replace="Amerindian", value=1)
        data['race'] = data['race'].replace(to_replace="Asian", value=2)
        data['race'] = data['race'].replace(to_replace="Black", value=3)
        data['race'] = data['race'].replace(to_replace="Hispanic", value=4)
        data['race'] = data['race'].replace(to_replace="Mexican", value=5)
        data['race'] = data['race'].replace(to_replace="Other", value=6)
        data['race'] = data['race'].replace(to_replace="Puertorican", value=7)

        data['LSAT'] = stats.zscore(data['LSAT'])
        data['UGPA'] = stats.zscore(data['UGPA'])

        data = data[['sex', 'race', 'LSAT', 'UGPA', 'ZFYA']]
        race = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7],
                              index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                              name='race')
        sex = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                              index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                              name='sex')
        self.__groups = pd.concat([sex, race], axis=1)
        self.__dataset = data


    def writeToCSV(self, path):
        self.__dataset.to_csv(path, index=False, header=False)


# if CREATE_DATASETS:
#     ######################################################################################
#     # GENDER
#     ######################################################################################
#     data = prepareGenderData()
#
#     ######################################################################################
#     # RACE
#     ######################################################################################
#
#     data = prepareOneRaceData('Asian', 'White')
#     data = prepareOneRaceData('Black', 'White')
#     data = prepareOneRaceData('Hispanic', 'White')
#     data = prepareOneRaceData('Mexican', 'White')
#     data = prepareOneRaceData('Puertorican', 'White')
#
#     #######################################################################################
#     # ALL IN ONE
#     #######################################################################################
#
#     data = prepareAllInOneData()












