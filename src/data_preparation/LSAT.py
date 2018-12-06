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

        data['LSAT'] = data['LSAT'].apply(str)
        data['LSAT'] = data['LSAT'].str.replace('.7', '.75', regex=False)
        data['LSAT'] = data['LSAT'].str.replace('.3', '.25', regex=False)
        data['LSAT'] = pd.to_numeric(data['LSAT'])

        data = data[['sex', 'LSAT', 'UGPA', 'ZFYA']]
        self.__groups = pd.DataFrame({"sex": [0, 1]})
        self.__dataset = data

    def prepareOneRaceData(self, protGroup, nonprotGroup):
        data = self.__origDataset.drop(columns=['region_first', 'sander_index', 'first_pf', 'sex'])

        data['race'] = data['race'].replace(to_replace=protGroup, value=1)
        data['race'] = data['race'].replace(to_replace=nonprotGroup, value=0)

        data = data[data['race'].isin([0, 1])]

        print(data['race'].value_counts())

        data['LSAT'] = data['LSAT'].apply(str)
        data['LSAT'] = data['LSAT'].str.replace('.7', '.75', regex=False)
        data['LSAT'] = data['LSAT'].str.replace('.3', '.25', regex=False)
        data['LSAT'] = pd.to_numeric(data['LSAT'])

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

        # find all values in column LSAT that end with .7 and replace them with .75
        data['LSAT'] = data['LSAT'].apply(str)
        data['LSAT'] = data['LSAT'].str.replace('.7', '.75', regex=False)
        data['LSAT'] = data['LSAT'].str.replace('.3', '.25', regex=False)
        data['LSAT'] = pd.to_numeric(data['LSAT'])

#         data['LSAT'] = stats.zscore(data['LSAT'])
#         data['UGPA'] = stats.zscore(data['UGPA'])

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

        data['LSAT'] = data['LSAT'].apply(str)
        data['LSAT'] = data['LSAT'].str.replace('.7', '.75', regex=False)
        data['LSAT'] = data['LSAT'].str.replace('.3', '.25', regex=False)
        data['LSAT'] = pd.to_numeric(data['LSAT'])

        data = data[['sex', 'race', 'LSAT', 'UGPA', 'ZFYA']]
        race = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7],
                              index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                              name='race')
        sex = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                              index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                              name='sex')
        self.__groups = pd.concat([sex, race], axis=1)
        self.__dataset = data

    def writeToCSV(self, pathToDataset, pathToGroups):
        self.__dataset.to_csv(pathToDataset, index=False, header=True)
        self.__groups.to_csv(pathToGroups, index=False, header=True)

