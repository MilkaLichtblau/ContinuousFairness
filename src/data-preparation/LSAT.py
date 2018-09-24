'''
Created on May 28, 2018

@author: mzehlike

protected attributes: sex, race
features: Law School Admission Test (LSAT), grade point average (UGPA)

scores: first year average grade (ZFYA)

excluding for now: region-first, sander-index, first_pf


höchste ID: 27476

Aufteilung in Trainings und Testdaten, 80% Training, 20% Testing, Random Sampling
'''

import pandas as pd
from scipy.stats import stats

CREATE_DATASETS = 0
PATH_TO_DATAFILE = '../../data/law_data.csv.xlsx'


def prepareGenderData():
    data = pd.read_excel(PATH_TO_DATAFILE)
    data = data.drop(columns=['region_first', 'sander_index', 'first_pf', 'race'])

    data['sex'] = data['sex'].replace([2], 0)

    print(data['sex'].value_counts())

    data['LSAT'] = stats.zscore(data['LSAT'])
    data['UGPA'] = stats.zscore(data['UGPA'])

    data = data[['sex', 'LSAT', 'UGPA', 'ZFYA']]
    return data


def prepareOneRaceData(protGroup, nonprotGroup):
    data = pd.read_excel('../../data/law_data.csv.xlsx')
    data = data.drop(columns=['region_first', 'sander_index', 'first_pf', 'sex'])

    data['race'] = data['race'].replace(to_replace=protGroup, value=1)
    data['race'] = data['race'].replace(to_replace=nonprotGroup, value=0)

    data = data[data['race'].isin([0, 1])]

    print(data['race'].value_counts())


    data['LSAT'] = stats.zscore(data['LSAT'])
    data['UGPA'] = stats.zscore(data['UGPA'])

    data = data[['race', 'LSAT', 'UGPA', 'ZFYA']]

    data = data.sort_values(by=['ZFYA'], ascending=False)
    return data


def prepareAllRaceData():
    data = pd.read_excel('../../data/law_data.csv.xlsx')
    data = data.drop(columns=['region_first', 'sander_index', 'first_pf', 'sex'])

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

    return data


def prepareAllInOneData():
    data = pd.read_excel('../../data/law_data.csv.xlsx')
    data = data.drop(columns=['region_first', 'sander_index', 'first_pf'])

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

    return data


if CREATE_DATASETS:
    ######################################################################################
    # GENDER
    ######################################################################################
    data = prepareGenderData()

    ######################################################################################
    # RACE
    ######################################################################################

    data = prepareRaceData('Asian', 'White')
    data = prepareRaceData('Black', 'White')
    data = prepareRaceData('Hispanic', 'White')
    data = prepareRaceData('Mexican', 'White')
    data = prepareRaceData('Puertorican', 'White')

    #######################################################################################
    # ALL IN ONE
    #######################################################################################

    data = prepareAllInOneData()












