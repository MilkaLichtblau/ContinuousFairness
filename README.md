# Continuous Fairness Algorithm

Implementation of the post-processing algorithm and experiments described in https://link.springer.com/article/10.1007/s10618-019-00658-8

## Dependencies
python 3.6

Python Optimal Transport library: https://github.com/rflamary/POT/tree/81b2796226f3abde29fc024752728444da77509a

## Usage

### Create Datasets
The code supports two datasets so far (see main() in main.py):
* ``python3 main.py --create synthetic``
* ``python3 main.py --create lsat``

The synthetic dataset contains 2 protected feature columns and 1 score column. One protected feature is binary {0, 1}, the other is drawn from the set {0, 1, 2}, leading to a total of 6 groups (0 0), (0 1), (0 2) etc. The score feature represents the number, that would be calculated by a ranking function. Each group is assigned a in integer score within [1,100], drawn from a normal distribution with different means and standard-deviations per group (see data/synthetic/scoreDistributionPerGroup.png)

The LSAT dataset (https://eric.ed.gov/?id=ED469370) has sex and race as protected features and LSAT, ZFYA and UGPA as protected features. Current experiments use LSAT as score feature. The current experimental configuration considers sex and race separately as protected features in two distinct experiments.

During dataset prepararation a csv-file is created that contains information about the groups in the dataset. This file is needed by the actual CFA Algorithm, to apply the barycenter calculation to each group separately. 

### Run Algorithm 

For each dataset the aforementioned group description csv file is needed. It is automatically generated during ``python3 main.py --create.``
Running the CFA requires the following parameters: dataset name, the lowest and highest score value, the step size between two consecutive score values, a theta value for each group, and a path where the results are stored

Examples for the synthetic dataset:
* ``python3 main.py --run synthetic 1,100 1 0,0,0,0,0,0 ../data/synthetic/results/theta=0/``
* ``python3 main.py --run synthetic 1,100 1 1,1,1,1,1,1 ../data/synthetic/results/theta=1/``

Example for LSAT with gender as protected feature:
* ``python3 main.py --run lsat_gender 11,48 1 0,0 ../data/LSAT/gender/results/theta=0/``

Example for LSAT with race as protected feature:
* ``python3 main.py --run lsat_race 11,48 1 1,1,1,1,1,1,1,1 ../data/LSAT/allRace/results/theta=1/``


### Visualize Data and Results
Evaluates relevance and fairness changes for a given experiment and plots the results. Relevance is evaluated in terms of NDCG and Precision@k. Fairness is evaluated in terms of percentage of protected candidates at position k.
Running the evaluation requires the following terminal arguments: dataset name, path to original dataset (before post-processing with CFA), path to result dataset (after applying the CFA). The evaluation files are stored in the same directory as the result dataset.

* ``python3 main.py --evaluate synthetic ../data/synthetic/dataset.csv ../data/synthetic/results/theta=0/resultData.csv``
* ``python3 main.py --evaluate lsat_race ../data/LSAT/allRace/allEthnicityLSAT.csv ../data/LSAT/allRace/results/theta=0/resultData.csv``
* ``python3 main.py --evaluate lsat_gender ../data/LSAT/gender/genderLSAT.csv ../data/LSAT/gender/results/theta=0/resultData.csv``
