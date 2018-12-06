## Experiments ##

### Synthetic Data ###
* two protected features: gender (0,1), ethnicity (0,1,2) 
* total of six groups
* scores are normally distributed integers with different mean and standard deviation for each group (see 
[this diagram](https://github.com/MilkaLichtblau/ContinuousFairness/blob/master/data/synthetic/scoreDistributionPerGroup.png))

| number of groups | thetas | bin size | group barycenters | fair scores |
| --- | --- | --- | --- | --- |
| 6 | 1,1,1,1,1,1 | 1 | [link](https://github.com/MilkaLichtblau/ContinuousFairness/blob/master/data/synthetic/results/theta%3D1/groupBarycenters.png) | [link](https://github.com/MilkaLichtblau/ContinuousFairness/blob/master/data/synthetic/results/theta%3D1/fairScoreDistributionPerGroup.png)|
| 6 | 0,0,0,0,0,0 | 1 |[link]() | [link]() |
| 6 | 0.5,0.5,0.5,0.5,0.5,0.5 | 1 | [link]() | [link]() |

### LSAT data ###
* two protected features: gender (male, female), ethnicity (White, Amerindian, Asian, Black, Hispanic, Mexican, Puertorican)
* total 14 groups
* scores are results from a university entrance test for a law school in the US (CITATION)

| number of groups | thetas | bin size | group barycenters | fair scores |
| --- | --- | --- | --- | --- |
| 2 (gender only) | 0,0 | 1 | [link]() | [link]() |
| 2 (gender only) | 1,1 | 2 | [link]() | [link]() |
| 2 (gender only) | 1,1 | 1 | [link]() | [link]() |
| 2 (gender only) | 0.5,0.5 | 1 | [link]() | [link]() |
| 7 (ethnicity only) | 0,0,0,0,0,0,0 |  1 | [link]() | [link]() |
| 7 (ethnicity only) | 0.5,0.5,0.5,0.5,0.5,0.5,0.5 |  1 | [link]() | [link]() |
| 7 (ethnicity only) | 1,1,1,1,1,1,1 |  1 | [link]() | [link]() |
| 14 (all combined) | 0,0,0,0,0,0,0,0,0,0,0,0,0,0 |  2 | [link]() | [link]() |
| 14 (all combined) | 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5 | 1 | [link]() | [link]() |
| 14 (all combined) | 1,1,1,1,1,1,1,1,1,1,1,1,1,1 | 2 |  [link]() | [link]() |
