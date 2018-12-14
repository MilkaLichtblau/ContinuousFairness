#!/bin/sh
# runs experiments on synthetic dataset and LSAT dataset with different thetas (and also different binsize, if necessary)

python3 main.py --run synthetic 1,100 1 0,0,0,0,0,0 ../data/synthetic/results/theta=0/
python3 main.py --run synthetic 1,100 1 0.5,0.5,0.5,0.5,0.5,0.5 ../data/synthetic/results/theta=0.5/
python3 main.py --run synthetic 1,100 1 1,1,1,1,1,1 ../data/synthetic/results/theta=1/

python3 main.py --run lsat_gender 11,48 1 0,0 ../data/LSAT/gender/results/theta=0/
python3 main.py --run lsat_gender 11,48 1 0.5,0.5 ../data/LSAT/gender/results/theta=0.5/
python3 main.py --run lsat_gender 11,48 1 1,1 ../data/LSAT/gender/results/theta=1/binsize=1/
python3 main.py --run lsat_gender 11,48 2 1,1 ../data/LSAT/gender/results/theta=1/binsize=2/

python3 main.py --run lsat_race 11,48 1 0,0,0,0,0,0,0,0 ../data/LSAT/allRace/results/theta=0/
python3 main.py --run lsat_race 11,48 1 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5 ../data/LSAT/allRace/results/theta=0.5/
python3 main.py --run lsat_race 11,48 1 1,1,1,1,1,1,1,1 ../data/LSAT/allRace/results/theta=1/

python3 main.py --run lsat_all 11,48 2 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ../data/LSAT/all/results/theta=0/
python3 main.py --run lsat_all 11,48 1 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5 ../data/LSAT/all/results/theta=0.5/
python3 main.py --run lsat_all 11,48 2 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 ../data/LSAT/all/results/theta=1/

