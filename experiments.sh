#!/bin/sh

DIR=$1

# rm -rf *.results

dataset() {

    for n in 16 32 64 128 256 512 1024;
    do
        # KNN-HDBSCAN
        # args: dataset, minpoints, separator
        python main_experiments.py "${DIR}/16d-${n}.dat" "16" " " "knn" >> "knn-hdbscan-dataset.results"

        python main_experiments.py "${DIR}/16d-${n}.dat" "16" " " "rng" >> "rng-hdbscan-dataset.results"
    done
}

minpoints() {

    for minpoints in 2 4 8 16 32 64 128;
    do
        # KNN-HDBSCAN
        # args: dataset, minpoints, separator
        python main_experiments.py "${DIR}/16d-128.dat" ${minpoints} " " "knn" >> "knn-hdbscan-minpoints.results"

        python main_experiments.py "${DIR}/16d-128.dat" ${minpoints} " " "rng" >> "rng-hdbscan-minpoints.results"
    done
}

dimensions() {

    for d in 2 4 8 16 32 64 128;
    do
        # KNN-HDBSCAN
        # args: dataset, minpoints, separator
        python main_experiments.py "${DIR}/${d}d-128.dat" "16" " " "knn" >> "knn-hdbscan-dimensions.results"

        python main_experiments.py "${DIR}/${d}d-128.dat" "16" " " "rng" >> "rng-hdbscan-dimensions.results"
    done
}

index="false"

# SMART + NAIVE
smartFilter="true"
naiveFilter="true"
incremental="false"

for i in $(seq 5)
do
	minpoints $smartFilter $naiveFilter $incremental $index
	dimensions $smartFilter $naiveFilter $incremental $index
	dataset $smartFilter $naiveFilter $incremental $index
done
