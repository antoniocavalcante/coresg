#!/bin/sh

DIR=$1

# rm -rf *.results

DEFAULT_KMAX=64
DEFAULT_DIM=64
DEFAULT_DATA=256


dataset() {

    for n in 8 16 32 64 128 256 512;
    do
        # KNN-HDBSCAN
        # args: dataset, minpoints, separator
        python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${n}.dat" "${DEFAULT_KMAX}" " " "knn" >> "knn-hdbscan-dataset.results"

        python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${n}.dat" "${DEFAULT_KMAX}" " " "knn_inc" >> "knn-inc-hdbscan-dataset.results"

        python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${n}.dat" "${DEFAULT_KMAX}" " " "rng" >> "rng-hdbscan-dataset.results"

    done
}

minpoints() {

    for minpoints in 2 4 8 16 32 64 128;
    do
        # KNN-HDBSCAN
        # args: dataset, minpoints, separator
        python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}.dat" ${minpoints} " " "knn" >> "knn-hdbscan-minpoints.results"

        python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}.dat" ${minpoints} " " "knn_inc" >> "knn-inc-hdbscan-minpoints.results"

        python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}.dat" ${minpoints} " " "rng" >> "rng-hdbscan-minpoints.results"

    done
}

dimensions() {

    for d in 2 4 8 16 32 64 128;
    do
        # KNN-HDBSCAN
        # args: dataset, minpoints, separator
        python main_experiments.py "${DIR}/${d}d-${DEFAULT_DATA}.dat" "${DEFAULT_KMAX}" " " "knn" >> "knn-hdbscan-dimensions.results"

        python main_experiments.py "${DIR}/${d}d-${DEFAULT_DATA}.dat" "${DEFAULT_KMAX}" " " "knn_inc" >> "knn-inc-hdbscan-dimensions.results"

        python main_experiments.py "${DIR}/${d}d-${DEFAULT_DATA}.dat" "${DEFAULT_KMAX}" " " "rng" >> "rng-hdbscan-dimensions.results"

    done
}


hdbscan_single() {

    for minpoints in 2 4 8 16 32 64 128;
    do
        # KNN-HDBSCAN
        # args: dataset, minpoints, separator
        python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}.dat" ${minpoints} " " "single" >> "hdbscan-single-minpoints.results"

        python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}.dat" ${minpoints} " " "single_k" >> "knn-hdbscan-single-minpoints.results"
    done
}


for i in $(seq 3)
do
    # hdbscan_single
	minpoints
	dimensions
	dataset
done
