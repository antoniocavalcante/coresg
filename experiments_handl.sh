#!/bin/bash

DIR=$1

DEFAULT_KMAX=60
DEFAULT_DIM=32
DEFAULT_DATA=10
DEFAULT_CLUS=30

KNN=false
KNN_INC=false
RNG=false
ALL=false

dataset() {

    for n in 1 5 10 50 100;
    do
        if $KNN || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${n}n-${DEFAULT_CLUS}c.dat" "${DEFAULT_KMAX}" " " "knn" >> "knn-hdbscan-dataset.results"
        fi

        if $KNN_INC || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${n}n-${DEFAULT_CLUS}c.dat" "${DEFAULT_KMAX}" " " "knn_inc" >> "knn-inc-hdbscan-dataset.results"
        fi
        
        if $RNG || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${n}n-${DEFAULT_CLUS}c.dat" "${DEFAULT_KMAX}" " " "rng" >> "rng-hdbscan-dataset.results"
        fi
    done
}

minpoints() {

    for minpoints in 20 40 60 80 100;
    do
        if $KNN || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${DEFAULT_CLUS}c.dat" ${minpoints} " " "knn" >> "knn-hdbscan-minpoints.results"
        fi

        if $KNN_INC || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${DEFAULT_CLUS}c.dat" ${minpoints} " " "knn_inc" >> "knn-inc-hdbscan-minpoints.results"
        fi

        if $RNG || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${DEFAULT_CLUS}c.dat" ${minpoints} " " "rng" >> "rng-hdbscan-minpoints.results"
        fi
    done
}

dimensions() {

    for dim in 4 8 16 32 64 128;
    do
        if $KNN || $ALL ; then
            python main_experiments.py "${DIR}/${dim}d-${DEFAULT_DATA}n-${DEFAULT_CLUS}c.dat" "${DEFAULT_KMAX}" " " "knn" >> "knn-hdbscan-dimensions.results"
        fi

        if $KNN_INC || $ALL ; then
            python main_experiments.py "${DIR}/${dim}d-${DEFAULT_DATA}n-${DEFAULT_CLUS}c.dat" "${DEFAULT_KMAX}" " " "knn_inc" >> "knn-inc-hdbscan-dimensions.results"
        fi

        if $RNG || $ALL ; then
            python main_experiments.py "${DIR}/${dim}d-${DEFAULT_DATA}n-${DEFAULT_CLUS}c.dat" "${DEFAULT_KMAX}" " " "rng" >> "rng-hdbscan-dimensions.results"
        fi
    done
}


clusters() {

    for clus in 10 20 30;
    do
        if $KNN || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${clus}c.dat" "${DEFAULT_KMAX}" " " "knn" >> "knn-hdbscan-clusters.results"
        fi

        if $KNN_INC || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${clus}c.dat" "${DEFAULT_KMAX}" " " "knn_inc" >> "knn-inc-hdbscan-clusters.results"
        fi

        if $RNG || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${clus}c.dat" "${DEFAULT_KMAX}" " " "rng" >> "rng-hdbscan-clusters.results"
        fi
    done
}


if [[ ! -d "$1" ]]
then
    echo "[ERROR] The directory $1 does not exist on your filesystem. Please enter a valid directory."
    exit 1
fi

if [[ ${2^^} == "KNN" || $2 == "ALL" ]]; then
    KNN=true
fi
if [[ ${2^^} == "KNN_INC" || $2 == "ALL" ]]; then
    KNN_INC=true
fi
if [[ ${2^^} == "RNG" || $2 == "ALL" ]]; then
    RNG=true
fi

ALL=[[ $KNN && $KNN_INC && $RNG ]]

SECONDS=0
for i in $(seq 1)
do
	dimensions
	minpoints
	dataset
    clusters
done

DURATION=$SECONDS

echo "Done!"
echo "Total Duration: $(($DURATION / 3600)) hours, $((($DURATION / 60) % 60)) minutes and $(($DURATION % 60)) seconds elapsed."