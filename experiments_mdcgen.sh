#!/bin/bash

DIR=$1

DEFAULT_KMAX=100
DEFAULT_DIM=128
DEFAULT_DATA=256
DEFAULT_C=50
DEFAULT_CF=4

KNN=false
KNN_INC=false
RNG=false
ALL=false

dataset() {

    for data in 32 64 128 256 512;
    do
        if $KNN || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${data}n-${DEFAULT_CF}cf-${DEFAULT_C}c.dat" "${DEFAULT_KMAX}" "," "knn" >> "mdcgen-knn-hdbscan-dataset.results"
        fi

        if $KNN_INC || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${data}n-${DEFAULT_CF}cf-${DEFAULT_C}c.dat" "${DEFAULT_KMAX}" "," "knn_inc" >> "mdcgen-knn-inc-hdbscan-dataset.results"
        fi
        
        if $RNG || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${data}n-${DEFAULT_CF}cf-${DEFAULT_C}c.dat" "${DEFAULT_KMAX}" "," "rng" >> "mdcgen-rng-hdbscan-dataset.results"
        fi
    done
}

minpoints() {

    for kmax in 20 40 60 80 100 120 140;
    do
        if $KNN || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${DEFAULT_CF}cf-${DEFAULT_C}c.dat" ${kmax} "," "knn" >> "mdcgen-knn-hdbscan-minpoints.results"
        fi

        if $KNN_INC || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${DEFAULT_CF}cf-${DEFAULT_C}c.dat" ${kmax} "," "knn_inc" >> "mdcgen-knn-inc-hdbscan-minpoints.results"
        fi

        if $RNG || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${DEFAULT_CF}cf-${DEFAULT_C}c.dat" ${kmax} "," "rng" >> "mdcgen-rng-hdbscan-minpoints.results"
        fi
    done
}

dimensions() {

    for dim in 16 32 64 128 256;
    do
        if $KNN || $ALL ; then
            python main_experiments.py "${DIR}/${dim}d-${DEFAULT_DATA}n-${DEFAULT_CF}cf-${DEFAULT_C}c.dat" "${DEFAULT_KMAX}" "," "knn" >> "mdcgen-knn-hdbscan-dimensions.results"
        fi

        if $KNN_INC || $ALL ; then
            python main_experiments.py "${DIR}/${dim}d-${DEFAULT_DATA}n-${DEFAULT_CF}cf-${DEFAULT_C}c.dat" "${DEFAULT_KMAX}" "," "knn_inc" >> "mdcgen-knn-inc-hdbscan-dimensions.results"
        fi

        if $RNG || $ALL ; then
            python main_experiments.py "${DIR}/${dim}d-${DEFAULT_DATA}n-${DEFAULT_CF}cf-${DEFAULT_C}c.dat" "${DEFAULT_KMAX}" "," "rng" >> "mdcgen-rng-hdbscan-dimensions.results"
        fi
    done
}


clusters() {

    for c in 10 30 50 70 90;
    do
        if $KNN || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${DEFAULT_CF}cf-${c}c.dat" "${DEFAULT_KMAX}" "," "knn" >> "mdcgen-knn-hdbscan-dimensions.results"
        fi

        if $KNN_INC || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${DEFAULT_CF}cf-${c}c.dat" "${DEFAULT_KMAX}" "," "knn_inc" >> "mdcgen-knn-inc-hdbscan-dimensions.results"
        fi

        if $RNG || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${DEFAULT_CF}cf-${c}c.dat" "${DEFAULT_KMAX}" "," "rng" >> "mdcgen-rng-hdbscan-dimensions.results"
        fi
    done
}


compactness() {

    for cf in 1 3 5 7 9;
    do
        if $KNN || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${cf}cf-${DEFAULT_C}c.dat" "${DEFAULT_KMAX}" "," "knn" >> "mdcgen-knn-hdbscan-dimensions.results"
        fi

        if $KNN_INC || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${cf}cf-${DEFAULT_C}c.dat" "${DEFAULT_KMAX}" "," "knn_inc" >> "mdcgen-knn-inc-hdbscan-dimensions.results"
        fi

        if $RNG || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${cf}cf-${DEFAULT_C}c.dat" "${DEFAULT_KMAX}" "," "rng" >> "mdcgen-rng-hdbscan-dimensions.results"
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
    compactness
    clusters
done

DURATION=$SECONDS

echo "Done!"
echo "Total Duration: $(($DURATION / 3600)) hours, $((($DURATION / 60) % 60)) minutes and $(($DURATION % 60)) seconds elapsed."