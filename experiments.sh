#!/bin/bash

DIR=$1

DEFAULT_KMAX=100
DEFAULT_DIM=64
DEFAULT_DATA=256

KNN=false
KNN_INC=false
RNG=false
ALL=false

dataset() {

    for n in 8 16 32 64 128 256 512;
    do
        if $KNN || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${n}.dat" "${DEFAULT_KMAX}" " " "knn" >> "knn-hdbscan-dataset.results"
        fi

        if $KNN_INC || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${n}.dat" "${DEFAULT_KMAX}" " " "knn_inc" >> "knn-inc-hdbscan-dataset.results"
        fi
        
        if $RNG || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${n}.dat" "${DEFAULT_KMAX}" " " "rng" >> "rng-hdbscan-dataset.results"
        fi
    done
}

minpoints() {

    for minpoints in 20 40 60 80 100 120 140;
    do
        if $KNN || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}.dat" ${minpoints} " " "knn" >> "knn-hdbscan-minpoints.results"
        fi

        if $KNN_INC || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}.dat" ${minpoints} " " "knn_inc" >> "knn-inc-hdbscan-minpoints.results"
        fi

        if $RNG || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}.dat" ${minpoints} " " "rng" >> "rng-hdbscan-minpoints.results"
        fi
    done
}

dimensions() {

    for dim in 2 4 8 16 32 64 128;
    do
        if $KNN || $ALL ; then
            python main_experiments.py "${DIR}/${dim}d-${DEFAULT_DATA}.dat" "${DEFAULT_KMAX}" " " "knn" >> "knn-hdbscan-dimensions.results"
        fi

        if $KNN_INC || $ALL ; then
            python main_experiments.py "${DIR}/${dim}d-${DEFAULT_DATA}.dat" "${DEFAULT_KMAX}" " " "knn_inc" >> "knn-inc-hdbscan-dimensions.results"
        fi

        if $RNG || $ALL ; then
            python main_experiments.py "${DIR}/${dim}d-${DEFAULT_DATA}.dat" "${DEFAULT_KMAX}" " " "rng" >> "rng-hdbscan-dimensions.results"
        fi
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


imagenet() {
    # DATA="/home/toni/Datasets/ImageNet/Imagenet16_train_npz/imagenet.data"
    DATA="/home/toni/Datasets/ImageNet/Imagenet16_train_npz/train_data_batch_1.csv"
    SEPARATOR=" "

    python main_experiments.py "${DATA}" 128 "${SEPARATOR}" "knn" >> "imagenet.results"
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
	minpoints
	dimensions
	dataset
done

DURATION=$SECONDS

echo "Done!"
echo "Total Duration: $(($DURATION / 3600)) hours, $((($DURATION / 60) % 60)) minutes and $(($DURATION % 60)) seconds elapsed."