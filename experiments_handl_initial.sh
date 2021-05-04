#!/bin/bash

DIR=$1

DEFAULT_KMAX=100
DEFAULT_DIM=64
DEFAULT_DATA=256
DEFAULT_CLUS=10

KNN=false
RNG=false
ALL=false

run() {

    for n in 64 #16 32 64;
    do
        for dim in 2 4 8 16 32 64 128;
        do
            for kmax in 10 30 50 70;
            do
                for c in 8;
                    # Example: 2d-64n-8c-no0.dat
                    if $KNN || $ALL ; then
                        python main_experiments.py "${DIR}/${dim}d-${n}n-${c}c-no0.dat" "${kmax}" " " "knn" >> "handl-core-initial-${n}n-${kmax}k.results"
                    fi
                    
                    if $RNG || $ALL ; then
                        python main_experiments.py "${DIR}/${dim}d-${n}n-${c}c-no0.dat" "${kmax}" " " "rng" >> "handl-rng-star-initial-${n}n-${kmax}k.results"
                    fi
                done
            done
        done
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

run

DURATION=$SECONDS

echo "Done!"
echo "Total Duration: $(($DURATION / 3600)) hours, $((($DURATION / 60) % 60)) minutes and $(($DURATION % 60)) seconds elapsed."