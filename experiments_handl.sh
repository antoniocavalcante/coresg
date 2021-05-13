#!/bin/bash

DIR=$1

DEFAULT_KMAX=60
DEFAULT_DIM=32
DEFAULT_DATA=10
DEFAULT_CLUS=30

CORE=false
CORE_INC=false
RNG=false
ALL=false

dataset() {

    for n in 1 5 10 50 100;
    do
        if $CORE || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${n}n-${DEFAULT_CLUS}c.dat" "${DEFAULT_KMAX}" " " "knn" >> "handl-core-dataset.results"
        fi

        if $CORE_INC || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${n}n-${DEFAULT_CLUS}c.dat" "${DEFAULT_KMAX}" " " "knn_inc" >> "handl-core-inc-dataset.results"
        fi
        
        if $RNG || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${n}n-${DEFAULT_CLUS}c.dat" "${DEFAULT_KMAX}" " " "rng" >> "handl-rng-dataset.results"
        fi
    done
}

minpoints() {

    for minpoints in 20 40 60 80 100;
    do
        if $CORE || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${DEFAULT_CLUS}c.dat" ${minpoints} " " "knn" >> "handl-core-minpoints.results"
        fi

        if $CORE_INC || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${DEFAULT_CLUS}c.dat" ${minpoints} " " "knn_inc" >> "handl-core-inc-minpoints.results"
        fi

        if $RNG || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${DEFAULT_CLUS}c.dat" ${minpoints} " " "rng" >> "handl-rng-minpoints.results"
        fi
    done
}

dimensions() {

    for dim in 4 8 16 32 64 128;
    do
        if $CORE || $ALL ; then
            python main_experiments.py "${DIR}/${dim}d-${DEFAULT_DATA}n-${DEFAULT_CLUS}c.dat" "${DEFAULT_KMAX}" " " "knn" >> "handl-core-dimensions.results"
        fi

        if $CORE_INC || $ALL ; then
            python main_experiments.py "${DIR}/${dim}d-${DEFAULT_DATA}n-${DEFAULT_CLUS}c.dat" "${DEFAULT_KMAX}" " " "knn_inc" >> "handl-core-inc-dimensions.results"
        fi

        if $RNG || $ALL ; then
            python main_experiments.py "${DIR}/${dim}d-${DEFAULT_DATA}n-${DEFAULT_CLUS}c.dat" "${DEFAULT_KMAX}" " " "rng" >> "handl-rng-dimensions.results"
        fi
    done
}


clusters() {

    for clus in 10 20 30;
    do
        if $CORE || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${clus}c.dat" "${DEFAULT_KMAX}" " " "knn" >> "handl-core-clusters.results"
        fi

        if $CORE_INC || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${clus}c.dat" "${DEFAULT_KMAX}" " " "knn_inc" >> "handl-core-inc-clusters.results"
        fi

        if $RNG || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${clus}c.dat" "${DEFAULT_KMAX}" " " "rng" >> "handl-rng-clusters.results"
        fi
    done
}

initial() {

    for n in 5 10 50;
    do
        for dim in 4 8 16 32 64 128;
        do
            for kmax in 10 30 50;
            do
                for c in 10 30 50;
                do
                    if $CORE || $ALL ; then
                        python main_experiments.py "${DIR}/${dim}d-${n}n-${c}c.dat" "${kmax}" " " "knn" >> "handl-core-initial-${n}n-${kmax}k-${c}c.results"
                    fi
                    
                    if $RNG || $ALL ; then
                        python main_experiments.py "${DIR}/${dim}d-${n}n-${c}c.dat" "${kmax}" " " "rng" >> "handl-rng-star-initial-${n}n-${kmax}k-${c}c.results"
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

if [[ ${2^^} == "CORE" || $2 == "ALL" ]]; then
    CORE=true
fi
if [[ ${2^^} == "CORE_INC" || $2 == "ALL" ]]; then
    CORE_INC=true
fi
if [[ ${2^^} == "RNG" || $2 == "ALL" ]]; then
    RNG=true
fi

ALL=[[ $CORE && $CORE_INC && $RNG ]]

SECONDS=0
for i in $(seq 1)
do
	dimensions
	minpoints
	dataset
    clusters
    initial
done

DURATION=$SECONDS

echo "Done!"
echo "Total Duration: $(($DURATION / 3600)) hours, $((($DURATION / 60) % 60)) minutes and $(($DURATION % 60)) seconds elapsed."