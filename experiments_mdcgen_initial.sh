#!/bin/bash

DIR=$1

DEFAULT_KMAX=100
DEFAULT_DIM=64
DEFAULT_DATA=256
DEFAULT_CF=5
DEFAULT_C=80

KNN=false
RNG=false
ALL=false

run() {

    for n in 8 #16 32 64;
    do
        for dim in 2 4 8 16 32 64 128;
        do
            for kmax in 10 30 50;
            do  
                for c in 10 80 640;
                do
                    echo "Running n=${n} dim=${dim} kmax=${kmax} c=${c} cf=${DEFAULT_CF}"
                    if $KNN || $ALL ; then
                        python main_experiments.py "${DIR}/${dim}d-${n}n-${DEFAULT_CF}cf-${c}c.dat" "${kmax}" "," "knn" >> "mdcgen-core-initial-${n}n-${kmax}k-${DEFAULT_CF}cf-${c}c.results"
                    fi

                    if $RNG || $ALL ; then
                        python main_experiments.py "${DIR}/${dim}d-${n}n-${DEFAULT_CF}cf-${c}c.dat" "${kmax}" "," "rng" >> "mdcgen-rng-star-initial-${n}n-${kmax}k-${DEFAULT_CF}cf-${c}c.results"
                    fi
                done

                for cf in 1 5 9;
                do
                    echo "Running n=${n} dim=${dim} kmax=${kmax} c=${DEFAULT_C} cf=${cf}"
                    if $KNN || $ALL ; then
                        python main_experiments.py "${DIR}/${dim}d-${n}n-${cf}cf-${DEFAULT_C}c.dat" "${kmax}" "," "knn" >> "mdcgen-core-initial-${n}n-${kmax}k-${cf}cf-${DEFAULT_C}c.results"
                    fi

                    if $RNG || $ALL ; then
                        python main_experiments.py "${DIR}/${dim}d-${n}n-${cf}cf-${DEFAULT_C}c.dat" "${kmax}" "," "rng" >> "mdcgen-rng-star-initial-${n}n-${kmax}k-${cf}cf-${DEFAULT_C}c.results"
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