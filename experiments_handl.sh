#!/bin/bash

DEFAULT_KMAX=60
DEFAULT_DIM=32
DEFAULT_DATA=10
DEFAULT_CLUS=30

CORE=false
CORE_INC=false
CORE_STAR=false
RNG=false
ALL=false

dataset() {

    for n in 1 5 10 50 100;
    do
        if $CORE || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${n}n-${DEFAULT_CLUS}c.dat" "${DEFAULT_KMAX}" " " "core" >> "handl-core-dataset.results"
        fi

        if $CORE_INC || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${n}n-${DEFAULT_CLUS}c.dat" "${DEFAULT_KMAX}" " " "core_inc" >> "handl-core-inc-dataset.results"
        fi

        if $CORE_STAR || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${n}n-${DEFAULT_CLUS}c.dat" "${DEFAULT_KMAX}" " " "core_star" >> "handl-core-star-dataset.results"
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
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${DEFAULT_CLUS}c.dat" ${minpoints} " " "core" >> "handl-core-minpoints.results"
        fi

        if $CORE_INC || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${DEFAULT_CLUS}c.dat" ${minpoints} " " "core_inc" >> "handl-core-inc-minpoints.results"
        fi

        if $CORE_STAR || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${DEFAULT_CLUS}c.dat" ${minpoints} " " "core_star" >> "handl-core-star-minpoints.results"
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
            python main_experiments.py "${DIR}/${dim}d-${DEFAULT_DATA}n-${DEFAULT_CLUS}c.dat" "${DEFAULT_KMAX}" " " "core" >> "handl-core-dimensions.results"
        fi

        if $CORE_INC || $ALL ; then
            python main_experiments.py "${DIR}/${dim}d-${DEFAULT_DATA}n-${DEFAULT_CLUS}c.dat" "${DEFAULT_KMAX}" " " "core_inc" >> "handl-core-inc-dimensions.results"
        fi

        if $CORE_STAR || $ALL ; then
            python main_experiments.py "${DIR}/${dim}d-${DEFAULT_DATA}n-${DEFAULT_CLUS}c.dat" "${DEFAULT_KMAX}" " " "core_star" >> "handl-core-star-dimensions.results"
        fi

        if $RNG || $ALL ; then
            python main_experiments.py "${DIR}/${dim}d-${DEFAULT_DATA}n-${DEFAULT_CLUS}c.dat" "${DEFAULT_KMAX}" " " "rng" >> "handl-rng-dimensions.results"
        fi
    done
}


clusters() {

    for clus in 10 30 50;
    do
        if $CORE || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${clus}c.dat" "${DEFAULT_KMAX}" " " "core" >> "handl-core-clusters.results"
        fi

        if $CORE_INC || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${clus}c.dat" "${DEFAULT_KMAX}" " " "core_inc" >> "handl-core-inc-clusters.results"
        fi

        if $CORE_STAR || $ALL ; then
            python main_experiments.py "${DIR}/${DEFAULT_DIM}d-${DEFAULT_DATA}n-${clus}c.dat" "${DEFAULT_KMAX}" " " "core_star" >> "handl-core-star-clusters.results"
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
                        python main_experiments.py "${DIR}/${dim}d-${n}n-${c}c.dat" "${kmax}" " " "core" >> "handl-core-initial-${n}n-${kmax}k-${c}c.results"
                    fi

                    if $CORE_STAR || $ALL ; then
                        python main_experiments.py "${DIR}/${dim}d-${n}n-${c}c.dat" "${kmax}" " " "core_star" >> "handl-core-star-initial-${n}n-${kmax}k-${c}c.results"
                    fi

                    if $RNG || $ALL ; then
                        python main_experiments.py "${DIR}/${dim}d-${n}n-${c}c.dat" "${kmax}" " " "rng" >> "handl-rng-star-initial-${n}n-${kmax}k-${c}c.results"
                    fi
                done
            done
        done
    done
}


speedup() {

    for n in 1 5 50;
    do
        for minpoints in 10 20 30 40 50 60 70 80 90 100;
        do

            python main_experiments.py "${DIR}/32d-${n}n-30c.dat" ${minpoints} " " "single" >> "handl-single-speedup-${n}.results"

            if $CORE || $ALL ; then
                python main_experiments.py "${DIR}/32d-${n}n-30c.dat" ${minpoints} " " "core" >> "handl-core-speedup-${n}.results"
            fi

            if $CORE_INC || $ALL ; then
                python main_experiments.py "${DIR}/32d-${n}n-30c.dat" ${minpoints} " " "core_inc" >> "handl-core-inc-speedup-${n}.results"
            fi

            if $CORE_STAR || $ALL ; then
                python main_experiments.py "${DIR}/32d-${n}n-30c.dat" ${minpoints} " " "core_star" >> "handl-core-star-speedup-${n}.results"
            fi

            if $RNG || $ALL ; then
                python main_experiments.py "${DIR}/32d-${n}n-30c.dat" ${minpoints} " " "rng" >> "handl-rng-speedup-${n}.results"
            fi
        done
    done
}

DIR=$1
MET=${2^^}

if [[ ! -d "$DIR" ]]
then
    echo "[ERROR] The directory $DIR does not exist on your filesystem. Please enter a valid directory."
    exit 1
fi

if [[ $MET == "CORE" || $MET == "ALL" ]]; then
    CORE=true
fi
if [[ $MET == "CORE_INC" || $MET == "ALL" ]]; then
    CORE_INC=true
fi
if [[ $MET == "CORE_STAR" || $MET == "ALL" ]]; then
    CORE_STAR=true
fi
if [[ $MET == "RNG" || $MET == "ALL" ]]; then
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
    speedup
done

DURATION=$SECONDS

echo "Done!"
echo "Total Duration: $(($DURATION / 3600)) hours, $((($DURATION / 60) % 60)) minutes and $(($DURATION % 60)) seconds elapsed."