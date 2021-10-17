#!/bin/bash

DIR="../dataset-real"

MPTS=60
SEPARATOR=","

CORE=false
CORE_INC=false
CORE_STAR=false
RNG=false
ALL=false

run_performance() {

    for data in "fma_chroma_cens" "fma_mfcc" "20news_1000d" "20news_500d_pca" "imagenet_sample";
    do
        if $CORE || $ALL ; then
            python main_experiments.py "${DIR}/${data}.csv" "${MPTS}" ${SEPARATOR} "core" >> "real-data-core.results"
        fi

        if $CORE_INC || $ALL ; then
            python main_experiments.py "${DIR}/${data}.csv" "${MPTS}" ${SEPARATOR} "core_inc" >> "real-data-core-inc.results"
        fi

        if $CORE_STAR || $ALL ; then
            python main_experiments.py "${DIR}/${data}.csv" "${MPTS}" ${SEPARATOR} "core_star" >> "real-data-core-star.results"
        fi

        if $RNG || $ALL ; then
            python main_experiments.py "${DIR}/${data}.csv" "${MPTS}" ${SEPARATOR} "rng" >> "real-data-rng.results"
        fi
    done
}

run_speedup() {

    for data in "fma_chroma_cens" "fma_mfcc" "20news_1000d" "20news_500d_pca" "imagenet_sample";
    do
        # python main_experiments.py "${DIR}/${data}.csv" 60 "${SEPARATOR}" "single_k" >> "real-data-single-k.results"    

        python main_experiments.py "${DIR}/${data}.csv" 60 "${SEPARATOR}" "single_k_star" >> "real-data-single-k-star.results"

        # python main_experiments.py "${DIR}/${data}.csv" 60 "${SEPARATOR}" "single" >> "real-data-single.results"
    done
}

MET=${1^^}

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
    run_performance

    run_speedup
done

DURATION=$SECONDS

echo "Done!"
echo "Total Duration: $(($DURATION / 3600)) hours, $((($DURATION / 60) % 60)) minutes and $(($DURATION % 60)) seconds elapsed."