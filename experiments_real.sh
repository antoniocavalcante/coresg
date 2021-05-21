#!/bin/bash

imagenet() {
    DATA="../dataset-real/imagenet_sample.csv"
    SEPARATOR=" "

    python main_experiments.py "${DATA}" 60 "${SEPARATOR}" "knn" >> "real-data.results"

    python main_experiments.py "${DATA}" 60 "${SEPARATOR}" "rng" >> "real-data-rng.results"
}


fma_chroma() {
    DATA="../dataset-real/fma_chroma_cens.csv"
    SEPARATOR=","

    python main_experiments.py "${DATA}" 60 "${SEPARATOR}" "knn" >> "real-data.results"

    python main_experiments.py "${DATA}" 60 "${SEPARATOR}" "rng" >> "real-data-rng.results"
}


fma_mfcc() {
    DATA="../dataset-real/fma_mfcc.csv"
    SEPARATOR=","

    python main_experiments.py "${DATA}" 60 "${SEPARATOR}" "knn" >> "real-data.results"

    python main_experiments.py "${DATA}" 60 "${SEPARATOR}" "rng" >> "real-data-rng.results"
}


20news_1000() {
    DATA="../dataset-real/20news_1000d.csv"
    SEPARATOR=","

    python main_experiments.py "${DATA}" 60 "${SEPARATOR}" "knn" >> "real-data.results"

    python main_experiments.py "${DATA}" 60 "${SEPARATOR}" "rng" >> "real-data-rng.results"
}

20news_500() {
    DATA="../dataset-real/20news_500d_pca.csv"
    SEPARATOR=","

    python main_experiments.py "${DATA}" 60 "${SEPARATOR}" "knn" >> "real-data.results"

    python main_experiments.py "${DATA}" 60 "${SEPARATOR}" "rng" >> "real-data-rng.results"
}

SECONDS=0

speedup() {

    DIR="../dataset-real"
    SEPARATOR=","

    for data in "fma_chroma_cens" "fma_mfcc" "20news_1000d" "20news_500d_pca";
    do
        python main_experiments.py "${DIR}/${data}.csv" 60 "${SEPARATOR}" "single_k" >> "real-data-single-k.results"    

        python main_experiments.py "${DIR}/${data}.csv" 60 "${SEPARATOR}" "single" >> "real-data-single.results"
    done
}

speedup_imagenet() {

    DIR="../dataset-real"
    SEPARATOR=" "

    for data in "imagenet_sample";
    do
        python main_experiments.py "${DIR}/${data}.csv" 60 "${SEPARATOR}" "single_k" >> "real-data-single-k.results"    

        python main_experiments.py "${DIR}/${data}.csv" 60 "${SEPARATOR}" "single" >> "real-data-single.results"
    done
}


for i in $(seq 1)
do
	fma_chroma
    fma_mfcc
    20news_500
    20news_1000
    imagenet
    
    #speedup
    #speedup_imagenet
done

DURATION=$SECONDS

echo "Done!"
echo "Total Duration: $(($DURATION / 3600)) hours, $((($DURATION / 60) % 60)) minutes and $(($DURATION % 60)) seconds elapsed."