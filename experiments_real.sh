#!/bin/bash

imagenet() {
    DATA="../dataset-real/imagenet-sample.csv"
    SEPARATOR=" "

    python main_experiments.py "${DATA}" 100 "${SEPARATOR}" "knn" >> "real-data.results"
}


fma_chroma() {
    DATA="../dataset-real/fma_chroma_cens.csv"
    SEPARATOR=","

    python main_experiments.py "${DATA}" 100 "${SEPARATOR}" "knn" >> "real-data.results"
}


fma_mfcc() {
    DATA="../dataset-real/fma_mfcc.csv"
    SEPARATOR=","

    python main_experiments.py "${DATA}" 100 "${SEPARATOR}" "knn" >> "real-data.results"
}



SECONDS=0

for i in $(seq 1)
do
	imagenet
	# fma_chroma
    # fma_mfcc
done

DURATION=$SECONDS

echo "Done!"
echo "Total Duration: $(($DURATION / 3600)) hours, $((($DURATION / 60) % 60)) minutes and $(($DURATION % 60)) seconds elapsed."