#!/bin/sh

DIR="../results/"

name() {

    output=${4}-"hdbscan"-$1

    if [ $4 = "rng" ]; then

        if [ $2 = true ]; then
            output=$output-"quick"
        fi

        if [ $3 = true ]; then
            output=$output-"naive"
        fi

    fi

    output=$output".results"
}

run() {
    for name in 'minpoints' 'dimensions' 'dataset'
    do
        for graph in 'knn' 'knn-inc' 'rng';
        do
            echo "${DIR}${graph}-hdbscan-${name}.results"
            python3 ../python/mergefiles.py "${DIR}${graph}-hdbscan-${name}.results"
        done
    done

    # python3 ../python/mergefiles.py "${DIR}knn-hdbscan-minpoints++.results"
}

kmaxdim() {
        # DIR="../results/kmaxdim/handl/*.results"
        #
        # for file in $DIR;
        # do
        #     echo $file
        #     python3 ../python/mergefiles.py $file
        # done

        # DIR="../results/kmaxdim/mdcgen/*.results"
        DIR="../results/*.results"

        for file in $DIR;
        do
            echo $file
            python3 ../python/mergefiles.py $file
        done
}

# run
kmaxdim
