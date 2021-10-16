#!/bin/sh

#set -o errexit
#set -o pipefail
#set -o nounset

DIR=$1

# rm -rf *.results

name() {
#    output="fast-rng-hdbscan"-$1
     output="knn-hdbscan"-$1

#    if [ $2 = true ]; then
#        output=$output-"smart"
#    fi

#    if [ $3 = true ]; then
#        output=$output-"naive"
#    fi

#    if [ $2 = true ] || [ $3 = true ]; then
#        if [ $4 = true ]; then
#            output=$output-"incremental"
#        fi
#    fi

#    if [ $5 = true ]; then
#        output=$output-"kdtree"
#    fi

    output=$output".results"
}

dataset() {

    name "dataset" $1 $2 $3 $4

    for n in 16 32 64 128 256 512 1024;
    do
        # kNNG-P-HDBSCAN*
        java -jar -XX:+UseConcMarkSweepGC -XX:+UseParNewGC -Xmx61g KNN-P-HDBSCAN.jar file="${DIR}/16d-${n}.dat" minPts="16" minClSize="0" run="$i" filter="quick" output="false" dist_function="euclidean" separator=" " mrd="star" coreK="16" outputExtension="vis" >> "knn-p-hdbscan-dataset.results"

        # kNNG-F-HDBSCAN*
        java -jar -XX:+UseConcMarkSweepGC -XX:+UseParNewGC -Xmx61g KNN-F-HDBSCAN.jar file="${DIR}/16d-${n}.dat" minPts="16" minClSize="0" run="$i" filter="quick" output="false" dist_function="euclidean" separator=" " mrd="star" coreK="16" outputExtension="vis" >> "knn-f-hdbscan-dataset.results"

        # RNG-HDBSCAN*
        java -jar -XX:+UseConcMarkSweepGC -XX:+UseParNewGC -Xmx61g Fast-RNG-HDBSCAN.jar file="${DIR}/16d-${n}.dat" minPts="16" minClSize="0" run="$i" filter="quick" output="false" dist_function="euclidean" separator=" " mrd="star" coreK="16" outputExtension="vis" >> "rng-hdbscan-dataset.results"

		# HDBSCAN
		# java -jar -Xmx61g -XX:+UseConcMarkSweepGC -XX:+UseParNewGC "${DIR}/16d-${n}.dat" 16 $i true >> hdbscan-dataset.results
    done
}

minpoints() {

    name "minpoints" $1 $2 $3 $4

    for minpoints in 2 4 8 16 32 64 128;
    do
		# kNNG-P-HDBSCAN*
        java -jar -XX:+UseConcMarkSweepGC -XX:+UseParNewGC -Xmx61g KNN-P-HDBSCAN.jar file="${DIR}/16d-128.dat" minPts="${minpoints}" minClSize="0" run="$i" filter="quick" output="false" dist_function="euclidean" separator=" " mrd="star" coreK="${minpoints}" outputExtension="vis" >> "knn-p-hdbscan-minpoints.results"

        # kNNG-F-HDBSCAN*
        java -jar -XX:+UseConcMarkSweepGC -XX:+UseParNewGC -Xmx61g KNN-F-HDBSCAN.jar file="${DIR}/16d-128.dat" minPts="${minpoints}" minClSize="0" run="$i" filter="quick" output="false" dist_function="euclidean" separator=" " mrd="star" coreK="${minpoints}" outputExtension="vis" >> "knn-f-hdbscan-minpoints.results"

        # RNG-HDBSCAN*
		java -jar -XX:+UseConcMarkSweepGC -XX:+UseParNewGC -Xmx61g Fast-RNG-HDBSCAN.jar file="${DIR}/16d-128.dat" minPts="${minpoints}" minClSize="0" run="$i" filter="quick" output="false" dist_function="euclidean" separator=" " mrd="star" coreK="${minpoints}" outputExtension="vis" >> "rng-hdbscan-minpoints.results"

		# HDBSCAN
        # java -jar -Xmx61g -XX:+UseConcMarkSweepGC -XX:+UseParNewGC HDBSCAN.jar "${DIR}/16d-128.dat" ${minpoints} $i true >> hdbscan-minpoints.results
    done
}

dimensions() {

    name "dimensions" $1 $2 $3 $4

    for d in 2 4 8 16 32 64 128;
    do
		# kNN-P-HDBSCAN*
        java -jar -XX:+UseConcMarkSweepGC -XX:+UseParNewGC -Xmx61g KNN-P-HDBSCAN.jar file="${DIR}/${d}d-128.dat" minPts="16" minClSize="0" run="$i" filter="quick" output="false" dist_function="euclidean" separator=" " mrd="star" coreK="16" outputExtension="vis" >> "knn-p-hdbscan-dimensions.results"

        # kNN-F-HDBSCAN*
        java -jar -XX:+UseConcMarkSweepGC -XX:+UseParNewGC -Xmx61g KNN-F-HDBSCAN.jar file="${DIR}/${d}d-128.dat" minPts="16" minClSize="0" run="$i" filter="quick" output="false" dist_function="euclidean" separator=" " mrd="star" coreK="16" outputExtension="vis" >> "knn-f-hdbscan-dimensions.results"

        # RNG-HDBSCAN*
        java -jar -XX:+UseConcMarkSweepGC -XX:+UseParNewGC -Xmx61g Fast-RNG-HDBSCAN.jar file="${DIR}/${d}d-128.dat" minPts="16" minClSize="0" run="$i" filter="quick" output="false" dist_function="euclidean" separator=" " mrd="star" coreK="16" outputExtension="vis" >> "rng-hdbscan-dimensions.results"

        # HDBSCAN
        # java -jar -Xmx61g -XX:+UseConcMarkSweepGC -XX:+UseParNewGC HDBSCAN.jar "${DIR}/${d}d-128.dat" 16 $i true >> hdbscan-dimensions.results
    done
}

index="false"

# SMART + NAIVE
smartFilter="true"
naiveFilter="false"
incremental="false"

for i in $(seq 5)
do
	minpoints $smartFilter $naiveFilter $incremental $index
	dimensions $smartFilter $naiveFilter $incremental $index
	dataset $smartFilter $naiveFilter $incremental $index
done
