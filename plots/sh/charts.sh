#!/bin/bash

DIR="../gnuplot"

kmaxdim () {
    # Handl Datasets
    gnuplot ${DIR}/exploration/plot-graph-size-exploration-multiplot-handl-c.gnuplot
    gnuplot ${DIR}/exploration/plot-runtime-exploration-multiplot-handl-c.gnuplot
}

runtime() {
    for x in "minpoints" "dimensions" "dataset" "clusters";
    do
        gnuplot -e "exp='${x}'" ${DIR}/plot-runtime-${x}.gnuplot
        # gnuplot -e "exp='${x}'" ${DIR}/plot-graph-size-${x}.gnuplot
    done
}

speedup() {
    # fix scale to make axes the same.
    # plot

    gnuplot ${DIR}/plot-speedup.gnuplot
}

real() {
    gnuplot ${DIR}/plot-runtime-real.gnuplot
}

kmaxdim
runtime
speedup
real
