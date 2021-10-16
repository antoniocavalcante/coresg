#!/bin/bash

#source copy.sh
source mergefiles.sh
#source replace.sh
source charts.sh

mkdir -p ../gnuplot/figs/png/

for file in ../gnuplot/figs/*.eps
do
    echo "Processing $file"
    filename=$(basename "$file")
    filename=${filename%.*}
    convert -density 600 $file ../gnuplot/figs/png/$filename.png
done
