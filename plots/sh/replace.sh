#!/bin/bash

DIR="../results"

# Remove prefix "data#6" from all the result files.
sed -i -e 's/data#6\///g' ${DIR}/*hdbscan*.results.avg

# Removes sufix "-128.dat" from the dimension experiment results.
sed -i -e 's/-128.dat//g' ${DIR}/*hdbscan-dimensions*.results.avg

# Adjust dataset files to be plotted.
sed -i -e 's/64d-//g'  ${DIR}/*hdbscan-dataset*.results.avg
sed -i -e 's/.dat/k/g' ${DIR}/*hdbscan-dataset*.results.avg


DIR="../results"
