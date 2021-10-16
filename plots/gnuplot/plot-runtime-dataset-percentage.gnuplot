load '../gnuplot/main-percentage.gnuplot'

set xlabel '#objects'

plot "../results/knn-hdbscan-".exp.".results.avg" using (( (column("msts")) / column("total") ) * 100):xtic(1) lt rgb C title "CORE-SG",\
     "../results/knn-inc-hdbscan-".exp.".results.avg" using (( (column("msts")) / column("total") ) * 100):xtic(1) lt rgb B title "CORE-SG*"
