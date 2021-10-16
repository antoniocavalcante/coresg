load '../gnuplot/plot-runtime.gnuplot'

set xlabel '#points'

# Time to compute the base graph.
#set yrange [0:600]
set output "../gnuplot/figs/runtime-dataset-total.eps"

plot "../results/handl-core-".exp.".results.avg" using ( (column("graph") + column("msts") ) / 60 ):xtic(1) lt rgb B title "CORE-SG (Parts A and B-I)",\
     "../results/handl-core-inc-".exp.".results.avg" using ( (column("graph") + column("msts") )  / 60 ):xtic(1) lt rgb A title "CORE-SG (Parts A and B-II)",\
     "../results/handl-rng-".exp.".results.avg" using ( (column("graph") + column("msts") ) / 60 ):xtic(1) lt rgb E title "RNG*"


# Average time to compute a single hierarchy.
#set yrange [0:1.5]
#set output "../gnuplot/figs/runtime-dataset-msts.eps"
#set ylabel "Time (sec)"

#plot "../results/handl-core-".exp.".results.avg" using ( ( (column("msts")) /1 ) / 100 ) lt rgb B title "CORE-SG (Part B-I)",\
#     "../results/handl-core-inc-".exp.".results.avg" using ( ( (column("msts")) /1 ) / 100 ):xtic(1) lt rgb A title "CORE-SG (Part B-II)",\
#     "../results/handl-rng-".exp.".results.avg" using ( ( (column("msts")) /1 ) / 100 ):xtic(1) lt rgb E title "RNG*"
