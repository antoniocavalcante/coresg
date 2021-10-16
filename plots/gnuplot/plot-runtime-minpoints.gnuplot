load '../gnuplot/plot-runtime.gnuplot'

set xlabel 'k_{max}'

# Time to compute the base graph.
set yrange [0:1.4]
set output "../gnuplot/figs/runtime-minpoints-total.eps"

plot "../results/handl-core-".exp.".results.avg" using ( (column("graph") + column("msts") ) / 60 ):xtic(1) lt rgb B title "CORE-SG (Parts A and B-I)",\
     "../results/handl-core-inc-".exp.".results.avg" using ( (column("graph") + column("msts") )  / 60 ):xtic(1) lt rgb A title "CORE-SG (Parts A and B-II)",\
     "../results/handl-rng-".exp.".results.avg" using ( (column("graph") + column("msts") ) / 60 ):xtic(1) lt rgb E title "RNG*"


# Average time to compute a single hierarchy.
#set yrange [0:0.02]
#set output "../gnuplot/figs/runtime-minpoints-msts.eps"
#set ylabel "Time (sec)"

#plot "../results/handl-core-".exp.".results.avg" using ( ( (column("msts")) /1 ) / column("minpoints") ) lt rgb B title "CORE-SG  (Part B-I)",\
#     "../results/handl-core-inc-".exp.".results.avg" using ( ( (column("msts")) /1 ) / column("minpoints") ):xtic(1) lt rgb A title "CORE-SG  (Part B-II)",\
#     "../results/handl-rng-".exp.".results.avg" using ( ( (column("msts")) /1 ) / column("minpoints") ):xtic(1) lt rgb E title "RNG*"
