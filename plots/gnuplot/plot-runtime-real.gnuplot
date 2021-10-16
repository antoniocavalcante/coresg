load '../gnuplot/plot-runtime.gnuplot'

#set xlabel 'Data Sets'

set terminal postscript eps enhanced color font 'Helvetica,70' size 15, 8

# Time to compute the base graph.
#set yrange [0:1.3]
set output "../gnuplot/figs/runtime-real-total.eps"

#set logscale y

set xtics font ",50"

plot "../results/real-data.results" using ( (column("graph") + column("msts")) / 60 ):xtic(1) lt rgb A title "CORE-SG (Parts A and B-I)",\
     "../results/real-data-rng.results" using ( (column("graph") + column("msts")) / 60 ) lt rgb E title "RNG*"


# Average time to compute a single hierarchy.
#set yrange [0:0.012]
#set output "../gnuplot/figs/runtime-clusters-msts.eps"
#set ylabel "Time (sec)"

#set output "../gnuplot/figs/runtime-real-msts.eps"

#plot "../results/real-data.results" using ( ( (column("msts")) / 1 ) / column("minpoints") ):xtic(1) lt rgb A title "CORE-SG (Part B-I)",\
#     "../results/real-data-rng.results" using ( ( (column("msts")) / 1 ) / column("minpoints") ) lt rgb E title "RNG*"
