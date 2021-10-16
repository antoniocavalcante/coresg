#load '../gnuplot/exploration/plot-exploration.gnuplot'
load '../gnuplot/plot-graph-size.gnuplot'

#set output "../gnuplot/figs/graph-size-".exp.".eps"

f(x) = 10000*60

set xlabel "#clusters"

plot "../results/handl-core-".exp.".results.avg" using (column("size")):xtic(1) t "CORE-SG1" w lp ls 1,\
     "../results/handl-core-".exp.".results.avg" using ((column("size") / 2) + (column("size") / column("minpoints"))):xtic(1) t "CORE-SG2 (avg)" w lp ls 2,\
     f(x) dashtype '-' lw 3 lc 'black' title "UB"
