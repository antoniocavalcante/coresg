load '../gnuplot/plot-graph-size.gnuplot'

#set output "../gnuplot/figs/graph-size-".exp.".eps"

set xlabel "k_{max}"

plot "../results/handl-core-".exp.".results.avg" using (column("size")):xtic(1) t "CORE-SG1" w lp ls 1,\
     "../results/handl-core-".exp.".results.avg" using ((column("size") / 2)) t "CORE-SG2 (avg)" w lp ls 2#,\
#     "../results/handl-rng-".exp.".results.avg" using (column("size")):xtic(1) t "RNG*" w lp ls 4
