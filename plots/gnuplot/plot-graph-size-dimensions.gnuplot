load '../gnuplot/plot-graph-size.gnuplot'

set xlabel "#dimensions"

f(x) = 10000*60

plot "../results/handl-core-".exp.".results.avg" using (column("size")):xtic(1) t "CORE-SG1" w lp ls 1,\
     "../results/handl-core-".exp.".results.avg" using ((column("size") / 2) + 128000) t "CORE-SG2 (avg)" w lp ls 2,\
     f(x) dashtype '-' lw 3 lc 'black' title "UB"

#     "../results/handl-rng-".exp.".results.avg" using (column("size")):xtic(1) t "RNG*" w lp ls 4,\
