load '../gnuplot/exploration/plot-exploration.gnuplot'
#load '../gnuplot/plot-graph-size.gnuplot'

set terminal postscript eps enhanced color font 'Helvetica,70' size 11, 8
#set size 1.03, 1.04

#set output "../gnuplot/figs/speedup-handl.eps"

if (!exists("MP_LEFT"))   MP_LEFT = 0.07
if (!exists("MP_RIGHT"))  MP_RIGHT = .98
if (!exists("MP_BOTTOM")) MP_BOTTOM = .15
if (!exists("MP_TOP"))    MP_TOP = .85
if (!exists("MP_xGAP"))   MP_xGAP = 0.05
if (!exists("MP_yGAP"))   MP_yGAP = 0.02

#set multiplot layout 1, 3 rowsfirst \
#    margins screen MP_LEFT, MP_RIGHT, MP_BOTTOM, MP_TOP spacing screen MP_xGAP, MP_yGAP

set ylabel "Speed-up"

#set logscale y

set key font ",50"
set key box width -4 height 1
set key inside top left horizontal maxcols 1

#set format y "10^{%L}"
#set ytics font ", 60"

#set label 1 '#points = 1k' at screen 0.16, 0.93 font "Helvetica,90"
#set label 2 '#points = 5k' at screen 0.47, 0.93 font "Helvetica,90"
#set label 3 '#points = 50k' at screen 0.78, 0.93 font "Helvetica,90"

#set xtics scale -1,-0.5

set xlabel "k_{max}"

set yrange [0:100]
set xrange [10:100]

f(x) = x

set output "../gnuplot/figs/speedup-handl-1k.eps"
set title ""
plot "../results/handl-core-speedup-1.results.avg" using 1:( (column("minpoints") * column("graph")) / (column("graph") + column("msts")) ) t "CORE-SG (Parts A and B-I)" w lp ls 1,\
     "../results/handl-core-inc-speedup-1.results.avg" using 1:( (column("minpoints") * column("graph")) / (column("graph") + column("msts")) ) t "CORE-SG (Parts A and B-II)" w lp ls 2,\
     "../results/handl-rng-speedup-1.results.avg" using 1:( (column("minpoints") * column("hdbscan")) / (column("graph") + column("msts")) ) t "RNG*" w lp ls 3,\
#     f(x) dashtype '-' lw 3 lc 'black' notitle

set output "../gnuplot/figs/speedup-handl-5k.eps"
#unset ylabel
set title ""
plot "../results/handl-core-speedup-5.results.avg" using 1:( (column("minpoints") * column("graph")) / (column("graph") + column("msts")) ) t "CORE-SG (Parts A and B-I)" w lp ls 1,\
     "../results/handl-core-inc-speedup-5.results.avg" using 1:( (column("minpoints") * column("graph")) / (column("graph") + column("msts")) ) t "CORE-SG (Parts A and B-II)" w lp ls 2,\
     "../results/handl-rng-speedup-5.results.avg" using 1:( (column("minpoints") * column("hdbscan")) / (column("graph") + column("msts")) ) t "RNG*" w lp ls 3,\
#     f(x) dashtype '-' lw 3 lc 'black' notitle

set output "../gnuplot/figs/speedup-handl-50k.eps"
#set title ""
plot "../results/handl-core-speedup-50.results.avg" using 1:( (column("minpoints") * column("graph")) / (column("graph") + column("msts")) ) t "CORE-SG (Parts A and B-I)" w lp ls 1,\
     "../results/handl-core-inc-speedup-50.results.avg" using 1:( (column("minpoints") * column("graph")) / (column("graph") + column("msts")) ) t "CORE-SG (Parts A and B-II)" w lp ls 2,\
     "../results/handl-rng-speedup-50.results.avg" using 1:( (column("minpoints") * column("hdbscan")) / (column("graph") + column("msts")) ) t "RNG*" w lp ls 3,\
#     f(x) dashtype '-' lw 3 lc 'black' notitle
