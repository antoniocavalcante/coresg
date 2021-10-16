load '../gnuplot/exploration/plot-exploration.gnuplot'

set terminal postscript eps enhanced color font 'Helvetica,70' size 38, 25
#set size 1.03, 1.04

set output "../gnuplot/figs/graph-size-exploration-handl.eps"

if (!exists("MP_LEFT"))   MP_LEFT = 0.13
if (!exists("MP_RIGHT"))  MP_RIGHT = .95
if (!exists("MP_BOTTOM")) MP_BOTTOM = .15
if (!exists("MP_TOP"))    MP_TOP = .9
if (!exists("MP_xGAP"))   MP_xGAP = 0.05
if (!exists("MP_yGAP"))   MP_yGAP = 0.02

set multiplot layout 3, 3 rowsfirst \
    margins screen MP_LEFT, MP_RIGHT, MP_BOTTOM, MP_TOP spacing screen MP_xGAP, MP_yGAP


set logscale y

set format y "10^{%L}"
set ytics font ", 60"

#set yrange [10000 : 10000000]

set label 1 '#points = 5k' at screen 0.20, 0.93 font "Helvetica,90"
set label 2 '#points = 10k' at screen 0.49, 0.93 font "Helvetica,90"
set label 3 '#points = 50k' at screen 0.78, 0.93 font "Helvetica,90"

set label 5 '#clusters = 10' at screen 0.05,0.72 rotate by 90 font "Helvetica,90"
set label 6 '#clusters = 30' at screen 0.05,0.44 rotate by 90 font "Helvetica,90"
set label 7 '#clusters = 50' at screen 0.05,0.20 rotate by 90 font "Helvetica,90"

set xlabel "#dimensions"

unset xtics
unset xlabel

set title ""
unset key
plot "../results/handl-core-initial-5n-10k-10c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 10)" w lp ls 1,\
     "../results/handl-rng-star-initial-5n-10k-10c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 10)" w lp ls 2,\
     "../results/handl-core-initial-5n-30k-10c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 30)" w lp ls 3,\
     "../results/handl-rng-star-initial-5n-30k-10c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 30)" w lp ls 4,\
     "../results/handl-core-initial-5n-50k-10c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 50)" w lp ls 5,\
     "../results/handl-rng-star-initial-5n-50k-10c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 50)" w lp ls 6

unset ylabel
#unset ytics

set title ""
unset key
plot "../results/handl-core-initial-10n-10k-10c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 10)" w lp ls 1,\
     "../results/handl-rng-star-initial-10n-10k-10c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 10)" w lp ls 2,\
     "../results/handl-core-initial-10n-30k-10c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 30)" w lp ls 3,\
     "../results/handl-rng-star-initial-10n-30k-10c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 30)" w lp ls 4,\
     "../results/handl-core-initial-10n-50k-10c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 50)" w lp ls 5,\
     "../results/handl-rng-star-initial-10n-50k-10c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 50)" w lp ls 6


set title ""
unset key
plot "../results/handl-core-initial-50n-10k-10c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 10)" w lp ls 1,\
     "../results/handl-rng-star-initial-50n-10k-10c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 10)" w lp ls 2,\
     "../results/handl-core-initial-50n-30k-10c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 30)" w lp ls 3,\
     "../results/handl-rng-star-initial-50n-30k-10c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 30)" w lp ls 4,\
     "../results/handl-core-initial-50n-50k-10c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 50)" w lp ls 5,\
     "../results/handl-rng-star-initial-50n-50k-10c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 50)" w lp ls 6

# ----------------------------------------------------------------------------------------------------------------------------------

set ylabel "#edges"
set ytics

set title ""
unset key
plot "../results/handl-core-initial-5n-10k-30c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 10)" w lp ls 1,\
     "../results/handl-rng-star-initial-5n-10k-30c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 10)" w lp ls 2,\
     "../results/handl-core-initial-5n-30k-30c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 30)" w lp ls 3,\
     "../results/handl-rng-star-initial-5n-30k-30c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 30)" w lp ls 4,\
     "../results/handl-core-initial-5n-50k-30c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 50)" w lp ls 5,\
     "../results/handl-rng-star-initial-5n-50k-30c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 50)" w lp ls 6

unset ylabel
#unset ytics

set title ""
unset key
plot "../results/handl-core-initial-10n-10k-30c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 10)" w lp ls 1,\
     "../results/handl-rng-star-initial-10n-10k-30c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 10)" w lp ls 2,\
     "../results/handl-core-initial-10n-30k-30c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 30)" w lp ls 3,\
     "../results/handl-rng-star-initial-10n-30k-30c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 30)" w lp ls 4,\
     "../results/handl-core-initial-10n-50k-30c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 50)" w lp ls 5,\
     "../results/handl-rng-star-initial-10n-50k-30c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 50)" w lp ls 6


set title ""
unset key
plot "../results/handl-core-initial-50n-10k-30c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 10)" w lp ls 1,\
     "../results/handl-rng-star-initial-50n-10k-30c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 10)" w lp ls 2,\
     "../results/handl-core-initial-50n-30k-30c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 30)" w lp ls 3,\
     "../results/handl-rng-star-initial-50n-30k-30c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 30)" w lp ls 4,\
     "../results/handl-core-initial-50n-50k-30c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 50)" w lp ls 5,\
     "../results/handl-rng-star-initial-50n-50k-30c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 50)" w lp ls 6

# ----------------------------------------------------------------------------------------------------------------------------------

set xtics
set xlabel "#dimensions"

set ylabel "#edges"
set ytics

set title ""
unset key
plot "../results/handl-core-initial-5n-10k-50c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 10)" w lp ls 1,\
     "../results/handl-rng-star-initial-5n-10k-50c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 10)" w lp ls 2,\
     "../results/handl-core-initial-5n-30k-50c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 30)" w lp ls 3,\
     "../results/handl-rng-star-initial-5n-30k-50c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 30)" w lp ls 4,\
     "../results/handl-core-initial-5n-50k-50c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 50)" w lp ls 5,\
     "../results/handl-rng-star-initial-5n-50k-50c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 50)" w lp ls 6

unset ylabel
#unset ytics

set title ""
unset key
plot "../results/handl-core-initial-10n-10k-50c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 10)" w lp ls 1,\
     "../results/handl-rng-star-initial-10n-10k-50c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 10)" w lp ls 2,\
     "../results/handl-core-initial-10n-30k-50c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 30)" w lp ls 3,\
     "../results/handl-rng-star-initial-10n-30k-50c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 30)" w lp ls 4,\
     "../results/handl-core-initial-10n-50k-50c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 50)" w lp ls 5,\
     "../results/handl-rng-star-initial-10n-50k-50c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 50)" w lp ls 6


set title ""
#set key horizontal font ", 100" at screen 0.5, 0.1
unset key
plot "../results/handl-core-initial-50n-10k-50c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 10)" w lp ls 1,\
     "../results/handl-rng-star-initial-50n-10k-50c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 10)" w lp ls 2,\
     "../results/handl-core-initial-50n-30k-50c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 30)" w lp ls 3,\
     "../results/handl-rng-star-initial-50n-30k-50c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 30)" w lp ls 4,\
     "../results/handl-core-initial-50n-50k-50c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 50)" w lp ls 5,\
     "../results/handl-rng-star-initial-50n-50k-50c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 50)" w lp ls 6


#<null>
#here we need to unset everything that was set previously
unset origin
unset border
unset tics
unset label
unset arrow
unset title
unset object
unset xlabel

#Now set the size of this plot to something BIG
set size 120,120 #however big you need it

#example key settings
set key box width 2 height 1
set key vertical samplen 2 maxrows 2 maxcols 4
set key at screen 0.5, 0.06 center top

#We need to set an explicit xrange.  Anything will work really.
set xrange [-1:1]
set yrange [-1:1]

plot "../results/handl-core-initial-50n-10k-50c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 10)" w lp ls 1,\
     "../results/handl-rng-star-initial-50n-10k-50c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 10)" w lp ls 2,\
     "../results/handl-core-initial-50n-30k-50c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 30)" w lp ls 3,\
     "../results/handl-rng-star-initial-50n-30k-50c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 30)" w lp ls 4,\
     "../results/handl-core-initial-50n-50k-50c.results.avg" using (column("size")):xtic(1) t "CORE-SG (k_{max} = 50)" w lp ls 5,\
     "../results/handl-rng-star-initial-50n-50k-50c.results.avg" using (column("size")):xtic(1) t "RNG* (k_{max} = 50)" w lp ls 6
