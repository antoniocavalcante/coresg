load '../gnuplot/colors.gnuplot'

set datafile separator whitespace

# Output to PNG, with Verdana 8pt font
#set terminal pngcairo nocrop enhanced font "verdana,23" size 1000,600

set terminal postscript eps enhanced color font 'Helvetica,24'

# Don't show the legend in the chart
set key outside horizontal center top

set style line 1 lc rgb A pt 4 ps 2 lt 1 lw 4 # RNG**
set style line 2 lc rgb B pt 5 ps 2 lt 1 lw 4 dashtype 2 # RNG*
set style line 3 lc rgb C pt 6 ps 2 lt 1 lw 4 # RNG
set style line 4 lc rgb D pt 7 ps 2 lt 1 lw 4 dashtype 2 # MRG
set style line 5 lc rgb E pt 8 ps 2 lt 1 lw 4 # MRG
set style line 6 lc rgb F pt 9 ps 2 lt 1 lw 4 dashtype 2 # MRG

set style line 6 lc rgb '#000000' lt 1 lw 2

set ylabel "#edges"

# Replace small stripes on the Y-axis with a horizontal gridlines
set tic scale 0
set grid ytics lc rgb "#D9D9D9"

# Remove border around chart
unset border

set xtics center
set offsets 0.3, 0.3, 0.3, 0.3

set logscale y

set format y "10^{%L}"

#set yrange [0 :]

set output "../gnuplot/figs/graph-size-".exp.".eps"
#set output "../gnuplot/figs/graph-size-".exp.".png"
