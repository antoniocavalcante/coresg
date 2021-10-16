load '../gnuplot/colors.gnuplot'

set datafile separator whitespace

# Output to PNG, with Verdana 8pt font
#set terminal pngcairo nocrop enhanced font "verdana,23" size 1000,600

# Don't show the legend in the chart
set key outside horizontal center right


set style line 1 lc rgb A pt 4 ps 6 lt 1 lw 12 # RNG**
set style line 2 lc rgb B pt 5 ps 6 lt 1 lw 10 dashtype 2 # RNG*
set style line 3 lc rgb C pt 6 ps 6 lt 1 lw 12 # RNG
set style line 4 lc rgb D pt 7 ps 6 lt 1 lw 10 dashtype 2 # MRG
set style line 5 lc rgb E pt 8 ps 6 lt 1 lw 12 # MRG
set style line 6 lc rgb F pt 9 ps 6 lt 1 lw 10 dashtype 2 # MRG

set style line 7 lc rgb '#000000' lt 9 lw 2

set ylabel "#edges"

# Replace small stripes on the Y-axis with a horizontal gridlines
set tic scale 0
set grid ytics lc rgb "#D9D9D9"

# Remove border around chart
#unset border

set xtics center
set offsets 0.1, 0.1, 0.1, 0.1

#set output "../gnuplot/figs/graph-size-".exp.".eps"
#set output "../gnuplot/figs/graph-size-".exp.".png"
