load '../gnuplot/colors.gnuplot'

set datafile separator whitespace

# Output to PNG, with Verdana 8pt font
#set terminal pngcairo nocrop enhanced font "verdana,23" size 1200,600

set terminal postscript eps enhanced color font 'Helvetica,26'

# Replace small stripes on the Y-axis with a horizontal gridlines
set tic scale 0
set grid ytics lc rgb "#D9D9D9"

# Place tics on the x axis.
set xtics center

# Remove border around chart
unset border

# Set key on the left top corner of the chart.
set key left top

set key samplen 2

# Thinner, filled bars
set boxwidth 0.85

#set style histogram clustered
#set style histogram rowstacked
set style data histograms
#set style histogram columnstacked

set style fill solid 1.00

set ylabel "Time (min)"

set yrange [0:7]

#set format y "%g%%"

#set output "../gnuplot/figs/".exp.".png"
#set output "../gnuplot/figs/runtime-".exp."-percentage.eps"
