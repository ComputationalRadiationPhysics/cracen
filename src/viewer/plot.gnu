set terminal png size 1920,1000
set output "plot.png"
set xlabel "time"
set ylabel "energy"
set yrange [-32000: 32000]
set xrange [0: 1000]
set x2range [0: 1000]
set arrow from 537,-32000 to 537,32000 nohead lc rgb 'black'
set arrow from 587,-32000 to 587,32000 nohead lc rgb 'blue'
set arrow from 637,-32000 to 637,32000 nohead lc rgb 'black'
f(x) = -377444*x**0 + 1299.31*x**1 + -1.10272*x**2
plot 'wave.txt' using 1 title "Raw Data (smoothed)" with line smooth sbezier axes x1y1 lt rgb "black", \
f(x) with line title "Fit" axes x2y1 lt rgb "green"
