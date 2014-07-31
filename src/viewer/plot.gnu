set terminal png size 1920,1000
set output "plot.png"
set xlabel "time"
set ylabel "energy"
set yrange [-32000: 32000]
f(x) = 0*x*x + 0*x + 0
plot 'wave.txt' using 1 title "Raw Data (smoothed)" with line smooth sbezier, \
f(x) with line title "Fit"
"Fit"
 title "Fit"

"
