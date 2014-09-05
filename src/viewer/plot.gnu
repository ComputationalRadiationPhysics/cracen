set terminal png size 1920,1000
set output "plot.png"
set xlabel "time"
set ylabel "energy"
set yrange [-32000: 32000]
set xrange [0: 1000]
set x2range [0: 1000]
f(x) = 23532.7*exp(-1*((x-603.973)/180.973)**2) + -31742.4
plot 'wave.txt' using 1 title "Raw Data (smoothed)" with line smooth sbezier axes x1y1 lt rgb "black", \
f(x) with line title "Fit" axes x2y1 lt rgb "green"
