set terminal png size 1920,1000
set output "plot.png"
set xlabel "time"
set ylabel "energy"
set yrange [-32000: 32000]
set xrange [0: 1000]
set x2range [0: 1000]
f(x) = 36544.5*exp(-1*((x-603.975)/184.978)**2) + -31465.3
plot 'wave.txt' using 1 title "Raw Data (smoothed)" with line smooth sbezier axes x1y1 lt rgb "black", \
f(x) with line title "Fit" axes x2y1 lt rgb "green"
