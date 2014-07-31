#ifndef CUDASTOPWATCH_HPP
#define CUDASTOPWATCH_HPP

struct TickCounter {
	unsigned long long min, max, avg, iter;
	__device__ TickCounter() :
		min(0xFFFFFFFFFFFFFFFF),
		max(0),
		avg(0),
		iter(0)
	{}
};

__global__ void initTicks(TickCounter* tickCounter) {
	tickCounter[threadIdx.x] = TickCounter();
}
template <unsigned int numberOfIntervalls>
__global__ void swPrint(TickCounter* counter) {
	double sum = 0;
	for(int i = 0; i < numberOfIntervalls; i++) {
		counter[i].avg/=counter[i].iter;
		sum = sum + counter[i].avg;			
	}
	for(int i = 0; i < numberOfIntervalls; i++) {
		const double perc = (counter[i].avg/sum)*100;
		printf("Intervall %i {min:%llu,	max:%llu,	avg:%llu,	perc:%f%}\n",i,counter[i].min,counter[i].max,counter[i].avg, perc);
	}
}
template <unsigned int numberOfIntervalls>
class CudaStopWatch {
private:
	TickCounter *counter;
	unsigned long long currentTime;
	int intervall;
public:
	__device__ CudaStopWatch(TickCounter *mem) :
		counter(mem),
		intervall(-1)
	{
	}
	__device__ void start() {
		intervall = 0;
		currentTime = clock64();
	}
	__device__ void stop() {
		if(intervall >= 0 && intervall < numberOfIntervalls) {
			unsigned long long diff = clock64() - currentTime;
			__syncthreads();
			atomicAdd(&counter[intervall].avg, diff);
			atomicMin(&counter[intervall].min, diff);
			atomicMax(&counter[intervall].max, diff);
			atomicAdd(&counter[intervall].iter, 1);
			intervall++;
			__syncthreads();
			currentTime = clock64();
		}
	}
};


#endif
