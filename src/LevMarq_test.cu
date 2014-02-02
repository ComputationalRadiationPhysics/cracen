#ifndef LEVMARQ_H
#define LEVMARQ_H

#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
//#include "Types.h"
//#include "Constants.h"

#define CUDA //defined: runs on GPU, otherwise on CPU (useful for debugging)

#ifdef CUDA
	#if (__CUDA_ARCH__ < 200)
		#define SHAREDMEMORYSIZE 16384
	#else
		#define SHAREDMEMORYSIZE 49152
	#endif

	#define GLOBAL __global__
	#define DEVICE __device__
	#define SHARED __shared__
#else
	#define GLOBAL
	#define DEVICE
	#define SHARED
#endif

#ifdef CUDA
	texture<float, 2, cudaReadModeElementType> dataTexture0, dataTexture1, dataTexture2, dataTexture3, dataTexture4, dataTexture5; //TODO rename to sampleTex(ture)

	template<unsigned int texIndex> DEVICE inline float getSampleY(int chunkIndex, float x);
	template<> DEVICE inline float getSampleY<0>(int chunkIndex, float x) { return tex2D(dataTexture0, x + 0.5, chunkIndex + 0.5); }
	template<> DEVICE inline float getSampleY<1>(int chunkIndex, float x) { return tex2D(dataTexture1, x + 0.5, chunkIndex + 0.5); }
	template<> DEVICE inline float getSampleY<2>(int chunkIndex, float x) { return tex2D(dataTexture2, x + 0.5, chunkIndex + 0.5); }
	template<> DEVICE inline float getSampleY<3>(int chunkIndex, float x) { return tex2D(dataTexture3, x + 0.5, chunkIndex + 0.5); }
	template<> DEVICE inline float getSampleY<4>(int chunkIndex, float x) { return tex2D(dataTexture4, x + 0.5, chunkIndex + 0.5); }
	template<> DEVICE inline float getSampleY<5>(int chunkIndex, float x) { return tex2D(dataTexture5, x + 0.5, chunkIndex + 0.5); }
#else
	float *data;

	template<unsigned int texIndex> DEVICE inline float getSampleY(int chunkIndex, float x);
	template<> DEVICE inline float getSampleY<0>(int chunkIndex, float x) { return data[(int)x]; } //texIndex must be 0 (other indices only for CUDA), chunkIndex has no effect (only for CUDA)
#endif

const int paramCount = 3; //max. 3 (see TODO in determinant-function)

#ifdef SHAREDMEMORYSIZE
	const int maxSampleCount = (SHAREDMEMORYSIZE / sizeof(float) - paramCount * (paramCount + 4)) / ((paramCount + 1));
#else
	const int maxSampleCount = 5000;
#endif

struct fitData {
	int status;

	//(no result if status is statusWrongInputData)
	float startY;
	float endY;

	//used samples for approximation (no result if status is statusWrongInputData)
	int firstSampleForFitX;
	int samplesForFitCount;

	//result of approximation (no result if status is statusWrongInputData or statusNotEnoughSamplesForFit)
	float param[paramCount];
	float extremumX;
	float extremumY;
	float sumSqrResidues; //sqrt(sumSqrResidues) -> euclidean norm
	float sumAbsResidues; //sumAbsResidues / samplesForFitCount -> average residue
	int iterationCount;
};
enum fitDataStatus { //<= 0: error; > 0: calculated
	statusWrongInputData = -1,        //error (no samples or wrong stepX)
	statusNotEnoughSamplesForFit = 0, //error (less samples for fit than fit-function parameter, note: maybe change fitThresholdY)
	statusSuccessful = 1,             //success (below maxParamChange)
	statusMaxIterationsReached = 2    //no error, but not good enough (max. iterations have been reached)
};

DEVICE const float fitThresholdY = 0.5;
DEVICE const float startEndProportion = 0.01;

//terminates if at least one of the following two conditions has been reached
DEVICE const float maxParamChange = 0.01;
const int maxIterations = 50;

DEVICE const float lowerBound = -0.1;       //maybe TODO make it dynamic by intelligent calculation
DEVICE const float upperBound = +0.1;       //maybe TODO make it dynamic by intelligent calculation
DEVICE const float dampingChangeFactor = 4; //maybe TODO make it dynamic by intelligent calculation

DEVICE inline float fitModel(int form, float* param, float x, float y)
{
	switch(form) {
		case 0:
			return x * x;
		case 1:
			return x;
		case 2:
			return 1;
		default:
			return param[0] * x * x + param[1] * x + param[2] - y;
	}
}

/*!
 * \brief fitFunction returns the y of a given x
 * \param x given x value to calculate y
 * \param param parameters to define the concrete current fit-function
 * \param y the returned y value
*/
DEVICE inline float fitFunction(float* param, float x)
{
	return param[0] * x * x + param[1] * x + param[2];
}

/*!
 * \brief fitFunctionExtremum returns the x of the min. or max. y value
 * \param param parameters to define the concrete current fit-function
 * \param x the returned x value
*/
DEVICE inline float fitFunctionExtremumX(float* param)
{
	return -param[1] / (2 * param[0]);
}

/*!
 * \brief paramStartValue returns the parameter start values for the fit-function calculation
 * \param firstValue first value of the data used for fit-function
 * \param lastValue last value of the data used for fit-function
 * \param chunkIndex index of the current dataset (GPU mode) or not used (CPU mode)
 * \param param the returned parameter start values
*/
template<unsigned int texIndex>
DEVICE void paramStartValue(int chunkIndex, float* param, int firstSampleForFitX, int lastSampleForFitX)
{
	long long int x1, y1, x2, y2, x3, y3, dv;

	x1 = firstSampleForFitX;
	x2 = (lastSampleForFitX - firstSampleForFitX) / 2 + firstSampleForFitX;
	x3 = lastSampleForFitX;
	y1 = getSampleY<texIndex>(chunkIndex, x1);
	y2 = getSampleY<texIndex>(chunkIndex, x2);
	y3 = getSampleY<texIndex>(chunkIndex, x3);

	//any value, but not { 0, 0, 0 }
	dv = (x2-x1)*(x3-x1)*(x2-x3);
	if (dv == 0) {
		param[0] = 1;
		param[1] = 1;
		param[2] = 1;
	}
	else {
		param[0] = (-x1*y2+x1*y3+x2*y1-x2*y3-x3*y1+x3*y2)/dv;
		param[1] = (x1*x1*y2-x1*x1*y3-x2*x2*y1+x2*x2*y3+x3*x3*y1-x3*x3*y2)/dv;
		param[2] = (x1*x1*x2*y3+x1*x1*-x3*y2-x1*x2*x2*y3+x1*x3*x3*y2+x2*x2*x3*y1-x2*x3*x3*y1)/dv;
	}
}

/*!
 * \brief maxValue returns the x and y where y has the greatest value
 * \param sampleCount number of samples
 * \param chunkIndex index of the current dataset (GPU mode) or not used (CPU mode)
 * \param x the returned x value
 * \param y the returned y value
*/
template<unsigned int texIndex>
DEVICE int maxYSampleX(int chunkIndex, int sampleCount)
{
	int i, x;
	float y;

	x = 0;
	y = getSampleY<texIndex>(chunkIndex, 0);
	for (i = 0; i < sampleCount; i++)
		if (getSampleY<texIndex>(chunkIndex, i) > y) {
			y = getSampleY<texIndex>(chunkIndex, i);
			x = i;
		}
	return x;
}

/*!
 * \brief averageValue returns the average of all y values in a given range
 * \param start first x for average calculation
 * \param count number of values for average calculation
 * \param chunkIndex index of the current dataset (GPU mode) or not used (CPU mode)
 * \param y the returned average
*/
template<unsigned int texIndex>
DEVICE float averageSamplesY(int chunkIndex, int firstSample, int sampleCount)
{
	int i;
	float sum = 0;

	for (i = firstSample; i < firstSample + sampleCount; i++)
		sum += getSampleY<texIndex>(chunkIndex, i);
	return sum / sampleCount;
}

/*!
 * \brief xOfValue returns the first x of a value y that is greater or equal of a given min. value
 * \param countData number of samples
 * \param chunkIndex index of the current dataset (GPU mode) or not used (CPU mode)
 * \param fromDirection 
 * \param minValue min. y value
 * \param x the returned x value, -1 if there is no x with a y greater or equal minValue
*/
template<unsigned int texIndex>
DEVICE int firstGreaterSampleX(int chunkIndex, int sampleCount, bool fromLeft, float minY)
{
	int i;

	if (fromLeft) {
		for (i = 0; i < sampleCount; i++)
			if (getSampleY<texIndex>(chunkIndex, i) >= minY)
				return i;
	}
	else
		for (i = sampleCount - 1; i >= 0; i--)
			if (getSampleY<texIndex>(chunkIndex, i) >= minY)
				return i;
	return -1;
}

DEVICE inline float determinant(float* matrix, int size)
{
	int size2 = 2 * size;

	if (size == 2)
		return matrix[0] * matrix[1 + size]
		     - matrix[1] * matrix[size];
	else if (size == 3)
		return matrix[0] * matrix[1 + size] * matrix[2 + size2]
		     + matrix[1] * matrix[2 + size] * matrix[size2]
		     + matrix[2] * matrix[size] * matrix[1 + size2]
		     - matrix[0] * matrix[2 + size] * matrix[1 + size2]
		     - matrix[1] * matrix[size] * matrix[2 + size2]
		     - matrix[2] * matrix[1 + size] * matrix[size2];
	return 0; //bigger size not implemented (TODO Laplace expansion of a determinant)
}

template<unsigned int texIndex>
GLOBAL void kernel(int sampleCount, float stepX, fitData *result)
{
#ifdef CUDA
	int chunkIndex = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int threadIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
	int threadCount = blockDim.x * blockDim.y * blockDim.z;
#else
	int chunkIndex = 0;
	int threadIndex = 0;
	int threadCount = 1;
#endif
	int i, j, k, iteration;

	SHARED float matA[maxSampleCount * paramCount];
	SHARED float vecB[maxSampleCount];
	SHARED float matATxA[paramCount * paramCount];
	SHARED float vecATxB[paramCount];
	SHARED float vecATxAsavedColumn[paramCount];
	SHARED float param[paramCount];
	SHARED float s[paramCount];
	float detA, detAiB;
	float bValue, cValue;
	float sum, sumSqrB, sumSqrBnew, sumSqrC, sumAbsBnew;
	float dampingParam, pm;
	bool discard;
	bool paramChanged;

	int firstSampleForFitX, lastSampleForFitX, samplesForFitCount, averageCount;
	float minY, maxY, x;
	int status;

	/*maxY = getSampleY<texIndex>(chunkIndex, maxYSampleX<texIndex>(chunkIndex, sampleCount));
	minY = getSampleY<texIndex>(chunkIndex, 0);
	firstSampleForFitX = firstGreaterSampleX<texIndex>(chunkIndex, sampleCount, true, (maxY - minY) * fitThresholdY + minY);
	minY = getSampleY<texIndex>(chunkIndex, sampleCount - 1);
	lastSampleForFitX = firstGreaterSampleX<texIndex>(chunkIndex, sampleCount, false, (maxY - minY) * fitThresholdY + minY);*/

	maxY = getSampleY<texIndex>(chunkIndex, maxYSampleX<texIndex>(chunkIndex, sampleCount));
	firstSampleForFitX = firstGreaterSampleX<texIndex>(chunkIndex, sampleCount, true, (maxY - getSampleY<texIndex>(chunkIndex, 0)) * fitThresholdY + getSampleY<texIndex>(chunkIndex, 0));
	lastSampleForFitX = firstGreaterSampleX<texIndex>(chunkIndex, sampleCount, false, (maxY - getSampleY<texIndex>(chunkIndex, sampleCount - 1)) * fitThresholdY + getSampleY<texIndex>(chunkIndex, sampleCount - 1));

	if (firstSampleForFitX >= 0 && lastSampleForFitX >= 0 && stepX > 0) {
		samplesForFitCount = (int)((lastSampleForFitX - firstSampleForFitX) / stepX) + 1;

#ifdef CUDA
		if (threadCount < samplesForFitCount) { //TODO one thread has to do more than one task
			result[chunkIndex].status = -10;
			return;
		}
#endif

		if (samplesForFitCount >= paramCount) {
			paramStartValue<texIndex>(chunkIndex, param, firstSampleForFitX, lastSampleForFitX);
			dampingParam = 1;

			iteration = 0;
			discard = false;

			do {
				iteration++;

#ifdef CUDA
				//calculate A and B
				if (threadIndex < samplesForFitCount) {
					x = threadIndex * stepX + firstSampleForFitX;
					if (!discard)
						for (j = 0; j < paramCount; j++)
							matA[threadIndex + j * samplesForFitCount] = fitModel(j, param, x, getSampleY<texIndex>(chunkIndex, x));
					vecB[threadIndex] = fitModel(-1, param, x, getSampleY<texIndex>(chunkIndex, x));
				}
				__syncthreads();

				//calculate the square sum of B
				if (!discard) {
					sumSqrB = 0;
					for (i = 0; i < samplesForFitCount; i++) {
						sumSqrB += vecB[i] * vecB[i];
					}
				}
#else
				if (!discard) {
					//calculate A, B and the square sum of B immediately (used later)
					sumSqrB = 0;
					for (i = 0; i < samplesForFitCount; i++) {
						x = i * stepX + firstSampleForFitX;
						for (j = 0; j < paramCount; j++)
							matA[i + j * samplesForFitCount] = fitModel(j, param, x, getSampleY<texIndex>(chunkIndex, x));
						vecB[i] = fitModel(-1, param, x, getSampleY<texIndex>(chunkIndex, x));

						sumSqrB += vecB[i] * vecB[i];
					}
				}
#endif
#ifdef CUDA
				//ATrans * A
				if (threadIndex < paramCount * paramCount) {
					i = threadIndex / paramCount;
					j = threadIndex - i * paramCount;

					sum = 0;
					for (k = 0; k < samplesForFitCount + paramCount; k++) {
						if (k < samplesForFitCount)
							sum += matA[k + i * samplesForFitCount] * matA[k + j * samplesForFitCount]; //first factor matA is transposed
						else
							sum += (i == k - samplesForFitCount ? dampingParam : 0) * (j == k - samplesForFitCount ? dampingParam : 0);
					}
					matATxA[threadIndex] = sum;
				}
				__syncthreads();
#else
				//ATrans * A
				for (i = 0; i < paramCount; i++) {
					for (j = 0; j < paramCount; j++) {
						sum = 0;
						for (k = 0; k < samplesForFitCount + paramCount; k++) {
							if (k < samplesForFitCount)
								sum += matA[k + i * samplesForFitCount] * matA[k + j * samplesForFitCount]; //first factor matA is transposed
							else
								sum += (i == k - samplesForFitCount ? dampingParam : 0) * (j == k - samplesForFitCount ? dampingParam : 0);
						}
						matATxA[i + j * paramCount] = sum;
					}
				}
#endif
#ifdef CUDA
				//ATrans * -B
				if (threadIndex < paramCount) {
					sum = 0;
					for (j = 0; j < samplesForFitCount; j++) {
						sum += matA[j + threadIndex * samplesForFitCount] * vecB[j]; //first factor matA is transposed
					}
					vecATxB[threadIndex] = -sum; //-sum instead of sum, because: ||A * x - B||2 instead of ||A * x + B||2
				}
				__syncthreads();
#else
				//ATrans * -B
				for (i = 0; i < paramCount; i++) {
					sum = 0;
					for (j = 0; j < samplesForFitCount; j++) {
						sum += matA[j + i * samplesForFitCount] * vecB[j]; //first factor matA is transposed
					}
					vecATxB[i] = -sum; //-sum instead of sum, because: ||A * x - B||2 instead of ||A * x + B||2
				}
#endif
#ifdef CUDA
				//||A * x - B||2 = min => matATxA * x = vecATxB -> x = ? -> Cramer's rule
				detA = determinant(matATxA, paramCount);
				__syncthreads();

				if (threadIndex == 0) {
					for (i = 0; i < paramCount; i++) {
						for (j = 0; j < paramCount; j++) {
							vecATxAsavedColumn[j] = matATxA[i + j * paramCount];
							matATxA[i + j * paramCount] = vecATxB[j];
						}

						detAiB = determinant(matATxA, paramCount);

						for (j = 0; j < paramCount; j++)
							matATxA[i + j * paramCount] = vecATxAsavedColumn[j];

						s[i] = detAiB / detA;

						param[i] += s[i];
					}
				}
				__syncthreads();
#else
				//||A * x - B||2 = min => matATxA * x = vecATxB -> x = ? -> Cramer's rule
				detA = determinant(matATxA, paramCount);
				for (i = 0; i < paramCount; i++) {
					for (j = 0; j < paramCount; j++) {
						vecATxAsavedColumn[j] = matATxA[i + j * paramCount];
						matATxA[i + j * paramCount] = vecATxB[j];
					}

					detAiB = determinant(matATxA, paramCount);

					for (j = 0; j < paramCount; j++)
						matATxA[i + j * paramCount] = vecATxAsavedColumn[j];

					s[i] = detAiB / detA;

					param[i] += s[i];
				}
#endif

				paramChanged = false;
				for (i = 0; i < paramCount; i++) {
					if (fabs(s[i]) > maxParamChange)
						paramChanged = true;
				}

#ifdef CUDA
				//calculate the square sum of B + A * s
				if (threadIndex < samplesForFitCount) {
					for (j = 0; j < paramCount; j++)
						vecB[threadIndex] += matA[threadIndex + j * samplesForFitCount] * s[j];
					vecB[threadIndex] = vecB[threadIndex] * vecB[threadIndex];
				}
				__syncthreads();
				sumSqrC = 0;
				for (i = 0; i < samplesForFitCount; i++) {
					sumSqrC += vecB[i];
				}

				//calculate the square sum and the absolute sum of new B
				if (threadIndex < samplesForFitCount) {
					x = threadIndex * stepX + firstSampleForFitX;
					vecB[threadIndex] = fabs(fitModel(-1, param, x, getSampleY<texIndex>(chunkIndex, x)));
				}
				__syncthreads();
				sumSqrBnew = 0;
				sumAbsBnew = 0;
				for (i = 0; i < samplesForFitCount; i++) {
					sumSqrBnew += vecB[i] * vecB[i];
					sumAbsBnew += vecB[i];
				}
#else
				sumSqrC = 0;
				sumSqrBnew = 0;
				sumAbsBnew = 0;
				for (i = 0; i < samplesForFitCount; i++) {
					//calculate the square sum of B + A * s
					cValue = vecB[i];
					for (j = 0; j < paramCount; j++)
						cValue += matA[i + j * samplesForFitCount] * s[j];
					sumSqrC += cValue * cValue;

					//calculate the square sum and the absolute sum of new B
					x = i * stepX + firstSampleForFitX;
					bValue = fitModel(-1, param, x, getSampleY<texIndex>(chunkIndex, x));
					sumSqrBnew += bValue * bValue;
					sumAbsBnew += fabs(bValue);
				}
#endif

				pm = (sumSqrB - sumSqrBnew) / (sumSqrB - sumSqrC);

				if (pm <= lowerBound) {
#ifdef CUDA
					if (threadIndex < paramCount)
						param[threadIndex] -= s[threadIndex];
					__syncthreads();
#else
					for (i = 0; i < paramCount; i++)
						param[i] -= s[i];
#endif
					dampingParam *= dampingChangeFactor;
					discard = true;
				}
				else if (lowerBound < pm && pm < upperBound) {
					discard = false;
				}
				else {
					dampingParam /= dampingChangeFactor;
					discard = false;
				}
			} while (iteration < maxIterations && (paramChanged || discard));

			if (!paramChanged)
				status = statusSuccessful;
			else
				status = statusMaxIterationsReached;
		}
		else
			status = statusNotEnoughSamplesForFit;
	}
	else
		status = statusWrongInputData;

	//set result values
	if (threadIndex == 0) {
		result[chunkIndex].status = status;

		if (status != statusWrongInputData) {
			averageCount = sampleCount * startEndProportion;
			if (sampleCount > 0 && averageCount == 0)
				averageCount = 1;
			result[chunkIndex].startY = averageSamplesY<texIndex>(chunkIndex, 0, averageCount);
			result[chunkIndex].endY = averageSamplesY<texIndex>(chunkIndex, sampleCount - averageCount, averageCount);

			result[chunkIndex].firstSampleForFitX = firstSampleForFitX;
			result[chunkIndex].samplesForFitCount = samplesForFitCount;

			if (status != statusNotEnoughSamplesForFit) {
				for (i = 0; i < paramCount; i++)
					result[chunkIndex].param[i] = param[i];
				result[chunkIndex].extremumX = fitFunctionExtremumX(param);
				result[chunkIndex].extremumY = fitFunction(param, result[chunkIndex].extremumX);
				result[chunkIndex].sumSqrResidues = sumSqrBnew;
				result[chunkIndex].sumAbsResidues = sumAbsBnew;
				result[chunkIndex].iterationCount = iteration;
			}
		}
	}
}



//example data, only for testing
int main()
{
#ifdef CUDA
	int sampleCount = 1000;
	int chunkCount = 1;
	//float testData[2][3] = { { 2, 5, 2 }, { 1, 4, 1 } };
	float testData[1][1000] = { { -31576, -31560, -31552, -31544, -31560, -31576, -31552, -31560, -31552, -31576, -31560, -31576, -31568, -31576, -31552, -31560, -31568, -31552, -31552, -31576, -31552, -31576, -31568, -31576, -31552, -31552, -31536, -31536, -31536, -31552, -31568, -31560, -31544, -31568, -31552, -31560, -31552, -31528, -31568, -31560, -31544, -31568, -31560, -31576, -31576, -31560, -31552, -31552, -31568, -31568, -31584, -31560, -31568, -31560, -31576, -31552, -31576, -31560, -31592, -31568, -31568, -31568, -31552, -31560, -31544, -31552, -31576, -31560, -31560, -31552, -31568, -31536, -31560, -31560, -31560, -31544, -31544, -31576, -31568, -31544, -31528, -31552, -31568, -31552, -31560, -31552, -31560, -31552, -31560, -31552, -31568, -31544, -31552, -31544, -31544, -31568, -31544, -31560, -31536, -31560, -31568, -31568, -31552, -31560, -31544, -31560, -31552, -31568, -31560, -31576, -31568, -31576, -31544, -31568, -31560, -31528, -31560, -31544, -31544, -31560, -31552, -31544, -31520, -31560, -31552, -31552, -31568, -31544, -31544, -31552, -31544, -31552, -31544, -31568, -31560, -31560, -31568, -31544, -31552, -31552, -31560, -31536, -31560, -31552, -31568, -31552, -31552, -31568, -31568, -31552, -31552, -31544, -31568, -31584, -31560, -31552, -31536, -31544, -31536, -31552, -31552, -31552, -31552, -31512, -31560, -31536, -31544, -31544, -31536, -31552, -31560, -31552, -31552, -31536, -31544, -31560, -31568, -31560, -31552, -31552, -31568, -31560, -31568, -31592, -31576, -31544, -31568, -31560, -31568, -31552, -31544, -31576, -31568, -31544, -31552, -31560, -31576, -31552, -31568, -31560, -31560, -31544, -31552, -31568, -31552, -31552, -31536, -31544, -31560, -31552, -31544, -31544, -31544, -31528, -31544, -31560, -31576, -31568, -31576, -31552, -31544, -31544, -31560, -31552, -31536, -31576, -31552, -31560, -31560, -31560, -31560, -31560, -31568, -31568, -31544, -31584, -31560, -31568, -31544, -31552, -31568, -31536, -31536, -31520, -31544, -31536, -31552, -31552, -31552, -31544, -31536, -31512, -31504, -31536, -31528, -31528, -31528, -31512, -31504, -31520, -31496, -31520, -31488, -31504, -31512, -31496, -31488, -31496, -31496, -31496, -31504, -31472, -31480, -31456, -31456, -31472, -31472, -31472, -31456, -31456, -31424, -31440, -31424, -31432, -31376, -31408, -31384, -31384, -31368, -31368, -31336, -31336, -31320, -31304, -31312, -31280, -31248, -31256, -31264, -31216, -31248, -31200, -31192, -31128, -31136, -31144, -31120, -31096, -31088, -31080, -31048, -31016, -30992, -30968, -30976, -30952, -30920, -30904, -30872, -30856, -30840, -30800, -30768, -30728, -30704, -30680, -30648, -30648, -30600, -30560, -30528, -30496, -30456, -30416, -30368, -30336, -30304, -30264, -30232, -30192, -30128, -30104, -30048, -30000, -29960, -29904, -29872, -29816, -29760, -29720, -29656, -29632, -29576, -29488, -29440, -29400, -29344, -29288, -29232, -29176, -29112, -29048, -28984, -28896, -28872, -28824, -28728, -28680, -28568, -28504, -28464, -28368, -28312, -28248, -28176, -28088, -28016, -27936, -27848, -27792, -27696, -27624, -27528, -27456, -27368, -27272, -27192, -27112, -26992, -26920, -26816, -26720, -26648, -26536, -26448, -26344, -26280, -26168, -26080, -25992, -25872, -25776, -25680, -25576, -25480, -25368, -25240, -25168, -25048, -24944, -24824, -24720, -24600, -24520, -24408, -24280, -24176, -24096, -23936, -23832, -23704, -23616, -23488, -23408, -23256, -23144, -23000, -22912, -22776, -22632, -22544, -22400, -22272, -22160, -22056, -21904, -21776, -21664, -21512, -21408, -21288, -21144, -21056, -20888, -20784, -20656, -20536, -20416, -20280, -20144, -20032, -19896, -19776, -19640, -19512, -19392, -19240, -19144, -19008, -18864, -18720, -18624, -18488, -18344, -18248, -18088, -17976, -17840, -17704, -17576, -17464, -17328, -17200, -17064, -16944, -16832, -16688, -16552, -16448, -16312, -16176, -16072, -15944, -15832, -15704, -15576, -15448, -15352, -15200, -15080, -14976, -14816, -14736, -14616, -14512, -14384, -14280, -14176, -14056, -13936, -13808, -13696, -13608, -13480, -13376, -13256, -13160, -13056, -12936, -12816, -12744, -12632, -12528, -12416, -12336, -12208, -12136, -12048, -11928, -11824, -11752, -11640, -11536, -11456, -11384, -11288, -11176, -11104, -11016, -10952, -10864, -10776, -10688, -10616, -10544, -10480, -10392, -10304, -10264, -10176, -10080, -10032, -9944, -9896, -9840, -9776, -9704, -9656, -9592, -9528, -9480, -9424, -9360, -9288, -9272, -9192, -9152, -9096, -9072, -9016, -8984, -8952, -8912, -8880, -8816, -8792, -8760, -8688, -8672, -8640, -8632, -8584, -8576, -8560, -8512, -8512, -8504, -8456, -8440, -8424, -8408, -8408, -8416, -8392, -8392, -8360, -8392, -8368, -8392, -8376, -8368, -8384, -8376, -8392, -8392, -8392, -8392, -8408, -8440, -8440, -8448, -8480, -8504, -8488, -8528, -8560, -8560, -8592, -8640, -8664, -8672, -8696, -8720, -8744, -8816, -8824, -8864, -8888, -8928, -8976, -9008, -9064, -9104, -9128, -9208, -9232, -9296, -9336, -9392, -9464, -9496, -9560, -9616, -9680, -9736, -9776, -9856, -9896, -9984, -10024, -10080, -10152, -10224, -10280, -10352, -10440, -10512, -10552, -10648, -10688, -10784, -10856, -10920, -10984, -11072, -11136, -11224, -11304, -11384, -11440, -11536, -11624, -11696, -11768, -11840, -11920, -12032, -12128, -12208, -12288, -12408, -12472, -12568, -12640, -12752, -12824, -12912, -13000, -13088, -13184, -13288, -13384, -13488, -13576, -13680, -13776, -13856, -13968, -14040, -14144, -14264, -14344, -14432, -14544, -14640, -14752, -14848, -14928, -15016, -15136, -15232, -15304, -15448, -15544, -15664, -15744, -15824, -15944, -16032, -16160, -16256, -16352, -16448, -16552, -16648, -16768, -16856, -16976, -17056, -17184, -17280, -17392, -17496, -17608, -17704, -17784, -17880, -17976, -18096, -18216, -18304, -18416, -18512, -18640, -18712, -18816, -18912, -19016, -19136, -19232, -19344, -19432, -19544, -19640, -19736, -19824, -19928, -20000, -20136, -20240, -20328, -20416, -20520, -20592, -20680, -20832, -20896, -20992, -21096, -21192, -21296, -21392, -21496, -21600, -21656, -21760, -21872, -21952, -22048, -22152, -22232, -22328, -22400, -22496, -22584, -22680, -22768, -22872, -22936, -23024, -23128, -23192, -23312, -23384, -23472, -23560, -23632, -23712, -23792, -23872, -23976, -24040, -24112, -24176, -24320, -24368, -24464, -24552, -24608, -24664, -24784, -24840, -24888, -24992, -25072, -25152, -25240, -25288, -25352, -25424, -25496, -25584, -25656, -25752, -25816, -25888, -25944, -25984, -26048, -26152, -26224, -26280, -26336, -26400, -26488, -26544, -26584, -26656, -26720, -26760, -26864, -26904, -26952, -27032, -27096, -27128, -27208, -27240, -27304, -27368, -27424, -27480, -27520, -27600, -27640, -27720, -27760, -27808, -27848, -27912, -27944, -27992, -28080, -28096, -28168, -28208, -28232, -28320, -28360, -28392, -28416, -28496, -28544, -28576, -28632, -28672, -28720, -28736, -28808, -28840, -28880, -28912, -28960, -28960, -29040, -29048, -29120, -29136, -29192, -29200, -29240, -29272, -29296, -29312, -29368, -29416, -29448, -29480, -29520, -29552, -29592, -29608, -29648, -29664, -29720, -29712, -29752, -29808, -29816, -29856, -29896, -29912, -29920, -29968, -30008, -30032, -30072, -30080, -30104, -30128, -30152, -30176, -30224, -30224, -30240, -30264, -30264, -30320, -30320, -30360, -30360, -30392, -30424, -30408, -30464, -30456, -30504, -30496, -30520, -30544, -30544, -30576, -30560, -30600, -30640, -30632, -30656, -30664, -30688, -30680, -30720, -30736, -30768, -30800, -30776, -30776, -30800, -30816, -30832, -30848, -30872, -30848, -30872, -30912, -30928, -30936, -30912, -30968, -30952, -30944, -30952, -30968, -30992, -31008, -30984, -31016, -31008, -31032, -31040, -31048, -31080, -31080, -31056, -31088, -31088, -31104, -31120, -31112, -31120, -31152, -31128, -31168, -31144, -31160, -31168, -31168, -31176, -31176, -31184, -31208, -31208, -31232, -31232, -31232, -31240, -31216, -31224, -31248, -31264, -31280 } };
	fitData result[1];

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	fitData* d_result;
	cudaMalloc((void**)&d_result, sizeof(fitData) * chunkCount);
	cudaMemcpy(d_result, result, sizeof(fitData) * chunkCount, cudaMemcpyHostToDevice);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray *dataArray;
	cudaMallocArray(&dataArray, &channelDesc, sampleCount, chunkCount);
	cudaMemcpyToArray(dataArray, 0, 0, testData, sizeof(float) * sampleCount * chunkCount, cudaMemcpyHostToDevice);

	dataTexture0.normalized = 0;
	dataTexture0.filterMode = cudaFilterModeLinear;
	dataTexture0.addressMode[0] = cudaAddressModeClamp;
	dataTexture0.addressMode[1] = cudaAddressModeClamp;

	cudaBindTextureToArray(dataTexture0, dataArray);

	dim3 grid(chunkCount, 1, 1); //number of blocks (in all dimensions) = number of chunks
	dim3 block(800, 1, 1); //number of threads (in all dimensions) must be >= samples for fit

	cudaEventRecord(start, 0);
	kernel<0><<<grid, block>>>(sampleCount, 1, d_result);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cudaMemcpy(result, d_result, sizeof(fitData) * chunkCount, cudaMemcpyDeviceToHost);

	cudaUnbindTexture(dataTexture0);

	cudaFreeArray(dataArray);
	cudaFree(d_result);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("calculation time: %f ms\n\n\n", time);
#else
	int sampleCount = 1000;
	int chunkCount = 1;
	//float testData[3] = { 2, 5, 2 }; //{ 2, 4.3, 5, 4.3, 2 };
	float testData[1000] = { -31576, -31560, -31552, -31544, -31560, -31576, -31552, -31560, -31552, -31576, -31560, -31576, -31568, -31576, -31552, -31560, -31568, -31552, -31552, -31576, -31552, -31576, -31568, -31576, -31552, -31552, -31536, -31536, -31536, -31552, -31568, -31560, -31544, -31568, -31552, -31560, -31552, -31528, -31568, -31560, -31544, -31568, -31560, -31576, -31576, -31560, -31552, -31552, -31568, -31568, -31584, -31560, -31568, -31560, -31576, -31552, -31576, -31560, -31592, -31568, -31568, -31568, -31552, -31560, -31544, -31552, -31576, -31560, -31560, -31552, -31568, -31536, -31560, -31560, -31560, -31544, -31544, -31576, -31568, -31544, -31528, -31552, -31568, -31552, -31560, -31552, -31560, -31552, -31560, -31552, -31568, -31544, -31552, -31544, -31544, -31568, -31544, -31560, -31536, -31560, -31568, -31568, -31552, -31560, -31544, -31560, -31552, -31568, -31560, -31576, -31568, -31576, -31544, -31568, -31560, -31528, -31560, -31544, -31544, -31560, -31552, -31544, -31520, -31560, -31552, -31552, -31568, -31544, -31544, -31552, -31544, -31552, -31544, -31568, -31560, -31560, -31568, -31544, -31552, -31552, -31560, -31536, -31560, -31552, -31568, -31552, -31552, -31568, -31568, -31552, -31552, -31544, -31568, -31584, -31560, -31552, -31536, -31544, -31536, -31552, -31552, -31552, -31552, -31512, -31560, -31536, -31544, -31544, -31536, -31552, -31560, -31552, -31552, -31536, -31544, -31560, -31568, -31560, -31552, -31552, -31568, -31560, -31568, -31592, -31576, -31544, -31568, -31560, -31568, -31552, -31544, -31576, -31568, -31544, -31552, -31560, -31576, -31552, -31568, -31560, -31560, -31544, -31552, -31568, -31552, -31552, -31536, -31544, -31560, -31552, -31544, -31544, -31544, -31528, -31544, -31560, -31576, -31568, -31576, -31552, -31544, -31544, -31560, -31552, -31536, -31576, -31552, -31560, -31560, -31560, -31560, -31560, -31568, -31568, -31544, -31584, -31560, -31568, -31544, -31552, -31568, -31536, -31536, -31520, -31544, -31536, -31552, -31552, -31552, -31544, -31536, -31512, -31504, -31536, -31528, -31528, -31528, -31512, -31504, -31520, -31496, -31520, -31488, -31504, -31512, -31496, -31488, -31496, -31496, -31496, -31504, -31472, -31480, -31456, -31456, -31472, -31472, -31472, -31456, -31456, -31424, -31440, -31424, -31432, -31376, -31408, -31384, -31384, -31368, -31368, -31336, -31336, -31320, -31304, -31312, -31280, -31248, -31256, -31264, -31216, -31248, -31200, -31192, -31128, -31136, -31144, -31120, -31096, -31088, -31080, -31048, -31016, -30992, -30968, -30976, -30952, -30920, -30904, -30872, -30856, -30840, -30800, -30768, -30728, -30704, -30680, -30648, -30648, -30600, -30560, -30528, -30496, -30456, -30416, -30368, -30336, -30304, -30264, -30232, -30192, -30128, -30104, -30048, -30000, -29960, -29904, -29872, -29816, -29760, -29720, -29656, -29632, -29576, -29488, -29440, -29400, -29344, -29288, -29232, -29176, -29112, -29048, -28984, -28896, -28872, -28824, -28728, -28680, -28568, -28504, -28464, -28368, -28312, -28248, -28176, -28088, -28016, -27936, -27848, -27792, -27696, -27624, -27528, -27456, -27368, -27272, -27192, -27112, -26992, -26920, -26816, -26720, -26648, -26536, -26448, -26344, -26280, -26168, -26080, -25992, -25872, -25776, -25680, -25576, -25480, -25368, -25240, -25168, -25048, -24944, -24824, -24720, -24600, -24520, -24408, -24280, -24176, -24096, -23936, -23832, -23704, -23616, -23488, -23408, -23256, -23144, -23000, -22912, -22776, -22632, -22544, -22400, -22272, -22160, -22056, -21904, -21776, -21664, -21512, -21408, -21288, -21144, -21056, -20888, -20784, -20656, -20536, -20416, -20280, -20144, -20032, -19896, -19776, -19640, -19512, -19392, -19240, -19144, -19008, -18864, -18720, -18624, -18488, -18344, -18248, -18088, -17976, -17840, -17704, -17576, -17464, -17328, -17200, -17064, -16944, -16832, -16688, -16552, -16448, -16312, -16176, -16072, -15944, -15832, -15704, -15576, -15448, -15352, -15200, -15080, -14976, -14816, -14736, -14616, -14512, -14384, -14280, -14176, -14056, -13936, -13808, -13696, -13608, -13480, -13376, -13256, -13160, -13056, -12936, -12816, -12744, -12632, -12528, -12416, -12336, -12208, -12136, -12048, -11928, -11824, -11752, -11640, -11536, -11456, -11384, -11288, -11176, -11104, -11016, -10952, -10864, -10776, -10688, -10616, -10544, -10480, -10392, -10304, -10264, -10176, -10080, -10032, -9944, -9896, -9840, -9776, -9704, -9656, -9592, -9528, -9480, -9424, -9360, -9288, -9272, -9192, -9152, -9096, -9072, -9016, -8984, -8952, -8912, -8880, -8816, -8792, -8760, -8688, -8672, -8640, -8632, -8584, -8576, -8560, -8512, -8512, -8504, -8456, -8440, -8424, -8408, -8408, -8416, -8392, -8392, -8360, -8392, -8368, -8392, -8376, -8368, -8384, -8376, -8392, -8392, -8392, -8392, -8408, -8440, -8440, -8448, -8480, -8504, -8488, -8528, -8560, -8560, -8592, -8640, -8664, -8672, -8696, -8720, -8744, -8816, -8824, -8864, -8888, -8928, -8976, -9008, -9064, -9104, -9128, -9208, -9232, -9296, -9336, -9392, -9464, -9496, -9560, -9616, -9680, -9736, -9776, -9856, -9896, -9984, -10024, -10080, -10152, -10224, -10280, -10352, -10440, -10512, -10552, -10648, -10688, -10784, -10856, -10920, -10984, -11072, -11136, -11224, -11304, -11384, -11440, -11536, -11624, -11696, -11768, -11840, -11920, -12032, -12128, -12208, -12288, -12408, -12472, -12568, -12640, -12752, -12824, -12912, -13000, -13088, -13184, -13288, -13384, -13488, -13576, -13680, -13776, -13856, -13968, -14040, -14144, -14264, -14344, -14432, -14544, -14640, -14752, -14848, -14928, -15016, -15136, -15232, -15304, -15448, -15544, -15664, -15744, -15824, -15944, -16032, -16160, -16256, -16352, -16448, -16552, -16648, -16768, -16856, -16976, -17056, -17184, -17280, -17392, -17496, -17608, -17704, -17784, -17880, -17976, -18096, -18216, -18304, -18416, -18512, -18640, -18712, -18816, -18912, -19016, -19136, -19232, -19344, -19432, -19544, -19640, -19736, -19824, -19928, -20000, -20136, -20240, -20328, -20416, -20520, -20592, -20680, -20832, -20896, -20992, -21096, -21192, -21296, -21392, -21496, -21600, -21656, -21760, -21872, -21952, -22048, -22152, -22232, -22328, -22400, -22496, -22584, -22680, -22768, -22872, -22936, -23024, -23128, -23192, -23312, -23384, -23472, -23560, -23632, -23712, -23792, -23872, -23976, -24040, -24112, -24176, -24320, -24368, -24464, -24552, -24608, -24664, -24784, -24840, -24888, -24992, -25072, -25152, -25240, -25288, -25352, -25424, -25496, -25584, -25656, -25752, -25816, -25888, -25944, -25984, -26048, -26152, -26224, -26280, -26336, -26400, -26488, -26544, -26584, -26656, -26720, -26760, -26864, -26904, -26952, -27032, -27096, -27128, -27208, -27240, -27304, -27368, -27424, -27480, -27520, -27600, -27640, -27720, -27760, -27808, -27848, -27912, -27944, -27992, -28080, -28096, -28168, -28208, -28232, -28320, -28360, -28392, -28416, -28496, -28544, -28576, -28632, -28672, -28720, -28736, -28808, -28840, -28880, -28912, -28960, -28960, -29040, -29048, -29120, -29136, -29192, -29200, -29240, -29272, -29296, -29312, -29368, -29416, -29448, -29480, -29520, -29552, -29592, -29608, -29648, -29664, -29720, -29712, -29752, -29808, -29816, -29856, -29896, -29912, -29920, -29968, -30008, -30032, -30072, -30080, -30104, -30128, -30152, -30176, -30224, -30224, -30240, -30264, -30264, -30320, -30320, -30360, -30360, -30392, -30424, -30408, -30464, -30456, -30504, -30496, -30520, -30544, -30544, -30576, -30560, -30600, -30640, -30632, -30656, -30664, -30688, -30680, -30720, -30736, -30768, -30800, -30776, -30776, -30800, -30816, -30832, -30848, -30872, -30848, -30872, -30912, -30928, -30936, -30912, -30968, -30952, -30944, -30952, -30968, -30992, -31008, -30984, -31016, -31008, -31032, -31040, -31048, -31080, -31080, -31056, -31088, -31088, -31104, -31120, -31112, -31120, -31152, -31128, -31168, -31144, -31160, -31168, -31168, -31176, -31176, -31184, -31208, -31208, -31232, -31232, -31232, -31240, -31216, -31224, -31248, -31264, -31280 };
	fitData result[1];

	data = testData;

	clock_t starttime = clock();
	kernel<0>(sampleCount, 1, result);
	clock_t stoptime = clock();

	printf("calculation time: %d ms\n\n\n", (int)(stoptime - starttime) * 1000 / CLOCKS_PER_SEC);
#endif

	for (int i = 0; i < chunkCount; i++) {
		printf("----- CHUNK %d -----\n\n", i);
		printf("status: %d\n\n", result[i].status);

		printf("y of start: %f\n", result[i].startY);
		printf("y of end:   %f\n\n", result[i].endY);

		printf("x of first sample used for fit: %d\n", result[i].firstSampleForFitX);
		printf("count of samples used for fit:  %d (max. %d)\n\n", result[i].samplesForFitCount, maxSampleCount);

		printf("fit-function:               y = %f * t ^ 2 + %f * t + %f\n", result[i].param[0], result[i].param[1], result[i].param[2]);
		printf("x of extremum:              %f\n", result[i].extremumX);
		printf("y of extremum:              %f\n", result[i].extremumY);
		printf("sum of squares of residues: %f\n", result[i].sumSqrResidues);
		printf("euclidean norm:             %f\n", sqrtf(result[i].sumSqrResidues));
		printf("sum of absolute residues:   %f\n", result[i].sumAbsResidues);
		printf("average residue:            %f\n", result[i].sumAbsResidues / result[i].samplesForFitCount);
		printf("count of iterations:        %d (max. %d)\n\n\n", result[i].iterationCount, maxIterations);
	}

	while (true);
	//return 0;
}

#endif
