//! \file

#ifndef LEVMARQ_H
#define LEVMARQ_H

#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include "Types.h"

#define CUDA //defined: runs on GPU, otherwise on CPU (useful for debugging)

#ifdef CUDA
texture<DATATYPE, 2, cudaReadModeElementType> dataTexture0, dataTexture1, dataTexture2, dataTexture3, dataTexture4, dataTexture5;

/*!
 * \brief getSample returns the y value of a given sample index
 * \param I sample index
 * \param INDEXDATASET index of the current dataset (GPU mode) or not used (CPU mode)
 * \return y value
*/
template<unsigned int tex>
__device__ float getSample(float I, int INDEXDATASET);

template<> __device__ float getSample<0>(float I, int INDEXDATASET) {
	return tex2D(dataTexture0, (I) + 0.5, (INDEXDATASET) + 0.5);
}
template<> __device__ float getSample<1>(float I, int INDEXDATASET) {
	return tex2D(dataTexture1, (I) + 0.5, (INDEXDATASET) + 0.5);
}
template<> __device__ float getSample<2>(float I, int INDEXDATASET) {
	return tex2D(dataTexture2, (I) + 0.5, (INDEXDATASET) + 0.5);
}
template<> __device__ float getSample<3>(float I, int INDEXDATASET) {
	return tex2D(dataTexture3, (I) + 0.5, (INDEXDATASET) + 0.5);
}
template<> __device__ float getSample<4>(float I, int INDEXDATASET) {
	return tex2D(dataTexture4, (I) + 0.5, (INDEXDATASET) + 0.5);
}
template<> __device__ float getSample<5>(float I, int INDEXDATASET) {
	return tex2D(dataTexture5, (I) + 0.5, (INDEXDATASET) + 0.5);
}
#else
DATATYPE *data;
#define getSample(I, INDEXDATASET) data[(int)(I)] //INDEXDATASET has no effect (only for CUDA)
#endif

#ifdef CUDA
#define GLOBAL __global__
#define DEVICE __device__
#define SHARED __shared__
#else
#define GLOBAL
#define DEVICE
#define SHARED
#endif

//machine-dependent constants from float.h
#define LM_MACHEP     FLT_EPSILON   //resolution of arithmetic
#define LM_DWARF      FLT_MIN       //smallest nonzero number
#define LM_SQRT_DWARF sqrt(FLT_MIN) //square should not underflow
#define LM_SQRT_GIANT sqrt(FLT_MAX) //square should not overflow
#define LM_USERTOL    30*LM_MACHEP  //users are recommened to require this

#define MIN(A, B) (((A) <= (B)) ? (A) : (B))
#define MAX(A, B) (((A) >= (B)) ? (A) : (B))
#define SQR(X)    ((X) * (X))


/*
TODO (see also Pflichtenheft.pdf)
 (1) maybe other fit-function (e. g. a*e^(b*(x+c)^2)+d (b < 0))
     -> a calculation of start and end values with this function is possible (instead of averaged data)
     Note: e-function with 4 parameters needs a good starting value, e. g. { 10000, -1, -500, -31000 } (first wave in Al_25keV-1.cdb)
	       -> it would be necessary to check the data before
 (2) maybe use shared memory with dynamic size (depending on countData and COUNTPARAM) instead of MAXCOUNTDATA
 (3) optimizing to use less memory -> higher MAXCOUNTDATA will be possible
 (4) maybe simultaneous calculations in one dataset (more than one thread per block)

CHANGES (already done)
 - horrible goto-instructions replaced
 - reduced to one file with integrated fit-function, residue calculation etc.
 - without controlling
 - without user interaction
 - fit-function
 - example data for testing
 - reduced to uniform distribution of x-coord. -> 1 array (instead of 2)
 - extremum calculation
 - cuda kernel etc.
 - return the fit function result
 - calculation of start and end values (averaged)
 - data array -> texture memory

 Note: Some original comments were not updated after code changes.
*/


/*
Original source: http://sourceforge.net/projects/lmfit/ (for Levenberg-Marquardt algorithm)
Authors:  Burton S. Garbow, Kenneth E. Hillstrom, Jorge J. More
          (lmdif and other routines from the public-domain library
          netlib::minpack, Argonne National Laboratories, March 1980);
          Steve Moshier (initial C translation);
          Joachim Wuttke (conversion into C++ compatible ANSI style,
          corrections, comments, wrappers, hosting).
*/

//#include "LevMarq.h"

const char *statusMessage[] = { //indexed by fitData.status
/* 0 */	"fatal coding error (improper input parameters)",
/* 1 */	"success (the relative error in the sum of squares is at most tol)",
/* 2 */	"success (the relative error between x and the solution is at most tol)",
/* 3 */	"success (the relative errors in the sum of squares and between x and the solution are at most tol)",
/* 4 */	"trapped by degeneracy (fvec is orthogonal to the columns of the jacobian)",
/* 5 */	"timeout (number of calls to fcn has reached maxcall*(n+1))",
/* 6 */	"failure (ftol<tol: cannot reduce sum of squares any further)",
/* 7 */	"failure (xtol<tol: cannot improve approximate solution any further)",
/* 8 */	"failure (gtol<tol: cannot improve approximate solution any further)"
};

//--- USER DEFINITIONS ---

/*!
 * \brief paramStartValue returns the parameter start values for the fit-function calculation
 * \param firstValue first value of the data used for fit-function
 * \param lastValue last value of the data used for fit-function
 * \param indexDataset index of the current dataset (GPU mode) or not used (CPU mode)
 * \param param the returned parameter start values
*/
template<unsigned int tex>
DEVICE void paramStartValue(int firstValue, int lastValue, int indexDataset, float *param)
{
	long long int x1, y1, x2, y2, x3, y3, dv;

	x1 = firstValue;
	x2 = (lastValue - firstValue) / 2 + firstValue;
	x3 = lastValue;
	y1 = getSample<tex>(x1, indexDataset);
	y2 = getSample<tex>(x2, indexDataset);
	y3 = getSample<tex>(x3, indexDataset);

	//any value, but not { 0, 0, 0 }
	dv = (x2-x1)*(x3-x1)*(x2-x3);
	param[0] = (-x1*y2+x1*y3+x2*y1-x2*y3-x3*y1+x3*y2)/dv;
	param[1] = (x1*x1*y2-x1*x1*y3-x2*x2*y1+x2*x2*y3+x3*x3*y1-x3*x3*y2)/dv;
	param[2] = (x1*x1*x2*y3+x1*x1*-x3*y2-x1*x2*x2*y3+x1*x3*x3*y2+x2*x2*x3*y1-x2*x3*x3*y1)/dv;
}

/*!
 * \brief fitFunction returns the y of a given x
 * \param x given x value to calculate y
 * \param param parameters to define the concrete current fit-function
 * \param y the returned y value
*/
DEVICE inline void fitFunction(float x, float *param, float *y)
{
	*y = param[0] * SQR(x) + param[1] * x + param[2];
	//*y = param[0] * exp(param[1] * (x + param[2]) * (x + param[2])) + param[3];
}

/*!
 * \brief fitFunctionExtremum returns the x of the min. or max. y value
 * \param param parameters to define the concrete current fit-function
 * \param x the returned x value
*/
DEVICE inline void fitFunctionExtremum(float *param, float *x) //get x
{
	//f': y = 2 * param[0] * x + param[1] and y = 0
	if (param[0] == 0)
		*x = 0; //no Extremum
	else
		*x = -param[1] / (2 * param[0]);
}

//------------------------

/*!
 * \brief evaluate calculates the residues between the given samples and the current fit-function
 * \param param parameters to define the concrete current fit-function
 * \param countData number of samples
 * \param fvec the returned residues
 * \param indexDataset index of the current dataset (GPU mode) or not used (CPU mode)
 * \param xOffset first x value that is used to calculate the fit-function
 * \param xStep the distance between two x values that are used to calculate the fit-function (if decimal then y values will be interpolated)
*/
template<unsigned int tex>
DEVICE void evaluate(float *param, int countData, float *fvec, int indexDataset, int xOffset, float xStep)
{
	int i;
	float y;

	for (i = 0; i < countData; i++) {
		fitFunction(i * xStep + xOffset, param, &y);
		fvec[i] = getSample<tex>(i * xStep + xOffset, indexDataset) - y;
	}
}

/*!
 * \brief qrSolve completes the solution of the problem if it is provided with the necessary information from the qr factorization, with column pivoting, of a
 * \param n width and height of array r
 * \param ldr a positive integer input variable not less than n which specifies the leading dimension of the array r
 * \param ipvt an integer input array of length n which defines the permutation matrix p such that a*p = q*r
 * \param diag an input array of length n which must contain the diagonal elements of the matrix d
 * \param qtb an input array of length n which must contain the first n elements of the vector (q transpose)*b
 * \param x an output array of length n which contains the least squares solution of the system a*x = b, d*x = 0
 * \param sdiag an output array of length n which contains the diagonal elements of the upper triangular matrix s
 * \param wa work array of length n
*/
DEVICE void qrSolve(int n, float *r, int ldr, int *ipvt, float *diag,
			   float *qtb, float *x, float *sdiag, float *wa)
{
	int i, kk, j, k, nsing;
	float qtbpj, sum, temp;
	float _sin, _cos, _tan, _cot;

	//copy r and (q transpose)*b to preserve input and initialize s
	//in particular, save the diagonal elements of r in x

	for (j = 0; j < n; j++) {
		for (i = j; i < n; i++)
			r[j * ldr + i] = r[i * ldr + j];
		x[j] = r[j * ldr + j];
		wa[j] = qtb[j];
	}

	//eliminate the diagonal matrix d using a givens rotation

	for (j = 0; j < n; j++) {

		//prepare the row of d to be eliminated, locating the diagonal element using p from the qr factorization.

		if (diag[ipvt[j]] != 0.)
		{
			for (k = j; k < n; k++)
				sdiag[k] = 0.;
			sdiag[j] = diag[ipvt[j]];

			//the transformations to eliminate the row of d modify only a single element of (q transpose)*b beyond the first n, which is initially 0.

			qtbpj = 0.;
			for (k = j; k < n; k++) {

				//determine a givens rotation which eliminates the appropriate element in the current row of d

				if (sdiag[k] == 0.)
					continue;
				kk = k + ldr * k;
				if (fabs(r[kk]) < fabs(sdiag[k])) {
					_cot = r[kk] / sdiag[k];
					_sin = 1 / sqrt(1 + SQR(_cot));
					_cos = _sin * _cot;
				} else {
					_tan = sdiag[k] / r[kk];
					_cos = 1 / sqrt(1 + SQR(_tan));
					_sin = _cos * _tan;
				}

				//compute the modified diagonal element of r and the modified element of ((q transpose)*b, 0)

				r[kk] = _cos * r[kk] + _sin * sdiag[k];
				temp = _cos * wa[k] + _sin * qtbpj;
				qtbpj = -_sin * wa[k] + _cos * qtbpj;
				wa[k] = temp;

				//accumulate the tranformation in the row of s

				for (i = k + 1; i < n; i++) {
					temp = _cos * r[k * ldr + i] + _sin * sdiag[i];
					sdiag[i] = -_sin * r[k * ldr + i] + _cos * sdiag[i];
					r[k * ldr + i] = temp;
				}
			}
		}

		//store the diagonal element of s and restore the corresponding diagonal element of r

		sdiag[j] = r[j * ldr + j];
		r[j * ldr + j] = x[j];
	}

	//solve the triangular system for z
	//if the system is singular, then obtain a least squares solution

	nsing = n;
	for (j = 0; j < n; j++) {
		if (sdiag[j] == 0. && nsing == n)
			nsing = j;
		if (nsing < n)
			wa[j] = 0;
	}

	for (j = nsing - 1; j >= 0; j--) {
		sum = 0;
		for (i = j + 1; i < nsing; i++)
			sum += r[j * ldr + i] * wa[i];
		wa[j] = (wa[j] - sum) / sdiag[j];
	}

	//permute the components of z back to components of x

	for (j = 0; j < n; j++)
		x[ipvt[j]] = wa[j];
}

/*!
 * \brief euclidNorm calculates the euclidean norm of x
 * \param n length of array x
 * \param x array for euclidean norm
 * \param result euclidean norm of x
*/
DEVICE void euclidNorm(int n, float *x, float* result)
{
	int i;
	float agiant, s1, s2, s3, xabs, x1max, x3max, temp;

	s1 = 0;
	s2 = 0;
	s3 = 0;
	x1max = 0;
	x3max = 0;
	agiant = LM_SQRT_GIANT / ((float) n);

	//sum squares
	for (i = 0; i < n; i++) {
		xabs = fabs(x[i]);
		if (xabs > LM_SQRT_DWARF && xabs < agiant) {
			//sum for intermediate components
			s2 += xabs * xabs;
			continue;
		}

		if (xabs > LM_SQRT_DWARF) {
			//sum for large components
			if (xabs > x1max) {
				temp = x1max / xabs;
				s1 = 1 + s1 * SQR(temp);
				x1max = xabs;
			} else {
				temp = xabs / x1max;
				s1 += SQR(temp);
			}
			continue;
		}
		//sum for small components
		if (xabs > x3max) {
			temp = x3max / xabs;
			s3 = 1 + s3 * SQR(temp);
			x3max = xabs;
		} else {
			if (xabs != 0.) {
				temp = xabs / x3max;
				s3 += SQR(temp);
			}
		}
	}

	///calculation of norm

	if (s1 != 0)
		*result = x1max * sqrt(s1 + (s2 / x1max) / x1max);
	else if (s2 != 0) {
		if (s2 >= x3max)
			*result = sqrt(s2 * (1 + (x3max / s2) * (x3max * s3)));
		else
			*result = sqrt(x3max * ((s2 / x3max) + (x3max * s3)));
	}
	else
		*result = x3max * sqrt(s3);

}

/*!
 * \brief lmpar determines a value for the parameter par such that x solves the system
 * \param n width and height of array r
 * \param ldr a positive integer input variable not less than n which specifies the leading dimension of the array r
 * \param ipvt an integer input array of length n which defines the permutation matrix p such that a*p = q*r
 * \param diag an input array of length n which must contain the diagonal elements of the matrix d
 * \param qtb an input array of length n which must contain the first n elements of the vector (q transpose)*b
 * \param delta a positive input variable which specifies an upper bound on the euclidean norm of d*x
 * \param par input: contains an initial estimate of the levenberg-marquardt parameter; output: contains the final estimate
 * \param x an output array of length n which contains the least squares solution of the system a*x = b, d*x = 0
 * \param sdiag an output array of length n which contains the diagonal elements of the upper triangular matrix s
 * \param wa1 work array of length n
 * \param wa2 work array of length n
*/
DEVICE void lmpar(int n, float *r, int ldr, int *ipvt, float *diag,
			  float *qtb, float delta, float *par, float *x,
			  float *sdiag, float *wa1, float *wa2)
{
	int i, iter, j, nsing;
	float dxnorm, fp, fp_old, gnorm, parc, parl, paru;
	float sum, temp;

	//compute and store in x the gauss-newton direction
	//if the jacobian is rank-deficient, obtain a least squares solution

	nsing = n;
	for (j = 0; j < n; j++) {
		wa1[j] = qtb[j];
		if (r[j * ldr + j] == 0 && nsing == n)
			nsing = j;
		if (nsing < n)
			wa1[j] = 0;
	}

	for (j = nsing - 1; j >= 0; j--) {
		wa1[j] = wa1[j] / r[j + ldr * j];
		temp = wa1[j];
		for (i = 0; i < j; i++)
			wa1[i] -= r[j * ldr + i] * temp;
	}

	for (j = 0; j < n; j++)
		x[ipvt[j]] = wa1[j];

	//initialize the iteration counter, evaluate the function at the origin, and test for acceptance of the gauss-newton direction

	iter = 0;
	for (j = 0; j < n; j++)
		wa2[j] = diag[j] * x[j];
	euclidNorm(n, wa2, &dxnorm);
	fp = dxnorm - delta;
	if (fp <= 0.1 * delta) {
		*par = 0;
		return;
	}

	//if the jacobian is not rank deficient, the newton step provides a lower bound, parl, for the 0. of the function
	//otherwise set this bound to 0.

	parl = 0;
	if (nsing >= n) {
		for (j = 0; j < n; j++)
			wa1[j] = diag[ipvt[j]] * wa2[ipvt[j]] / dxnorm;

		for (j = 0; j < n; j++) {
			sum = 0.;
			for (i = 0; i < j; i++)
				sum += r[j * ldr + i] * wa1[i];
			wa1[j] = (wa1[j] - sum) / r[j + ldr * j];
		}
		euclidNorm(n, wa1, &temp);
		parl = fp / delta / temp / temp;
	}

	//calculate an upper bound, paru, for the 0. of the function

	for (j = 0; j < n; j++) {
		sum = 0;
		for (i = 0; i <= j; i++)
			sum += r[j * ldr + i] * qtb[i];
		wa1[j] = sum / diag[ipvt[j]];
	}
	euclidNorm(n, wa1, &gnorm);
	paru = gnorm / delta;
	if (paru == 0.)
		paru = LM_DWARF / MIN(delta, 0.1);

	//if the input par lies outside of the interval (parl, paru), set par to the closer endpoint

	*par = MAX(*par, parl);
	*par = MIN(*par, paru);
	if (*par == 0.)
		*par = gnorm / dxnorm;

	for (;; iter++) {

		//evaluate the function at the current value of par

		if (*par == 0.)
			*par = MAX(LM_DWARF, 0.001 * paru);
		temp = sqrt(*par);
		for (j = 0; j < n; j++)
			wa1[j] = temp * diag[j];
		qrSolve(n, r, ldr, ipvt, wa1, qtb, x, sdiag, wa2);
		for (j = 0; j < n; j++)
			wa2[j] = diag[j] * x[j];
		euclidNorm(n, wa2, &dxnorm);
		fp_old = fp;
		fp = dxnorm - delta;

		//if the function is small enough, accept the current value of par
		//also test for the exceptional cases where parl is zero or the number of iterations has reached 10

		if (fabs(fp) <= 0.1 * delta
			|| (parl == 0. && fp <= fp_old && fp_old < 0.)
			|| iter == 10)
			break; //the only exit from the iteration

		//compute the Newton correction

		for (j = 0; j < n; j++)
			wa1[j] = diag[ipvt[j]] * wa2[ipvt[j]] / dxnorm;

		for (j = 0; j < n; j++) {
			wa1[j] = wa1[j] / sdiag[j];
			for (i = j + 1; i < n; i++)
				wa1[i] -= r[j * ldr + i] * wa1[j];
		}
		euclidNorm(n, wa1, &temp);
		parc = fp / delta / temp / temp;

		//depending on the sign of the function, update parl or paru

		if (fp > 0)
			parl = MAX(parl, *par);
		else if (fp < 0)
			paru = MIN(paru, *par);
		//the case fp==0 is precluded by the break condition

		//compute an improved estimate for par

		*par = MAX(parl, *par + parc);
	}
}

/*!
 * \brief qrFactorization uses householder transformations with column pivoting (optional) to compute a qr factorization of the m by n matrix a
 * \param m height of array a
 * \param n width of array a
 * \param a input: contains the matrix for which the qr factorization is to be computed; output: the strict upper trapezoidal part of a contains the strict upper trapezoidal part of r, and the lower trapezoidal part of a contains a factored form of q
 * \param pivot if is set true then column pivoting is enforced; if is set false then no column pivoting is done
 * \param ipvt defines the permutation matrix p such that a*p = q*r
 * \param rdiag an output array of length n which contains the diagonal elements of r
 * \param acnorm an output array of length n which contains the norms of the corresponding columns of the input matrix a
 * \param wa work array of length n
*/
DEVICE void qrFactorization(int m, int n, float *a, int pivot, int *ipvt,
			  float *rdiag, float *acnorm, float *wa)
{
	int i, j, k, kmax, minmn;
	float ajnorm, sum, temp;

	//compute initial column norms and initialize several arrays

	for (j = 0; j < n; j++) {
		euclidNorm(m, &a[j * m], &acnorm[j]);
		rdiag[j] = acnorm[j];
		wa[j] = rdiag[j];
		if (pivot)
			ipvt[j] = j;
	}

	//reduce a to r with householder transformations

	minmn = MIN(m, n);
	for (j = 0; j < minmn; j++) {
		if (pivot)
		{
			//bring the column of largest norm into the pivot position

			kmax = j;
			for (k = j + 1; k < n; k++)
				if (rdiag[k] > rdiag[kmax])
					kmax = k;
			if (kmax != j)
			{
				for (i = 0; i < m; i++) {
					temp = a[j * m + i];
					a[j * m + i] = a[kmax * m + i];
					a[kmax * m + i] = temp;
				}
				rdiag[kmax] = rdiag[j];
				wa[kmax] = wa[j];
				k = ipvt[j];
				ipvt[j] = ipvt[kmax];
				ipvt[kmax] = k;
			}
		}

        //pivot_ok

		//compute the Householder transformation to reduce the j-th column of a to a multiple of the j-th unit vector

		euclidNorm(m - j, &a[j * m + j], &ajnorm);
		if (ajnorm == 0.) {
			rdiag[j] = 0;
			continue;
		}

		if (a[j * m + j] < 0.)
			ajnorm = -ajnorm;
		for (i = j; i < m; i++)
			a[j * m + i] /= ajnorm;
		a[j * m + j] += 1;

		//apply the transformation to the remaining columns and update the norms

		for (k = j + 1; k < n; k++) {
			sum = 0;

			for (i = j; i < m; i++)
				sum += a[j * m + i] * a[k * m + i];

			temp = sum / a[j + m * j];

			for (i = j; i < m; i++)
				a[k * m + i] -= temp * a[j * m + i];

			if (pivot && rdiag[k] != 0.) {
				temp = a[m * k + j] / rdiag[k];
				temp = MAX(0., 1 - temp * temp);
				rdiag[k] *= sqrt(temp);
				temp = rdiag[k] / wa[k];
				if (0.05 * SQR(temp) <= LM_MACHEP) {
					euclidNorm(m - j - 1, &a[m * k + j + 1], &rdiag[k]);
					wa[k] = rdiag[k];
				}
			}
		}

		rdiag[j] = -ajnorm;
	}
}

/*!
 * \brief lmdif minimizes the sum of the squares of m nonlinear functions in n variables by a modification of the levenberg-marquardt algorithm
 * \param m number of samples
 * \param n number of parameters
 * \param x input: must contain an initial estimate of the solution vector; output: contains the final estimate of the solution vector
 * \param fvec an output array of length m which contains the functions evaluated at the output x
 * \param ftol measures the relative error desired in the sum of squares
 * \param xtol measures the relative error desired in the approximate solution
 * \param gtol gtol measures the orthogonality desired between the function vector and the columns of the jacobian
 * \param maxfev a integer input variable that is used to terminate when the number of calls is at least maxfev by the end of an iteration
 * \param epsfcn an input variable used in determining a suitable step length for the forward-difference approximation
 * \param mode if mode = 1 then the variables will be scaled internally, if mode = 2 then the scaling is specified by the input diag
 * \param factor a input variable used in determining the initial step bound. This bound is set to the product of factor and the euclidean norm of diag*x
 * \param info an integer output variable that indicates the termination status of lmdif (see statusMessage)
 * \param nfev an output variable set to the number of calls to the user-supplied routine *evaluate
 * \param fjac an output m by n array. The upper n by n submatrix of fjac contains an upper triangular matrix r with diagonal elements of nonincreasing magnitude
 * \param ipvt an integer output array of length n that defines a permutation matrix p such that jac*p = q*r
 * \param qtf an output array of length n which contains the first n elements of the vector (q transpose)*fvec
 * \param wa1 work array of length n
 * \param wa2 work array of length n
 * \param wa3 work array of length n
 * \param wa4 work array of length m
 * \param indexDataset index of the current dataset (GPU mode) or not used (CPU mode)
 * \param xOffset first x value that is used to calculate the fit-function
 * \param xStep the distance between two x values that are used to calculate the fit-function (if decimal then y values will be interpolated)
*/
template<unsigned int tex>
DEVICE void lmdif(int m, int n, float *x, float *fvec, float ftol, float xtol, float gtol, int maxfev, float epsfcn,
			  float *diag, int mode, float factor, int *info, int *nfev, float *fjac, int *ipvt, float *qtf, float *wa1,
			  float *wa2, float *wa3, float *wa4, int indexDataset, int xOffset, float xStep)
{
	int i, iter, j;
	float actred, delta, dirder, eps, fnorm, fnorm1, gnorm, par, pnorm,
		prered, ratio, step, sum, temp, temp1, temp2, temp3, xnorm;

	*nfev = 0;			//function evaluation counter
	iter = 1;			//outer loop counter
	par = 0;			//levenberg-marquardt parameter
	delta = 0;	 //to prevent a warning (initialization within if-clause)
	xnorm = 0;	 //ditto
	temp = MAX(epsfcn, LM_MACHEP);
	eps = sqrt(temp); //for calculating the Jacobian by forward differences

	//check input parameters for errors.

	if ((n <= 0) || (m < n) || (ftol < 0.)
		|| (xtol < 0.) || (gtol < 0.) || (maxfev <= 0) || (factor <= 0.)) {
			*info = 0;		//invalid parameter
			return;
	}
	if (mode == 2) {		//scaling by diag[]
		for (j = 0; j < n; j++) {	//check for nonpositive elements
			if (diag[j] <= 0.0) {
				*info = 0;	//invalid parameter
				return;
			}
		}
	}

	//evaluate function at starting point and calculate norm

	*info = 0;
	evaluate<tex>(x, m, fvec, indexDataset, xOffset, xStep);
	++(*nfev);
	euclidNorm(m, fvec, &fnorm);

	do {

		//calculate the jacobian matrix

		for (j = 0; j < n; j++) {
			temp = x[j];
			step = eps * fabs(temp);
			if (step == 0.)
				step = eps;
			x[j] = temp + step;
			*info = 0;
			evaluate<tex>(x, m, wa4, indexDataset, xOffset, xStep);
			for (i = 0; i < m; i++) //changed in 2.3, Mark Bydder
				fjac[j * m + i] = (wa4[i] - fvec[i]) / (x[j] - temp);
			x[j] = temp;
		}

		//compute the qr factorization of the jacobian

		qrFactorization(m, n, fjac, 1, ipvt, wa1, wa2, wa3);

		if (iter == 1) { //first iteration
			if (mode != 2) {
				//diag := norms of the columns of the initial jacobian
				for (j = 0; j < n; j++) {
					diag[j] = wa2[j];
					if (wa2[j] == 0.)
						diag[j] = 1.;
				}
			}
			//use diag to scale x, then calculate the norm
			for (j = 0; j < n; j++)
				wa3[j] = diag[j] * x[j];
			euclidNorm(n, wa3, &xnorm);
			//initialize the step bound delta
			delta = factor * xnorm;
			if (delta == 0.)
				delta = factor;
		}

		//form (q transpose)*fvec and store first n components in qtf

		for (i = 0; i < m; i++)
			wa4[i] = fvec[i];

		for (j = 0; j < n; j++) {
			temp3 = fjac[j * m + j];
			if (temp3 != 0.) {
				sum = 0;
				for (i = j; i < m; i++)
					sum += fjac[j * m + i] * wa4[i];
				temp = -sum / temp3;
				for (i = j; i < m; i++)
					wa4[i] += fjac[j * m + i] * temp;
			}
			fjac[j * m + j] = wa1[j];
			qtf[j] = wa4[j];
		}

		//compute norm of scaled gradient and test for convergence

		gnorm = 0;
		if (fnorm != 0) {
			for (j = 0; j < n; j++) {
				if (wa2[ipvt[j]] == 0)
					continue;

				sum = 0.;
				for (i = 0; i <= j; i++)
					sum += fjac[j * m + i] * qtf[i] / fnorm;
				gnorm = MAX(gnorm, fabs(sum / wa2[ipvt[j]]));
			}
		}

		if (gnorm <= gtol) {
			*info = 4;
			return;
		}

		//rescale if necessary

		if (mode != 2) {
			for (j = 0; j < n; j++)
				diag[j] = MAX(diag[j], wa2[j]);
		}

		do {
			//determine the levenberg-marquardt parameter

			lmpar(n, fjac, m, ipvt, diag, qtf, delta, &par,
				wa1, wa2, wa3, wa4);

			//store the direction p and x + p; calculate the norm of p

			for (j = 0; j < n; j++) {
				wa1[j] = -wa1[j];
				wa2[j] = x[j] + wa1[j];
				wa3[j] = diag[j] * wa1[j];
			}
			euclidNorm(n, wa3, &pnorm);

			//on the first iteration, adjust the initial step bound

			if (*nfev <= 1 + n)
				delta = MIN(delta, pnorm);

			//evaluate the function at x + p and calculate its norm

			*info = 0;
			evaluate<tex>(wa2, m, wa4, indexDataset, xOffset, xStep);
			++(*nfev);

			euclidNorm(m, wa4, &fnorm1);

			//compute the scaled actual reduction

			if (0.1 * fnorm1 < fnorm)
				actred = 1 - SQR(fnorm1 / fnorm);
			else
				actred = -1;

			//compute the scaled predicted reduction and the scaled directional derivative

			for (j = 0; j < n; j++) {
				wa3[j] = 0;
				for (i = 0; i <= j; i++)
					wa3[i] += fjac[j * m + i] * wa1[ipvt[j]];
			}
			euclidNorm(n, wa3, &temp1);
			temp1 /= fnorm;
			temp2 = sqrt(par) * pnorm / fnorm;
			prered = SQR(temp1) + 2 * SQR(temp2);
			dirder = -(SQR(temp1) + SQR(temp2));

			//compute the ratio of the actual to the predicted reduction

			ratio = prered != 0 ? actred / prered : 0;

			//update the step bound

			if (ratio <= 0.25) {
				if (actred >= 0.)
					temp = 0.5;
				else
					temp = 0.5 * dirder / (dirder + 0.55 * actred);
				if (0.1 * fnorm1 >= fnorm || temp < 0.1)
					temp = 0.1;
				delta = temp * MIN(delta, pnorm / 0.1);
				par /= temp;
			} else if (par == 0. || ratio >= 0.75) {
				delta = pnorm / 0.5;
				par *= 0.5;
			}

			//test for successful iteration

			if (ratio >= 0.0001) {
				//yes, success: update x, fvec, and their norms
				for (j = 0; j < n; j++) {
					x[j] = wa2[j];
					wa2[j] = diag[j] * x[j];
				}
				for (i = 0; i < m; i++)
					fvec[i] = wa4[i];
				euclidNorm(n, wa2, &xnorm);
				fnorm = fnorm1;
				iter++;
			}

			//tests for convergence (otherwise *info = 1, 2, or 3)

			*info = 0; //do not terminate (unless overwritten by nonzero)
			if (fabs(actred) <= ftol && prered <= ftol && 0.5 * ratio <= 1)
				*info = 1;
			if (delta <= xtol * xnorm)
				*info += 2;
			if (*info != 0)
				return;

			//tests for termination and stringent tolerances

			if (*nfev >= maxfev)
				*info = 5;
			if (fabs(actred) <= LM_MACHEP &&
				prered <= LM_MACHEP && 0.5 * ratio <= 1)
				*info = 6;
			if (delta <= LM_MACHEP * xnorm)
				*info = 7;
			if (gnorm <= LM_MACHEP)
				*info = 8;
			if (*info != 0)
				return;

			//repeat if iteration unsuccessful
		} while (ratio < 0.0001);
	} while (1);
}

/*!
 * \brief maxValue returns the x and y where y has the greatest value
 * \param countData number of samples
 * \param indexDataset index of the current dataset (GPU mode) or not used (CPU mode)
 * \param x the returned x value
 * \param y the returned y value
*/
template<unsigned int tex>
DEVICE void maxValue(int countData, int indexDataset, int *x, DATATYPE *y)
{
	int i;
	*x = 0;
	*y = getSample<tex>(0, indexDataset);
	for (i = 0; i < countData; i++)
		if (getSample<tex>(i, indexDataset) > *y) {
			*y = getSample<tex>(i, indexDataset);
			*x = i;
		}
}

/*!
 * \brief averageValue returns the average of all y values in a given range
 * \param start first x for average calculation
 * \param count number of values for average calculation
 * \param indexDataset index of the current dataset (GPU mode) or not used (CPU mode)
 * \param y the returned average
*/
template<unsigned int tex>
DEVICE void averageValue(int start, int count, int indexDataset, float *y)
{
	int i;
	float sum = 0;

	for (i = start; i < start + count; i++)
		sum += getSample<tex>(i, indexDataset);
	*y = sum / count;
}

/*!
 * \brief xOfValue returns the first x of a value y that is greater or equal of a given min. value
 * \param countData number of samples
 * \param indexDataset index of the current dataset (GPU mode) or not used (CPU mode)
 * \param fromDirection 
 * \param minValue min. y value
 * \param x the returned x value, -1 if there is no x with a y greater or equal minValue
*/
template<unsigned int tex>
DEVICE void xOfValue(int countData, int indexDataset, char fromDirection, DATATYPE minValue, int *x)
{
	int i;
	*x = -1;
	if (fromDirection == 'l') {
		for (i = 0; i < countData; i++)
			if (getSample<tex>(i, indexDataset) >= minValue) {
				*x = i;
				break;
			}
	}
	else if (fromDirection == 'r')
		for (i = countData - 1; i >= 0; i--)
			if (getSample<tex>(i, indexDataset) >= minValue) {
				*x = i;
				break;
			}
}

/*!
 * \brief averageAbsResidues returns the average of the residues absolute value
 * \param countResidues number of residues
 * \param residues array of length countResidues that contains the residues
 * \param average the returned average
*/
DEVICE void averageAbsResidues(int countResidues, float *residues, float *average)
{
	int i;
	float sum = 0;

	for (i = 0; i < countResidues; i++)
		sum += fabs(residues[i]);
	*average = sum / countResidues;
}

/*!
 * \brief kernel is the start method for calculation (you have to set the dataTexture (GPU mode) or data variable (CPU mode) before calling this method)
 * \param countData number of samples
 * \param step the distance between two x values that are used to calculate the fit-function (if decimal then y values will be interpolated)
 * \param result fit-function and other parameters, defined in fitData struct
*/
template<unsigned int tex>
GLOBAL void kernel(int countData, float step, struct fitData *result)
{
#ifdef CUDA
	int indexDataset = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

	if (threadIdx.x > 0 || threadIdx.y > 0 || threadIdx.z > 0)
		return; //currently, the number of threads per block must be 1 (in the future used for simultaneous calculations in one dataset)
#else
	int indexDataset = 0;
#endif
	int nfev = 0, info = 0, i;
	int maxX, firstValue, lastValue, countAverage;
	DATATYPE maxY;

	SHARED float param[COUNTPARAM];

	SHARED float fvec[MAXCOUNTDATA];
	SHARED float fjac[COUNTPARAM * MAXCOUNTDATA];
	SHARED float wa4[MAXCOUNTDATA];

	SHARED float diag[COUNTPARAM], qtf[COUNTPARAM];
	SHARED float wa1[COUNTPARAM], wa2[COUNTPARAM], wa3[COUNTPARAM];
	SHARED int ipvt[COUNTPARAM];

	maxValue<tex>(countData, indexDataset, &maxX, &maxY);
	xOfValue<tex>(countData, indexDataset, 'l', (maxY - getSample<tex>(0, indexDataset)) * FITVALUETHRESHOLD + getSample<tex>(0, indexDataset), &firstValue);
	xOfValue<tex>(countData, indexDataset, 'r', (maxY - getSample<tex>(countData - 1, indexDataset)) * FITVALUETHRESHOLD + getSample<tex>(countData - 1, indexDataset), &lastValue);

	paramStartValue<tex>(firstValue, lastValue, indexDataset, param);

	lmdif<tex>((int)((lastValue - firstValue) / step) + 1, COUNTPARAM, param, fvec, LM_USERTOL, LM_USERTOL, LM_USERTOL,
		MAXCALL * (COUNTPARAM + 1), LM_USERTOL, diag, 1, 100, &info,
		&nfev, fjac, ipvt, qtf, wa1, wa2, wa3, wa4, indexDataset, firstValue, step);

	euclidNorm((int)((lastValue - firstValue) / step) + 1, fvec, &result[indexDataset].euclidNormResidues);
	averageAbsResidues((int)((lastValue - firstValue) / step) + 1, fvec, &result[indexDataset].averageAbsResidues);

	for (i = 0; i < COUNTPARAM; i++)
		result[indexDataset].param[i] = param[i];
	countAverage = countData * STARTENDPROPORTION;
	if (countData > 0 && countAverage == 0)
		countAverage = 1;
	averageValue<tex>(0, countAverage, indexDataset, &result[indexDataset].startValue);
	averageValue<tex>(countData - countAverage, countAverage, indexDataset, &result[indexDataset].endValue);
	fitFunctionExtremum(param, &result[indexDataset].extremumPos);
	fitFunction(result[indexDataset].extremumPos, param, &result[indexDataset].extremumValue);
	result[indexDataset].status = info;
}



//example data, only for testing
/*int main()
{
#ifdef CUDA
	int countData = 1000;
	int countDatasets = 1;
	//DATATYPE testData[2][3] = { { 2, 5, 2 }, { 1, 4, 1 } };
	DATATYPE testData[1][1000] = { { -31576, -31560, -31552, -31544, -31560, -31576, -31552, -31560, -31552, -31576, -31560, -31576, -31568, -31576, -31552, -31560, -31568, -31552, -31552, -31576, -31552, -31576, -31568, -31576, -31552, -31552, -31536, -31536, -31536, -31552, -31568, -31560, -31544, -31568, -31552, -31560, -31552, -31528, -31568, -31560, -31544, -31568, -31560, -31576, -31576, -31560, -31552, -31552, -31568, -31568, -31584, -31560, -31568, -31560, -31576, -31552, -31576, -31560, -31592, -31568, -31568, -31568, -31552, -31560, -31544, -31552, -31576, -31560, -31560, -31552, -31568, -31536, -31560, -31560, -31560, -31544, -31544, -31576, -31568, -31544, -31528, -31552, -31568, -31552, -31560, -31552, -31560, -31552, -31560, -31552, -31568, -31544, -31552, -31544, -31544, -31568, -31544, -31560, -31536, -31560, -31568, -31568, -31552, -31560, -31544, -31560, -31552, -31568, -31560, -31576, -31568, -31576, -31544, -31568, -31560, -31528, -31560, -31544, -31544, -31560, -31552, -31544, -31520, -31560, -31552, -31552, -31568, -31544, -31544, -31552, -31544, -31552, -31544, -31568, -31560, -31560, -31568, -31544, -31552, -31552, -31560, -31536, -31560, -31552, -31568, -31552, -31552, -31568, -31568, -31552, -31552, -31544, -31568, -31584, -31560, -31552, -31536, -31544, -31536, -31552, -31552, -31552, -31552, -31512, -31560, -31536, -31544, -31544, -31536, -31552, -31560, -31552, -31552, -31536, -31544, -31560, -31568, -31560, -31552, -31552, -31568, -31560, -31568, -31592, -31576, -31544, -31568, -31560, -31568, -31552, -31544, -31576, -31568, -31544, -31552, -31560, -31576, -31552, -31568, -31560, -31560, -31544, -31552, -31568, -31552, -31552, -31536, -31544, -31560, -31552, -31544, -31544, -31544, -31528, -31544, -31560, -31576, -31568, -31576, -31552, -31544, -31544, -31560, -31552, -31536, -31576, -31552, -31560, -31560, -31560, -31560, -31560, -31568, -31568, -31544, -31584, -31560, -31568, -31544, -31552, -31568, -31536, -31536, -31520, -31544, -31536, -31552, -31552, -31552, -31544, -31536, -31512, -31504, -31536, -31528, -31528, -31528, -31512, -31504, -31520, -31496, -31520, -31488, -31504, -31512, -31496, -31488, -31496, -31496, -31496, -31504, -31472, -31480, -31456, -31456, -31472, -31472, -31472, -31456, -31456, -31424, -31440, -31424, -31432, -31376, -31408, -31384, -31384, -31368, -31368, -31336, -31336, -31320, -31304, -31312, -31280, -31248, -31256, -31264, -31216, -31248, -31200, -31192, -31128, -31136, -31144, -31120, -31096, -31088, -31080, -31048, -31016, -30992, -30968, -30976, -30952, -30920, -30904, -30872, -30856, -30840, -30800, -30768, -30728, -30704, -30680, -30648, -30648, -30600, -30560, -30528, -30496, -30456, -30416, -30368, -30336, -30304, -30264, -30232, -30192, -30128, -30104, -30048, -30000, -29960, -29904, -29872, -29816, -29760, -29720, -29656, -29632, -29576, -29488, -29440, -29400, -29344, -29288, -29232, -29176, -29112, -29048, -28984, -28896, -28872, -28824, -28728, -28680, -28568, -28504, -28464, -28368, -28312, -28248, -28176, -28088, -28016, -27936, -27848, -27792, -27696, -27624, -27528, -27456, -27368, -27272, -27192, -27112, -26992, -26920, -26816, -26720, -26648, -26536, -26448, -26344, -26280, -26168, -26080, -25992, -25872, -25776, -25680, -25576, -25480, -25368, -25240, -25168, -25048, -24944, -24824, -24720, -24600, -24520, -24408, -24280, -24176, -24096, -23936, -23832, -23704, -23616, -23488, -23408, -23256, -23144, -23000, -22912, -22776, -22632, -22544, -22400, -22272, -22160, -22056, -21904, -21776, -21664, -21512, -21408, -21288, -21144, -21056, -20888, -20784, -20656, -20536, -20416, -20280, -20144, -20032, -19896, -19776, -19640, -19512, -19392, -19240, -19144, -19008, -18864, -18720, -18624, -18488, -18344, -18248, -18088, -17976, -17840, -17704, -17576, -17464, -17328, -17200, -17064, -16944, -16832, -16688, -16552, -16448, -16312, -16176, -16072, -15944, -15832, -15704, -15576, -15448, -15352, -15200, -15080, -14976, -14816, -14736, -14616, -14512, -14384, -14280, -14176, -14056, -13936, -13808, -13696, -13608, -13480, -13376, -13256, -13160, -13056, -12936, -12816, -12744, -12632, -12528, -12416, -12336, -12208, -12136, -12048, -11928, -11824, -11752, -11640, -11536, -11456, -11384, -11288, -11176, -11104, -11016, -10952, -10864, -10776, -10688, -10616, -10544, -10480, -10392, -10304, -10264, -10176, -10080, -10032, -9944, -9896, -9840, -9776, -9704, -9656, -9592, -9528, -9480, -9424, -9360, -9288, -9272, -9192, -9152, -9096, -9072, -9016, -8984, -8952, -8912, -8880, -8816, -8792, -8760, -8688, -8672, -8640, -8632, -8584, -8576, -8560, -8512, -8512, -8504, -8456, -8440, -8424, -8408, -8408, -8416, -8392, -8392, -8360, -8392, -8368, -8392, -8376, -8368, -8384, -8376, -8392, -8392, -8392, -8392, -8408, -8440, -8440, -8448, -8480, -8504, -8488, -8528, -8560, -8560, -8592, -8640, -8664, -8672, -8696, -8720, -8744, -8816, -8824, -8864, -8888, -8928, -8976, -9008, -9064, -9104, -9128, -9208, -9232, -9296, -9336, -9392, -9464, -9496, -9560, -9616, -9680, -9736, -9776, -9856, -9896, -9984, -10024, -10080, -10152, -10224, -10280, -10352, -10440, -10512, -10552, -10648, -10688, -10784, -10856, -10920, -10984, -11072, -11136, -11224, -11304, -11384, -11440, -11536, -11624, -11696, -11768, -11840, -11920, -12032, -12128, -12208, -12288, -12408, -12472, -12568, -12640, -12752, -12824, -12912, -13000, -13088, -13184, -13288, -13384, -13488, -13576, -13680, -13776, -13856, -13968, -14040, -14144, -14264, -14344, -14432, -14544, -14640, -14752, -14848, -14928, -15016, -15136, -15232, -15304, -15448, -15544, -15664, -15744, -15824, -15944, -16032, -16160, -16256, -16352, -16448, -16552, -16648, -16768, -16856, -16976, -17056, -17184, -17280, -17392, -17496, -17608, -17704, -17784, -17880, -17976, -18096, -18216, -18304, -18416, -18512, -18640, -18712, -18816, -18912, -19016, -19136, -19232, -19344, -19432, -19544, -19640, -19736, -19824, -19928, -20000, -20136, -20240, -20328, -20416, -20520, -20592, -20680, -20832, -20896, -20992, -21096, -21192, -21296, -21392, -21496, -21600, -21656, -21760, -21872, -21952, -22048, -22152, -22232, -22328, -22400, -22496, -22584, -22680, -22768, -22872, -22936, -23024, -23128, -23192, -23312, -23384, -23472, -23560, -23632, -23712, -23792, -23872, -23976, -24040, -24112, -24176, -24320, -24368, -24464, -24552, -24608, -24664, -24784, -24840, -24888, -24992, -25072, -25152, -25240, -25288, -25352, -25424, -25496, -25584, -25656, -25752, -25816, -25888, -25944, -25984, -26048, -26152, -26224, -26280, -26336, -26400, -26488, -26544, -26584, -26656, -26720, -26760, -26864, -26904, -26952, -27032, -27096, -27128, -27208, -27240, -27304, -27368, -27424, -27480, -27520, -27600, -27640, -27720, -27760, -27808, -27848, -27912, -27944, -27992, -28080, -28096, -28168, -28208, -28232, -28320, -28360, -28392, -28416, -28496, -28544, -28576, -28632, -28672, -28720, -28736, -28808, -28840, -28880, -28912, -28960, -28960, -29040, -29048, -29120, -29136, -29192, -29200, -29240, -29272, -29296, -29312, -29368, -29416, -29448, -29480, -29520, -29552, -29592, -29608, -29648, -29664, -29720, -29712, -29752, -29808, -29816, -29856, -29896, -29912, -29920, -29968, -30008, -30032, -30072, -30080, -30104, -30128, -30152, -30176, -30224, -30224, -30240, -30264, -30264, -30320, -30320, -30360, -30360, -30392, -30424, -30408, -30464, -30456, -30504, -30496, -30520, -30544, -30544, -30576, -30560, -30600, -30640, -30632, -30656, -30664, -30688, -30680, -30720, -30736, -30768, -30800, -30776, -30776, -30800, -30816, -30832, -30848, -30872, -30848, -30872, -30912, -30928, -30936, -30912, -30968, -30952, -30944, -30952, -30968, -30992, -31008, -30984, -31016, -31008, -31032, -31040, -31048, -31080, -31080, -31056, -31088, -31088, -31104, -31120, -31112, -31120, -31152, -31128, -31168, -31144, -31160, -31168, -31168, -31176, -31176, -31184, -31208, -31208, -31232, -31232, -31232, -31240, -31216, -31224, -31248, -31264, -31280 } };
	struct fitData result[1];

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	struct fitData* d_result;
	cudaMalloc((void**)&d_result, sizeof(struct fitData) * countDatasets);
	cudaMemcpy(d_result, result, sizeof(struct fitData) * countDatasets, cudaMemcpyHostToDevice);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<DATATYPE>();
	cudaArray *dataArray;
	cudaMallocArray(&dataArray, &channelDesc, countData, countDatasets);
	cudaMemcpyToArray(dataArray, 0, 0, testData, sizeof(DATATYPE) * countData * countDatasets, cudaMemcpyHostToDevice);

	dataTexture0.normalized = 0;
	dataTexture0.filterMode = cudaFilterModeLinear;
	dataTexture0.addressMode[0] = cudaAddressModeClamp;
	dataTexture0.addressMode[1] = cudaAddressModeClamp;

	cudaBindTextureToArray(dataTexture0, dataArray);

	dim3 grid(countDatasets, 1, 1); //number of blocks (in all dimensions) = number of datasets
	//currently, the number of threads per block must be 1 (in the future used for simultaneous calculations in one dataset)
	
	cudaEventRecord(start, 0);
	kernel<0><<<grid, 1>>>(countData, 1, d_result);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cudaMemcpy(result, d_result, sizeof(struct fitData) * countDatasets, cudaMemcpyDeviceToHost);

	cudaUnbindTexture(dataTexture0);
	cudaFreeArray(dataArray);
	cudaFree(d_result);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("calculation time: %f ms\n", time);
#else
	int countData = 1000;
	int countDatasets = 1;
	//DATATYPE testData[5] = { 2, 4.3, 5, 4.3, 2 };
	DATATYPE testData[1000] = { -31576, -31560, -31552, -31544, -31560, -31576, -31552, -31560, -31552, -31576, -31560, -31576, -31568, -31576, -31552, -31560, -31568, -31552, -31552, -31576, -31552, -31576, -31568, -31576, -31552, -31552, -31536, -31536, -31536, -31552, -31568, -31560, -31544, -31568, -31552, -31560, -31552, -31528, -31568, -31560, -31544, -31568, -31560, -31576, -31576, -31560, -31552, -31552, -31568, -31568, -31584, -31560, -31568, -31560, -31576, -31552, -31576, -31560, -31592, -31568, -31568, -31568, -31552, -31560, -31544, -31552, -31576, -31560, -31560, -31552, -31568, -31536, -31560, -31560, -31560, -31544, -31544, -31576, -31568, -31544, -31528, -31552, -31568, -31552, -31560, -31552, -31560, -31552, -31560, -31552, -31568, -31544, -31552, -31544, -31544, -31568, -31544, -31560, -31536, -31560, -31568, -31568, -31552, -31560, -31544, -31560, -31552, -31568, -31560, -31576, -31568, -31576, -31544, -31568, -31560, -31528, -31560, -31544, -31544, -31560, -31552, -31544, -31520, -31560, -31552, -31552, -31568, -31544, -31544, -31552, -31544, -31552, -31544, -31568, -31560, -31560, -31568, -31544, -31552, -31552, -31560, -31536, -31560, -31552, -31568, -31552, -31552, -31568, -31568, -31552, -31552, -31544, -31568, -31584, -31560, -31552, -31536, -31544, -31536, -31552, -31552, -31552, -31552, -31512, -31560, -31536, -31544, -31544, -31536, -31552, -31560, -31552, -31552, -31536, -31544, -31560, -31568, -31560, -31552, -31552, -31568, -31560, -31568, -31592, -31576, -31544, -31568, -31560, -31568, -31552, -31544, -31576, -31568, -31544, -31552, -31560, -31576, -31552, -31568, -31560, -31560, -31544, -31552, -31568, -31552, -31552, -31536, -31544, -31560, -31552, -31544, -31544, -31544, -31528, -31544, -31560, -31576, -31568, -31576, -31552, -31544, -31544, -31560, -31552, -31536, -31576, -31552, -31560, -31560, -31560, -31560, -31560, -31568, -31568, -31544, -31584, -31560, -31568, -31544, -31552, -31568, -31536, -31536, -31520, -31544, -31536, -31552, -31552, -31552, -31544, -31536, -31512, -31504, -31536, -31528, -31528, -31528, -31512, -31504, -31520, -31496, -31520, -31488, -31504, -31512, -31496, -31488, -31496, -31496, -31496, -31504, -31472, -31480, -31456, -31456, -31472, -31472, -31472, -31456, -31456, -31424, -31440, -31424, -31432, -31376, -31408, -31384, -31384, -31368, -31368, -31336, -31336, -31320, -31304, -31312, -31280, -31248, -31256, -31264, -31216, -31248, -31200, -31192, -31128, -31136, -31144, -31120, -31096, -31088, -31080, -31048, -31016, -30992, -30968, -30976, -30952, -30920, -30904, -30872, -30856, -30840, -30800, -30768, -30728, -30704, -30680, -30648, -30648, -30600, -30560, -30528, -30496, -30456, -30416, -30368, -30336, -30304, -30264, -30232, -30192, -30128, -30104, -30048, -30000, -29960, -29904, -29872, -29816, -29760, -29720, -29656, -29632, -29576, -29488, -29440, -29400, -29344, -29288, -29232, -29176, -29112, -29048, -28984, -28896, -28872, -28824, -28728, -28680, -28568, -28504, -28464, -28368, -28312, -28248, -28176, -28088, -28016, -27936, -27848, -27792, -27696, -27624, -27528, -27456, -27368, -27272, -27192, -27112, -26992, -26920, -26816, -26720, -26648, -26536, -26448, -26344, -26280, -26168, -26080, -25992, -25872, -25776, -25680, -25576, -25480, -25368, -25240, -25168, -25048, -24944, -24824, -24720, -24600, -24520, -24408, -24280, -24176, -24096, -23936, -23832, -23704, -23616, -23488, -23408, -23256, -23144, -23000, -22912, -22776, -22632, -22544, -22400, -22272, -22160, -22056, -21904, -21776, -21664, -21512, -21408, -21288, -21144, -21056, -20888, -20784, -20656, -20536, -20416, -20280, -20144, -20032, -19896, -19776, -19640, -19512, -19392, -19240, -19144, -19008, -18864, -18720, -18624, -18488, -18344, -18248, -18088, -17976, -17840, -17704, -17576, -17464, -17328, -17200, -17064, -16944, -16832, -16688, -16552, -16448, -16312, -16176, -16072, -15944, -15832, -15704, -15576, -15448, -15352, -15200, -15080, -14976, -14816, -14736, -14616, -14512, -14384, -14280, -14176, -14056, -13936, -13808, -13696, -13608, -13480, -13376, -13256, -13160, -13056, -12936, -12816, -12744, -12632, -12528, -12416, -12336, -12208, -12136, -12048, -11928, -11824, -11752, -11640, -11536, -11456, -11384, -11288, -11176, -11104, -11016, -10952, -10864, -10776, -10688, -10616, -10544, -10480, -10392, -10304, -10264, -10176, -10080, -10032, -9944, -9896, -9840, -9776, -9704, -9656, -9592, -9528, -9480, -9424, -9360, -9288, -9272, -9192, -9152, -9096, -9072, -9016, -8984, -8952, -8912, -8880, -8816, -8792, -8760, -8688, -8672, -8640, -8632, -8584, -8576, -8560, -8512, -8512, -8504, -8456, -8440, -8424, -8408, -8408, -8416, -8392, -8392, -8360, -8392, -8368, -8392, -8376, -8368, -8384, -8376, -8392, -8392, -8392, -8392, -8408, -8440, -8440, -8448, -8480, -8504, -8488, -8528, -8560, -8560, -8592, -8640, -8664, -8672, -8696, -8720, -8744, -8816, -8824, -8864, -8888, -8928, -8976, -9008, -9064, -9104, -9128, -9208, -9232, -9296, -9336, -9392, -9464, -9496, -9560, -9616, -9680, -9736, -9776, -9856, -9896, -9984, -10024, -10080, -10152, -10224, -10280, -10352, -10440, -10512, -10552, -10648, -10688, -10784, -10856, -10920, -10984, -11072, -11136, -11224, -11304, -11384, -11440, -11536, -11624, -11696, -11768, -11840, -11920, -12032, -12128, -12208, -12288, -12408, -12472, -12568, -12640, -12752, -12824, -12912, -13000, -13088, -13184, -13288, -13384, -13488, -13576, -13680, -13776, -13856, -13968, -14040, -14144, -14264, -14344, -14432, -14544, -14640, -14752, -14848, -14928, -15016, -15136, -15232, -15304, -15448, -15544, -15664, -15744, -15824, -15944, -16032, -16160, -16256, -16352, -16448, -16552, -16648, -16768, -16856, -16976, -17056, -17184, -17280, -17392, -17496, -17608, -17704, -17784, -17880, -17976, -18096, -18216, -18304, -18416, -18512, -18640, -18712, -18816, -18912, -19016, -19136, -19232, -19344, -19432, -19544, -19640, -19736, -19824, -19928, -20000, -20136, -20240, -20328, -20416, -20520, -20592, -20680, -20832, -20896, -20992, -21096, -21192, -21296, -21392, -21496, -21600, -21656, -21760, -21872, -21952, -22048, -22152, -22232, -22328, -22400, -22496, -22584, -22680, -22768, -22872, -22936, -23024, -23128, -23192, -23312, -23384, -23472, -23560, -23632, -23712, -23792, -23872, -23976, -24040, -24112, -24176, -24320, -24368, -24464, -24552, -24608, -24664, -24784, -24840, -24888, -24992, -25072, -25152, -25240, -25288, -25352, -25424, -25496, -25584, -25656, -25752, -25816, -25888, -25944, -25984, -26048, -26152, -26224, -26280, -26336, -26400, -26488, -26544, -26584, -26656, -26720, -26760, -26864, -26904, -26952, -27032, -27096, -27128, -27208, -27240, -27304, -27368, -27424, -27480, -27520, -27600, -27640, -27720, -27760, -27808, -27848, -27912, -27944, -27992, -28080, -28096, -28168, -28208, -28232, -28320, -28360, -28392, -28416, -28496, -28544, -28576, -28632, -28672, -28720, -28736, -28808, -28840, -28880, -28912, -28960, -28960, -29040, -29048, -29120, -29136, -29192, -29200, -29240, -29272, -29296, -29312, -29368, -29416, -29448, -29480, -29520, -29552, -29592, -29608, -29648, -29664, -29720, -29712, -29752, -29808, -29816, -29856, -29896, -29912, -29920, -29968, -30008, -30032, -30072, -30080, -30104, -30128, -30152, -30176, -30224, -30224, -30240, -30264, -30264, -30320, -30320, -30360, -30360, -30392, -30424, -30408, -30464, -30456, -30504, -30496, -30520, -30544, -30544, -30576, -30560, -30600, -30640, -30632, -30656, -30664, -30688, -30680, -30720, -30736, -30768, -30800, -30776, -30776, -30800, -30816, -30832, -30848, -30872, -30848, -30872, -30912, -30928, -30936, -30912, -30968, -30952, -30944, -30952, -30968, -30992, -31008, -30984, -31016, -31008, -31032, -31040, -31048, -31080, -31080, -31056, -31088, -31088, -31104, -31120, -31112, -31120, -31152, -31128, -31168, -31144, -31160, -31168, -31168, -31176, -31176, -31184, -31208, -31208, -31232, -31232, -31232, -31240, -31216, -31224, -31248, -31264, -31280 };
	struct fitData result[1];

	data = testData;
	kernel<0>(countData, 1, result);
#endif

	for (int i = 0; i < countDatasets; i++) {
		printf("f: y = %f * t ^ 2 + %f * t + %f\n", result[i].param[0], result[i].param[1], result[i].param[2]);
		//printf("f: %f * e ^ (%f * (x + %f) ^ 2) + %f\n", result[i].param[0], result[i].param[1], result[i].param[2], result[i].param[3]);
		printf("min/max - x: %f\n", result[i].extremumPos);
		printf("min/max - y: %f\n", result[i].extremumValue);
		printf("start - y: %f\n", result[i].startValue);
		printf("end - y: %f\n", result[i].endValue);
		printf("euclidean norm: %f\n", result[i].euclidNormResidues);
		printf("average: %f\n", result[i].averageAbsResidues);
		printf("status: %d: %s\n", result[i].status, statusMessage[result[i].status]);
	}

	return 0;
}*/

#endif
