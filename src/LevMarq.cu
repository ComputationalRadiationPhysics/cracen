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

#include "LevMarq.h"

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

DEVICE void evaluate(float *param, int countData, float *fvec, int indexDataset, int xOffset)
{
	int i;
	float y;

	for (i = 0; i < countData; i++) {
		fitFunction(i + xOffset, param, &y);
		fvec[i] = GETSAMPLE(i, indexDataset) - y;
	}
}

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

DEVICE void lmdif(int m, int n, float *x, float *fvec, float ftol,
			  float xtol, float gtol, int maxfev, float epsfcn,
			  float *diag, int mode, float factor, int *info, int *nfev,
			  float *fjac, int *ipvt, float *qtf, float *wa1,
			  float *wa2, float *wa3, float *wa4,
			  int indexDataset, int xOffset)
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
	evaluate(x, m, fvec, indexDataset, xOffset);
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
			evaluate(x, m, wa4, indexDataset, xOffset);
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
			evaluate(wa2, m, wa4, indexDataset, xOffset);
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
DEVICE void maxValue(int countData, int indexDataset, int *x, DATATYPE *y)
{
	int i;
	*x = 0;
	*y = GETSAMPLE(0, indexDataset);
	for (i = 0; i < countData; i++)
		if (GETSAMPLE(i, indexDataset) > *y) {
			*y = GETSAMPLE(i, indexDataset);
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
DEVICE void averageValue(int start, int count, int indexDataset, float *y)
{
	int i;
	float sum = 0;

	for (i = start; i < start + count; i++)
		sum += GETSAMPLE(i, indexDataset);
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
DEVICE void xOfValue(int countData, int indexDataset, char fromDirection, DATATYPE minValue, int *x)
{
	int i;
	*x = -1;
	if (fromDirection == 'l') {
		for (i = 0; i < countData; i++)
			if (GETSAMPLE(i, indexDataset) >= minValue) {
				*x = i;
				break;
			}
	}
	else if (fromDirection == 'r')
		for (i = countData - 1; i >= 0; i--)
			if (GETSAMPLE(i, indexDataset) >= minValue) {
				*x = i;
				break;
			}
}

/*!
 * \brief kernel is the start method for calculation (you have to set the dataTexture (GPU mode) or data variable (CPU mode) before calling this method)
 * \param countData number of samples
 * \param result fit-function and other parameters, defined in fitData struct
*/
GLOBAL void kernel(int countData, struct fitData *result)
{
#ifdef CUDA
	int indexDataset = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

	if (threadIdx.x > 0 || threadIdx.y > 0 || threadIdx.z > 0)
		return; //currently, the number of threads per block must be 1 (in the future used for simultaneous calculations in one dataset)
#else
	int indexDataset = 0;
#endif

	float param[COUNTPARAM] = PARAMSTARTVALUE;
	int nfev = 0, info = 0, i;

	int maxX, firstValue, lastValue, countAverage;
	DATATYPE maxY;

	SHARED float fvec[MAXCOUNTDATA];
	SHARED float fjac[COUNTPARAM * MAXCOUNTDATA];
	SHARED float wa4[MAXCOUNTDATA];

	float diag[COUNTPARAM], qtf[COUNTPARAM];
	float wa1[COUNTPARAM], wa2[COUNTPARAM], wa3[COUNTPARAM];
	int ipvt[COUNTPARAM];

	maxValue(countData, indexDataset, &maxX, &maxY);
	xOfValue(countData, indexDataset, 'l', (maxY - GETSAMPLE(0, indexDataset)) * FITVALUETHRESHOLD + GETSAMPLE(0, indexDataset), &firstValue);
	xOfValue(countData, indexDataset, 'r', (maxY - GETSAMPLE(countData - 1, indexDataset)) * FITVALUETHRESHOLD + GETSAMPLE(countData - 1, indexDataset), &lastValue);

	lmdif(lastValue - firstValue + 1, COUNTPARAM, param, fvec, LM_USERTOL, LM_USERTOL, LM_USERTOL,
		MAXCALL * (COUNTPARAM + 1), LM_USERTOL, diag, 1, 100, &info,
		&nfev, fjac, ipvt, qtf, wa1, wa2, wa3, wa4, indexDataset, firstValue);

	for (i = 0; i < COUNTPARAM; i++)
		result[indexDataset].param[i] = param[i];
	countAverage = countData * STARTENDPROPORTION;
	if (countData > 0 && countAverage == 0)
		countAverage = 1;
	averageValue(0, countAverage, indexDataset, &result[indexDataset].startValue);
	averageValue(countData - countAverage, countAverage, indexDataset, &result[indexDataset].endValue);
	fitFunctionExtremum(param, &result[indexDataset].extremumPos);
	fitFunction(result[indexDataset].extremumPos, param, &result[indexDataset].extremumValue);
	result[indexDataset].status = info;
}

/*
//example data, only for testing
int main()
{
#ifdef CUDA
	int countData = 3;
	int countDatasets = 2;
	DATATYPE testData[2][3] = { { 2, 5, 2 }, { 1, 4, 1 } };
	struct fitData result[2];

	struct fitData* d_result;
	cudaMalloc((void**)&d_result, sizeof(struct fitData) * countDatasets);
	cudaMemcpy(d_result, result, sizeof(struct fitData) * countDatasets, cudaMemcpyHostToDevice);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<DATATYPE>();
	cudaArray *dataArray;
	cudaMallocArray(&dataArray, &channelDesc, countData, countDatasets);
	cudaMemcpyToArray(dataArray, 0, 0, testData, sizeof(DATATYPE) * countData * countDatasets, cudaMemcpyHostToDevice);

	dataTexture.normalized = 0;
	dataTexture.filterMode = cudaFilterModeLinear;
	dataTexture.addressMode[0] = cudaAddressModeClamp;
	dataTexture.addressMode[1] = cudaAddressModeClamp;

	cudaBindTextureToArray(dataTexture, dataArray);

	dim3 grid(countDatasets, 1, 1); //number of blocks (in all dimensions) = number of datasets
	//currently, the number of threads per block must be 1 (in the future used for simultaneous calculations in one dataset)
	kernel<<<grid, 1>>>(countData, d_result);

	cudaMemcpy(result, d_result, sizeof(struct fitData) * countDatasets, cudaMemcpyDeviceToHost);

	cudaUnbindTexture(dataTexture);
	cudaFreeArray(dataArray);
	cudaFree(d_result);
#else
	int countData = 3;
	int countDatasets = 1;
	DATATYPE testData[3] = { 2, 5, 2 };
	struct fitData result[1];

	data = testData;
	kernel(countData, result);
#endif

	for (int i = 0; i < countDatasets; i++) {
		printf("f: y = %f * t ^ 2 + %f * t + %f\n", result[i].param[0], result[i].param[1], result[i].param[2]);
		//printf("f: %f * e ^ (%f * (x + %f) ^ 2) + %f\n", result[i].param[0], result[i].param[1], result[i].param[2], result[i].param[3]);
		printf("min/max - x: %f\n", result[i].extremumPos);
		printf("min/max - y: %f\n", result[i].extremumValue);
		printf("start - y: %f\n", result[i].startValue);
		printf("end - y: %f\n", result[i].endValue);
		printf("status: %d: %s\n", result[i].status, statusMessage[result[i].status]);
	}

	return 0;
}*/
