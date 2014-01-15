/*
TODO (see also Pflichtenheft.pdf)
 (1) data array -> texture memory
 (2) calculation of start and end values (maybe averaged)
 (3) maybe other fit-function* (e. g. a*e^(b*(x+c)^2)+d, b < 0) (before (2)) or trim data (after (2))
 (4) return a boolean (successful / not successful) in the result (no lm_infmsg)
 (5) maybe use shared memory with dynamic size (depending on countData and COUNTPARAM) instead of MAXCOUNTDATA
 (6) optimizing to use less memory -> higher MAXCOUNTDATA is possible
 (7) maybe kernel with sub-kernels (parallelization)

 * (3) Note: e-function with 4 parameters needs a good starting value, e. g. { 10000, -1, -500, -31000 } -> it would be necessary to check the data before

CHANGES (already done)
 - horrible goto-instructions replaced
 - reduced to one file with integrated fit-function, residue calculation etc.
 - without controlling
 - without user interaction
 - fit-function
 - example data for testing
 - reduced to uniform distribution of x-coord. -> 1 array (instead of 2)
 - extremum calculation
 - double -> DATATYPE (= short int) (pertains measured data)
 - cuda kernel etc.
 - return the fit function result
 
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


//--- USER DEFINITIONS ---

#define CUDA //defined: runs on GPU, otherwise on CPU (useful for debugging)

//#define MAXCOUNTDATA 2450 //for compute capability 2.0 or higher - currently ca. 2450 is max. because (COUNTPARAM + 2) * MAXCOUNTDATA * sizeof(float) = 48 kB (= max. shared memory)
#define MAXCOUNTDATA 800 //for compute capability 1.x - currently ca. 800 is max. because (COUNTPARAM + 2) * MAXCOUNTDATA * sizeof(float) = 16 kB (= max. shared memory)

#define DATATYPE short int
#define MAXCALL 100
#define COUNTPARAM 3
#define PARAMSTARTVALUE { 1, 1, 1 } //any value, but not { 0, 0, 0 } (count = COUNTPARAM)

//------------------------

#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

struct fitData {
	float param[COUNTPARAM];
	float startValue;
	float endValue;
	float extremumPos; //or replace to timestamp
	float extremumValue;
};

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

//following messages are indexed by the variable info
const char *lm_infmsg[] = {
	"fatal coding error (improper input parameters)",
	"success (the relative error in the sum of squares is at most tol)",
	"success (the relative error between x and the solution is at most tol)",
	"success (both errors are at most tol)",
	"trapped by degeneracy (fvec is orthogonal to the columns of the jacobian)"
	"timeout (number of calls to fcn has reached maxcall*(n+1))",
	"failure (ftol<tol: cannot reduce sum of squares any further)",
	"failure (xtol<tol: cannot improve approximate solution any further)",
	"failure (gtol<tol: cannot improve approximate solution any further)",
	"exception (not enough memory)"
};

#define MIN(a,b) (((a)<=(b)) ? (a) : (b))
#define MAX(a,b) (((a)>=(b)) ? (a) : (b))
#define SQR(x)   (x)*(x)

DEVICE void fitFunction(float x, float* param, float* y) //get y(x)
{
	*y = param[0] * x * x + param[1] * x + param[2];
	//*y = param[0] * exp(param[1] * (x + param[2]) * (x + param[2])) + param[3];
}

DEVICE void fitFunctionExtremum(float* param, float* x) //get x
{
	//f': y = 2 * param[0] * x + param[1] and y = 0
	if (param[0] == 0)
		*x = 0; //no Extremum
	else
		*x = -param[1] / (2 * param[0]);
}

DEVICE void lm_evaluate(float *param, int countData, float *fvec, DATATYPE *data)
{
	int i;
	float y;

	for (i = 0; i < countData; i++) {
		fitFunction(i, param, &y); //fitFunction(xdata[i], param, &y); //if i is not equivalent to x-coord. (add parameter xdata)
		fvec[i] = data[i] - y;
	}
}

DEVICE void lm_qrsolv(int n, float *r, int ldr, int *ipvt, float *diag,
			   float *qtb, float *x, float *sdiag, float *wa)
{
	int i, kk, j, k, nsing;
	float qtbpj, sum, temp;
	float _sin, _cos, _tan, _cot; /* local variables, not functions */

	/*** qrsolv: copy r and (q transpose)*b to preserve input and initialize s.
	in particular, save the diagonal elements of r in x. ***/

	for (j = 0; j < n; j++) {
		for (i = j; i < n; i++)
			r[j * ldr + i] = r[i * ldr + j];
		x[j] = r[j * ldr + j];
		wa[j] = qtb[j];
	}

	/*** qrsolv: eliminate the diagonal matrix d using a givens rotation. ***/

	for (j = 0; j < n; j++) {

		/*** qrsolv: prepare the row of d to be eliminated, locating the
		diagonal element using p from the qr factorization. ***/

		if (diag[ipvt[j]] != 0.)
		{
			for (k = j; k < n; k++)
				sdiag[k] = 0.;
			sdiag[j] = diag[ipvt[j]];

			/*** qrsolv: the transformations to eliminate the row of d modify only 
			a single element of (q transpose)*b beyond the first n, which is
			initially 0.. ***/

			qtbpj = 0.;
			for (k = j; k < n; k++) {

				/** determine a givens rotation which eliminates the
				appropriate element in the current row of d. **/

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

				/** compute the modified diagonal element of r and
				the modified element of ((q transpose)*b,0). **/

				r[kk] = _cos * r[kk] + _sin * sdiag[k];
				temp = _cos * wa[k] + _sin * qtbpj;
				qtbpj = -_sin * wa[k] + _cos * qtbpj;
				wa[k] = temp;

				/** accumulate the tranformation in the row of s. **/

				for (i = k + 1; i < n; i++) {
					temp = _cos * r[k * ldr + i] + _sin * sdiag[i];
					sdiag[i] = -_sin * r[k * ldr + i] + _cos * sdiag[i];
					r[k * ldr + i] = temp;
				}
			}
		}

		/** store the diagonal element of s and restore
		the corresponding diagonal element of r. **/

		sdiag[j] = r[j * ldr + j];
		r[j * ldr + j] = x[j];
	}

	/*** qrsolv: solve the triangular system for z. if the system is
	singular, then obtain a least squares solution. ***/

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

	/*** qrsolv: permute the components of z back to components of x. ***/

	for (j = 0; j < n; j++)
		x[ipvt[j]] = wa[j];
}

DEVICE void lm_enorm(int n, float *x, float* result)
{
	int i;
	float agiant, s1, s2, s3, xabs, x1max, x3max, temp;

	s1 = 0;
	s2 = 0;
	s3 = 0;
	x1max = 0;
	x3max = 0;
	agiant = LM_SQRT_GIANT / ((float) n);

	/** sum squares. **/
	for (i = 0; i < n; i++) {
		xabs = fabs(x[i]);
		if (xabs > LM_SQRT_DWARF && xabs < agiant) {
			/*  sum for intermediate components. */
			s2 += xabs * xabs;
			continue;
		}

		if (xabs > LM_SQRT_DWARF) {
			/*  sum for large components. */
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
		/*  sum for small components. */
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

	/** calculation of norm. **/

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

DEVICE void lm_lmpar(int n, float *r, int ldr, int *ipvt, float *diag,
			  float *qtb, float delta, float *par, float *x,
			  float *sdiag, float *wa1, float *wa2)
{
	int i, iter, j, nsing;
	float dxnorm, fp, fp_old, gnorm, parc, parl, paru;
	float sum, temp;

	/*** lmpar: compute and store in x the gauss-newton direction. if the
	jacobian is rank-deficient, obtain a least squares solution. ***/

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

	/*** lmpar: initialize the iteration counter, evaluate the function at the
	origin, and test for acceptance of the gauss-newton direction. ***/

	iter = 0;
	for (j = 0; j < n; j++)
		wa2[j] = diag[j] * x[j];
	lm_enorm(n, wa2, &dxnorm);
	fp = dxnorm - delta;
	if (fp <= 0.1 * delta) {
		*par = 0;
		return;
	}

	/*** lmpar: if the jacobian is not rank deficient, the newton
	step provides a lower bound, parl, for the 0. of
	the function. otherwise set this bound to 0.. ***/

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
		lm_enorm(n, wa1, &temp);
		parl = fp / delta / temp / temp;
	}

	/*** lmpar: calculate an upper bound, paru, for the 0. of the function. ***/

	for (j = 0; j < n; j++) {
		sum = 0;
		for (i = 0; i <= j; i++)
			sum += r[j * ldr + i] * qtb[i];
		wa1[j] = sum / diag[ipvt[j]];
	}
	lm_enorm(n, wa1, &gnorm);
	paru = gnorm / delta;
	if (paru == 0.)
		paru = LM_DWARF / MIN(delta, 0.1);

	/*** lmpar: if the input par lies outside of the interval (parl,paru),
	set par to the closer endpoint. ***/

	*par = MAX(*par, parl);
	*par = MIN(*par, paru);
	if (*par == 0.)
		*par = gnorm / dxnorm;

	/*** lmpar: iterate. ***/

	for (;; iter++) {

		/** evaluate the function at the current value of par. **/

		if (*par == 0.)
			*par = MAX(LM_DWARF, 0.001 * paru);
		temp = sqrt(*par);
		for (j = 0; j < n; j++)
			wa1[j] = temp * diag[j];
		lm_qrsolv(n, r, ldr, ipvt, wa1, qtb, x, sdiag, wa2);
		for (j = 0; j < n; j++)
			wa2[j] = diag[j] * x[j];
		lm_enorm(n, wa2, &dxnorm);
		fp_old = fp;
		fp = dxnorm - delta;

		/** if the function is small enough, accept the current value
		of par. Also test for the exceptional cases where parl
		is zero or the number of iterations has reached 10. **/

		if (fabs(fp) <= 0.1 * delta
			|| (parl == 0. && fp <= fp_old && fp_old < 0.)
			|| iter == 10)
			break; /* the only exit from the iteration. */

		/** compute the Newton correction. **/

		for (j = 0; j < n; j++)
			wa1[j] = diag[ipvt[j]] * wa2[ipvt[j]] / dxnorm;

		for (j = 0; j < n; j++) {
			wa1[j] = wa1[j] / sdiag[j];
			for (i = j + 1; i < n; i++)
				wa1[i] -= r[j * ldr + i] * wa1[j];
		}
		lm_enorm(n, wa1, &temp);
		parc = fp / delta / temp / temp;

		/** depending on the sign of the function, update parl or paru. **/

		if (fp > 0)
			parl = MAX(parl, *par);
		else if (fp < 0)
			paru = MIN(paru, *par);
		/* the case fp==0 is precluded by the break condition  */

		/** compute an improved estimate for par. **/

		*par = MAX(parl, *par + parc);
	}
}

DEVICE void lm_qrfac(int m, int n, float *a, int pivot, int *ipvt,
			  float *rdiag, float *acnorm, float *wa)
{
	int i, j, k, kmax, minmn;
	float ajnorm, sum, temp;

	/*** qrfac: compute initial column norms and initialize several arrays. ***/

	for (j = 0; j < n; j++) {
		lm_enorm(m, &a[j * m], &acnorm[j]);
		rdiag[j] = acnorm[j];
		wa[j] = rdiag[j];
		if (pivot)
			ipvt[j] = j;
	}

	/*** qrfac: reduce a to r with householder transformations. ***/

	minmn = MIN(m, n);
	for (j = 0; j < minmn; j++) {
		if (pivot)
		{
			/** bring the column of largest norm into the pivot position. **/

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

        /** pivot_ok **/

		/** compute the Householder transformation to reduce the
		j-th column of a to a multiple of the j-th unit vector. **/

		lm_enorm(m - j, &a[j * m + j], &ajnorm);
		if (ajnorm == 0.) {
			rdiag[j] = 0;
			continue;
		}

		if (a[j * m + j] < 0.)
			ajnorm = -ajnorm;
		for (i = j; i < m; i++)
			a[j * m + i] /= ajnorm;
		a[j * m + j] += 1;

		/** apply the transformation to the remaining columns
		and update the norms. **/

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
					lm_enorm(m - j - 1, &a[m * k + j + 1], &rdiag[k]);
					wa[k] = rdiag[k];
				}
			}
		}

		rdiag[j] = -ajnorm;
	}
}

DEVICE void lm_lmdif(int m, int n, float *x, float *fvec, float ftol,
			  float xtol, float gtol, int maxfev, float epsfcn,
			  float *diag, int mode, float factor, int *info, int *nfev,
			  float *fjac, int *ipvt, float *qtf, float *wa1,
			  float *wa2, float *wa3, float *wa4,
			  DATATYPE *data)
{
	int i, iter, j;
	float actred, delta, dirder, eps, fnorm, fnorm1, gnorm, par, pnorm,
		prered, ratio, step, sum, temp, temp1, temp2, temp3, xnorm;

	*nfev = 0;			/* function evaluation counter */
	iter = 1;			/* outer loop counter */
	par = 0;			/* levenberg-marquardt parameter */
	delta = 0;	 /* to prevent a warning (initialization within if-clause) */
	xnorm = 0;	 /* ditto */
	temp = MAX(epsfcn, LM_MACHEP);
	eps = sqrt(temp); /* for calculating the Jacobian by forward differences */

	/*** lmdif: check input parameters for errors. ***/

	if ((n <= 0) || (m < n) || (ftol < 0.)
		|| (xtol < 0.) || (gtol < 0.) || (maxfev <= 0) || (factor <= 0.)) {
			*info = 0;		// invalid parameter
			return;
	}
	if (mode == 2) {		/* scaling by diag[] */
		for (j = 0; j < n; j++) {	/* check for nonpositive elements */
			if (diag[j] <= 0.0) {
				*info = 0;	// invalid parameter
				return;
			}
		}
	}

	/*** lmdif: evaluate function at starting point and calculate norm. ***/

	*info = 0;
	lm_evaluate(x, m, fvec, data);
	++(*nfev);
	lm_enorm(m, fvec, &fnorm);

	/*** lmdif: the outer loop. ***/

	do {

		/*** outer: calculate the jacobian matrix. ***/

		for (j = 0; j < n; j++) {
			temp = x[j];
			step = eps * fabs(temp);
			if (step == 0.)
				step = eps;
			x[j] = temp + step;
			*info = 0;
			lm_evaluate(x, m, wa4, data);
			for (i = 0; i < m; i++) /* changed in 2.3, Mark Bydder */
				fjac[j * m + i] = (wa4[i] - fvec[i]) / (x[j] - temp);
			x[j] = temp;
		}

		/*** outer: compute the qr factorization of the jacobian. ***/

		lm_qrfac(m, n, fjac, 1, ipvt, wa1, wa2, wa3);

		if (iter == 1) { /* first iteration */
			if (mode != 2) {
				/* diag := norms of the columns of the initial jacobian */
				for (j = 0; j < n; j++) {
					diag[j] = wa2[j];
					if (wa2[j] == 0.)
						diag[j] = 1.;
				}
			}
			/* use diag to scale x, then calculate the norm */
			for (j = 0; j < n; j++)
				wa3[j] = diag[j] * x[j];
			lm_enorm(n, wa3, &xnorm);
			/* initialize the step bound delta. */
			delta = factor * xnorm;
			if (delta == 0.)
				delta = factor;
		}

		/*** outer: form (q transpose)*fvec and store first n components in qtf. ***/

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

		/** outer: compute norm of scaled gradient and test for convergence. ***/

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

		/*** outer: rescale if necessary. ***/

		if (mode != 2) {
			for (j = 0; j < n; j++)
				diag[j] = MAX(diag[j], wa2[j]);
		}

		/*** the inner loop. ***/
		do {
			/*** inner: determine the levenberg-marquardt parameter. ***/

			lm_lmpar(n, fjac, m, ipvt, diag, qtf, delta, &par,
				wa1, wa2, wa3, wa4);

			/*** inner: store the direction p and x + p; calculate the norm of p. ***/

			for (j = 0; j < n; j++) {
				wa1[j] = -wa1[j];
				wa2[j] = x[j] + wa1[j];
				wa3[j] = diag[j] * wa1[j];
			}
			lm_enorm(n, wa3, &pnorm);

			/*** inner: on the first iteration, adjust the initial step bound. ***/

			if (*nfev <= 1 + n)
				delta = MIN(delta, pnorm);

			/* evaluate the function at x + p and calculate its norm. */

			*info = 0;
			lm_evaluate(wa2, m, wa4, data);
			++(*nfev);

			lm_enorm(m, wa4, &fnorm1);

			/*** inner: compute the scaled actual reduction. ***/

			if (0.1 * fnorm1 < fnorm)
				actred = 1 - SQR(fnorm1 / fnorm);
			else
				actred = -1;

			/*** inner: compute the scaled predicted reduction and 
			the scaled directional derivative. ***/

			for (j = 0; j < n; j++) {
				wa3[j] = 0;
				for (i = 0; i <= j; i++)
					wa3[i] += fjac[j * m + i] * wa1[ipvt[j]];
			}
			lm_enorm(n, wa3, &temp1);
			temp1 /= fnorm;
			temp2 = sqrt(par) * pnorm / fnorm;
			prered = SQR(temp1) + 2 * SQR(temp2);
			dirder = -(SQR(temp1) + SQR(temp2));

			/*** inner: compute the ratio of the actual to the predicted reduction. ***/

			ratio = prered != 0 ? actred / prered : 0;

			/*** inner: update the step bound. ***/

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

			/*** inner: test for successful iteration. ***/

			if (ratio >= 0.0001) {
				/* yes, success: update x, fvec, and their norms. */
				for (j = 0; j < n; j++) {
					x[j] = wa2[j];
					wa2[j] = diag[j] * x[j];
				}
				for (i = 0; i < m; i++)
					fvec[i] = wa4[i];
				lm_enorm(n, wa2, &xnorm);
				fnorm = fnorm1;
				iter++;
			}

			/*** inner: tests for convergence ( otherwise *info = 1, 2, or 3 ). ***/

			*info = 0; /* do not terminate (unless overwritten by nonzero) */
			if (fabs(actred) <= ftol && prered <= ftol && 0.5 * ratio <= 1)
				*info = 1;
			if (delta <= xtol * xnorm)
				*info += 2;
			if (*info != 0)
				return;

			/*** inner: tests for termination and stringent tolerances. ***/

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

			/*** inner: end of the loop. repeat if iteration unsuccessful. ***/

		} while (ratio < 0.0001);

		/*** outer: end of the loop. ***/

	} while (1);

}

GLOBAL void kernel(int countData, DATATYPE *data, struct fitData *result)
{
	float param[COUNTPARAM] = PARAMSTARTVALUE;
	int nfev = 0, info = 0, i;

	//TODO IMPORTANT: check if only one thread per shared memory runs the following code
	SHARED float fvec[MAXCOUNTDATA];
	SHARED float fjac[COUNTPARAM * MAXCOUNTDATA];
	SHARED float wa4[MAXCOUNTDATA];

	float diag[COUNTPARAM], qtf[COUNTPARAM];
	float wa1[COUNTPARAM], wa2[COUNTPARAM], wa3[COUNTPARAM];
	int ipvt[COUNTPARAM];

	lm_lmdif(countData, COUNTPARAM, param, fvec, LM_USERTOL, LM_USERTOL, LM_USERTOL,
		MAXCALL * (COUNTPARAM + 1), LM_USERTOL, diag, 1, 100, &info,
		&nfev, fjac, ipvt, qtf, wa1, wa2, wa3, wa4, data);

	for (i = 0; i < COUNTPARAM; i++)
		result->param[i] = param[i];
	//result->startValue = ... coming soon, see TODO (2) and (3)
	//result->endValue = ...
	fitFunctionExtremum(param, &result->extremumPos);
	fitFunction(result->extremumPos, param, &result->extremumValue);
}


//only for testing
int main()
{
	int countData = 3;
	DATATYPE data[3] = { 2, 5, 2 }; //if index is equivalent to x-coord. (-> xdata[3] = { 0, 1, 2 }) then result should be: param[3] = { -3, 6, 2 }
	struct fitData result;
	
#ifdef CUDA
	DATATYPE* d_data;
	cudaMalloc((void**)&d_data, sizeof(float) * countData);
	cudaMemcpy(d_data, data, sizeof(float) * countData, cudaMemcpyHostToDevice);
	
	struct fitData* d_result;
	cudaMalloc((void**)&d_result, sizeof(struct fitData));
	cudaMemcpy(d_result, &result, sizeof(struct fitData), cudaMemcpyHostToDevice);
	
	kernel<<<1, 1>>>(countData, d_data, d_result);

	cudaMemcpy(&result, d_result, sizeof(struct fitData), cudaMemcpyDeviceToHost);

	cudaFree(d_data);
	cudaFree(d_result); 
#else
	kernel(countData, data, &result);
#endif

	printf("f: y = %f * t ^ 2 + %f * t + %f\n", result.param[0], result.param[1], result.param[2]);
	//printf("f: %f * e ^ (%f * (x + %f) ^ 2) + %f\n", result.param[0], result.param[1], result.param[2], result.param[3]);
	printf("min/max - x: %f\n", result.extremumPos);
	printf("min/max - y: %f\n", result.extremumValue);

	return 0;
}
