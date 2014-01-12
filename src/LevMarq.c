/*TODO
 - double -> short int (pertains measured data)
 - main -> kernel
 - malloc -> gpu memory
 - array -> cuda array
 - calculation of start and end values (maybe averaged)
 - return the result
 - maybe kernel with sub-kernels (parallelization)
 - maybe other fit-function (e. g. e^(-x^2) with params) or trim data
 (see Pflichtenheft.pdf)
*/

/* CHANGES (in preparation for cuda)
 - horrible goto-instructions replaced
 - reduced to one file with integrated fit-function, residue calculation etc.
 - without controlling
 - without user interaction
 - fit-function
 - example data for testing
 - reduced to uniform distribution of x-coord. -> 1 array (instead of 2)
 - extremum calculation
 
 Note: Some original comments were not updated after code changes.
*/


/*
* Authors:  Burton S. Garbow, Kenneth E. Hillstrom, Jorge J. More
*           (lmdif and other routines from the public-domain library
*           netlib::minpack, Argonne National Laboratories, March 1980);
*           Steve Moshier (initial C translation);
*           Joachim Wuttke (conversion into C++ compatible ANSI style,
*           corrections, comments, wrappers, hosting).
* Original source: http://sourceforge.net/projects/lmfit/
*/

#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

/* machine-dependent constants from float.h */
#define LM_MACHEP     DBL_EPSILON   /* resolution of arithmetic */
#define LM_DWARF      DBL_MIN       /* smallest nonzero number */
#define LM_SQRT_DWARF sqrt(DBL_MIN) /* square should not underflow */
#define LM_SQRT_GIANT sqrt(DBL_MAX) /* square should not overflow */
#define LM_USERTOL    30*LM_MACHEP  /* users are recommened to require this */

// following messages are indexed by the variable info
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

double fitFunction(double x, double *param) //get y(x)
{
	return param[0] * x * x + param[1] * x + param[2];
}

double fitFunctionExtremum(double *param) //get x
{
	//f': y = 2 * param[0] * x + param[1] and y = 0
	if (param[0] == 0)
		return 0; //no Extremum
	else
		return -param[1] / (2 * param[0]);
}

void lm_evaluate(double *par, int m_dat, double *fvec, double *ydat)
{
	int i;

	for (i = 0; i < m_dat; i++)
		fvec[i] = ydat[i] - fitFunction(i, par);
		//fvec[i] = ydat[i] - fitFunction(xdat[i], par); //if i is not equivalent to x-coord.
}

void lm_qrsolv(int n, double *r, int ldr, int *ipvt, double *diag,
			   double *qtb, double *x, double *sdiag, double *wa)
{
	int i, kk, j, k, nsing;
	double qtbpj, sum, temp;
	double _sin, _cos, _tan, _cot; /* local variables, not functions */

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

} /*** lm_qrsolv. ***/

double lm_enorm(int n, double *x)
{
	int i;
	double agiant, s1, s2, s3, xabs, x1max, x3max, temp;

	s1 = 0;
	s2 = 0;
	s3 = 0;
	x1max = 0;
	x3max = 0;
	agiant = LM_SQRT_GIANT / ((double) n);

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
		return x1max * sqrt(s1 + (s2 / x1max) / x1max);
	if (s2 != 0) {
		if (s2 >= x3max)
			return sqrt(s2 * (1 + (x3max / s2) * (x3max * s3)));
		else
			return sqrt(x3max * ((s2 / x3max) + (x3max * s3)));
	}

	return x3max * sqrt(s3);

} /*** lm_enorm. ***/

void lm_lmpar(int n, double *r, int ldr, int *ipvt, double *diag,
			  double *qtb, double delta, double *par, double *x,
			  double *sdiag, double *wa1, double *wa2)
{
	int i, iter, j, nsing;
	double dxnorm, fp, fp_old, gnorm, parc, parl, paru;
	double sum, temp;
	static double p1 = 0.1;

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
	dxnorm = lm_enorm(n, wa2);
	fp = dxnorm - delta;
	if (fp <= p1 * delta) {
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
		temp = lm_enorm(n, wa1);
		parl = fp / delta / temp / temp;
	}

	/*** lmpar: calculate an upper bound, paru, for the 0. of the function. ***/

	for (j = 0; j < n; j++) {
		sum = 0;
		for (i = 0; i <= j; i++)
			sum += r[j * ldr + i] * qtb[i];
		wa1[j] = sum / diag[ipvt[j]];
	}
	gnorm = lm_enorm(n, wa1);
	paru = gnorm / delta;
	if (paru == 0.)
		paru = LM_DWARF / MIN(delta, p1);

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
		dxnorm = lm_enorm(n, wa2);
		fp_old = fp;
		fp = dxnorm - delta;

		/** if the function is small enough, accept the current value
		of par. Also test for the exceptional cases where parl
		is zero or the number of iterations has reached 10. **/

		if (fabs(fp) <= p1 * delta
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
		temp = lm_enorm(n, wa1);
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

} /*** lm_lmpar. ***/

void lm_qrfac(int m, int n, double *a, int pivot, int *ipvt,
			  double *rdiag, double *acnorm, double *wa)
{
	int i, j, k, kmax, minmn;
	double ajnorm, sum, temp;
	static double p05 = 0.05;

	/*** qrfac: compute initial column norms and initialize several arrays. ***/

	for (j = 0; j < n; j++) {
		acnorm[j] = lm_enorm(m, &a[j * m]);
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

		ajnorm = lm_enorm(m - j, &a[j * m + j]);
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
				if (p05 * SQR(temp) <= LM_MACHEP) {
					rdiag[k] = lm_enorm(m - j - 1, &a[m * k + j + 1]);
					wa[k] = rdiag[k];
				}
			}
		}

		rdiag[j] = -ajnorm;
	}
}

void lm_lmdif(int m, int n, double *x, double *fvec, double ftol,
			  double xtol, double gtol, int maxfev, double epsfcn,
			  double *diag, int mode, double factor, int *info, int *nfev,
			  double *fjac, int *ipvt, double *qtf, double *wa1,
			  double *wa2, double *wa3, double *wa4,
			  double *ydat)
{
	int i, iter, j;
	double actred, delta, dirder, eps, fnorm, fnorm1, gnorm, par, pnorm,
		prered, ratio, step, sum, temp, temp1, temp2, temp3, xnorm;
	static double p1 = 0.1;
	static double p0001 = 1.0e-4;

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
	lm_evaluate(x, m, fvec, ydat);
	++(*nfev);
	fnorm = lm_enorm(m, fvec);

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
			lm_evaluate(x, m, wa4, ydat);
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
			xnorm = lm_enorm(n, wa3);
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
			pnorm = lm_enorm(n, wa3);

			/*** inner: on the first iteration, adjust the initial step bound. ***/

			if (*nfev <= 1 + n)
				delta = MIN(delta, pnorm);

			/* evaluate the function at x + p and calculate its norm. */

			*info = 0;
			lm_evaluate(wa2, m, wa4, ydat);
			++(*nfev);

			fnorm1 = lm_enorm(m, wa4);

			/*** inner: compute the scaled actual reduction. ***/

			if (p1 * fnorm1 < fnorm)
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
			temp1 = lm_enorm(n, wa3) / fnorm;
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
				if (p1 * fnorm1 >= fnorm || temp < p1)
					temp = p1;
				delta = temp * MIN(delta, pnorm / p1);
				par /= temp;
			} else if (par == 0. || ratio >= 0.75) {
				delta = pnorm / 0.5;
				par *= 0.5;
			}

			/*** inner: test for successful iteration. ***/

			if (ratio >= p0001) {
				/* yes, success: update x, fvec, and their norms. */
				for (j = 0; j < n; j++) {
					x[j] = wa2[j];
					wa2[j] = diag[j] * x[j];
				}
				for (i = 0; i < m; i++)
					fvec[i] = wa4[i];
				xnorm = lm_enorm(n, wa2);
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

		} while (ratio < p0001);

		/*** outer: end of the loop. ***/

	} while (1);

} /*** lm_lmdif. ***/

void lm_minimize(int countDat, int countParam, double *param, double *ydat, int maxcall)
{

	/*** allocate work space. ***/

	double *fvec, *diag, *fjac, *qtf, *wa1, *wa2, *wa3, *wa4;
	int *ipvt;

	int nfev = 0, info = 0;

	if ((fvec = (double *) malloc(countDat * sizeof(double))) == NULL ||
		(diag = (double *) malloc(countParam * sizeof(double))) == NULL ||
		(qtf  = (double *) malloc(countParam * sizeof(double))) == NULL ||
		(fjac = (double *) malloc(countParam * countDat * sizeof(double))) == NULL ||
		(wa1  = (double *) malloc(countParam * sizeof(double))) == NULL ||
		(wa2  = (double *) malloc(countParam * sizeof(double))) == NULL ||
		(wa3  = (double *) malloc(countParam * sizeof(double))) == NULL ||
		(wa4  = (double *) malloc(countDat * sizeof(double))) == NULL ||
		(ipvt = (int *)    malloc(countParam * sizeof(int))) == NULL) {
			info = 9;
			return;
	}

	/*** perform fit. ***/

	/* this goes through the modified legacy interface: */
	lm_lmdif(countDat, countParam, param, fvec, LM_USERTOL, LM_USERTOL, LM_USERTOL,
		maxcall * (countParam + 1), LM_USERTOL, diag, 1, 100, &info,
		&nfev, fjac, ipvt, qtf, wa1, wa2, wa3, wa4, ydat);

	//double fnorm = lm_enorm(countDat, fvec);

	free(fvec);
	free(diag);
	free(qtf);
	free(fjac);
	free(wa1);
	free(wa2);
	free(wa3);
	free(wa4);
	free(ipvt);
} /*** lm_minimize. ***/

int main()
{
	int countDat = 3;
	int countParam = 3;
	double y[3] = { 2, 5, 2 }; //countDat
	//if index is equivalent to x-coord. (x[3] = { 0, 1, 2 }) then result: param[3] = { -3, 6, 2 }
	//if not then use
	// double x[3] = { -1, 1, 2 }; //countDat
	// double y[3] = { -7, 5, 2 }; //countDat
	//and add x to lm_minimize, ...
	double param[3] = { 1, 1, 1 }; //countParam //starting value (any value, but not { 0, 0 ,0 } (?))

	lm_minimize(countDat, countParam, param, y, 100);

	printf("f: y = %f * t^2 + %f * t + %f\n", param[0], param[1], param[2]);

	printf("min/max - x: %f\n", fitFunctionExtremum(param));
	printf("min/max - y: %f\n", fitFunction(fitFunctionExtremum(param), param));

	return 0;
}
