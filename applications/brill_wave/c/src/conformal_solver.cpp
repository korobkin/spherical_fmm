#include "conformal_solver.hpp"

#include <cassert>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

//static double cs_sq(double x) { return x * x; }

//void conformal_solver_t::init(size_t n_, real M_, real softening_,
//							  bool exclude_self_) {
//	n = n_;
//	r_xyz.resize(n_);
//	V.resize(n);
//	vol.resize(n);
//
//	assert(n > 0);
//	assert(M > 0.0);
//	assert(softening > 0.0);
//
//	r_xyz.resize(n);
//	V.resize(n);
//	vol.resize(n);
//	M = M_;
//	softening = softening_;
//	exclude_self = exclude_self_ ? 1 : 0;
//}
//
//void cs_initial_guess_tilde_chi(const conformal_solver_t *s,
//								double *tilde_chi_out) {
//	const size_t n = s->n;
//	const double invM2 = 1.0 / (s->M * s->M);
//
//	for (size_t i = 0; i < n; ++i) {
//		const double x = s->r_xyz[3 * i + 0];
//		const double y = s->r_xyz[3 * i + 1];
//		const double z = s->r_xyz[3 * i + 2];
//		const double r2 = x * x + y * y + z * z;
//		tilde_chi_out[i] = 1.0 / sqrt(1.0 + 4.0 * r2 * invM2);
//	}
//}
//
//int cs_normalize_chi(const conformal_solver_t *s, const double *tilde_chi,
//					 double *C_out, double *chi_out) {
//	const size_t n = s->n;
//	double denom = 0.0;
//	for (size_t i = 0; i < n; ++i) {
//		denom += s->vol[i] * s->V[i] * tilde_chi[i];
//	}
//	if (denom == 0.0)
//		return 1;
//
//	const double C = -(4.0 * M_PI * s->M + s->W) / denom;
//	if (C_out)
//		*C_out = C;
//	for (size_t i = 0; i < n; ++i) {
//		chi_out[i] = C * tilde_chi[i];
//	}
//	return 0;
//}
//
//void cs_masses_from_chi(const conformal_solver_t *s, const double *chi,
//						double *m_out) {
//	const size_t n = s->n;
//	const double inv4pi = 1.0 / (4.0 * M_PI);
//	for (size_t i = 0; i < n; ++i) {
//		m_out[i] = s->vol[i] * s->V[i] * (1.0 + chi[i]) * inv4pi;
//	}
//}
//
//void cs_chi_from_masses(const conformal_solver_t *s, const double *m,
//						double *tilde_chi_next_out) {
//	const size_t n = s->n;
//	const double inv4pi = 1.0 / (4.0 * M_PI);
//	const double eps2 = s->softening * s->softening;
//
//	for (size_t i = 0; i < n; ++i) {
//		const double xi = s->r_xyz[3 * i + 0];
//		const double yi = s->r_xyz[3 * i + 1];
//		const double zi = s->r_xyz[3 * i + 2];
//
//		double sum = 0.0;
//		for (size_t j = 0; j < n; ++j) {
//			if (s->exclude_self && eps2 == 0.0 && j == i)
//				continue;
//
//			const double xj = s->r_xyz[3 * j + 0];
//			const double yj = s->r_xyz[3 * j + 1];
//			const double zj = s->r_xyz[3 * j + 2];
//
//			const double dx = xi - xj;
//			const double dy = yi - yj;
//			const double dz = zi - zj;
//			const double r2 = dx * dx + dy * dy + dz * dz + eps2;
//			sum += m[j] / sqrt(r2);
//		}
//
//		tilde_chi_next_out[i] = -inv4pi * sum;
//	}
//}
//
//static double cs_error_linf(size_t n, const double *a, const double *b) {
//	double m = 0.0;
//	for (size_t i = 0; i < n; ++i) {
//		const double d = fabs(a[i] - b[i]);
//		if (d > m)
//			m = d;
//	}
//	return m;
//}
//
//static double cs_error_l2(size_t n, const double *a, const double *b) {
//	double s2 = 0.0;
//	for (size_t i = 0; i < n; ++i) {
//		const double d = a[i] - b[i];
//		s2 += d * d;
//	}
//	return sqrt(s2 / (double)n);
//}
//
//int cs_solve(const conformal_solver_t *s, double tol, int max_iter,
//			 cs_error_norm_t norm, int verbose, double *psi_out,
//			 cs_convergence_info_t *info_out) {
//	if (!s || !psi_out || !info_out)
//		return 1;
//	if (!(tol > 0.0))
//		return 2;
//	if (max_iter <= 0)
//		return 3;
//
//	const size_t n = s->n;
//
//	double *tilde = (double *)malloc(n * sizeof(double));
//	double *tilde_next = (double *)malloc(n * sizeof(double));
//	double *chi = (double *)malloc(n * sizeof(double));
//	double *m = (double *)malloc(n * sizeof(double));
//
//	if (!tilde || !tilde_next || !chi || !m) {
//		free(tilde);
//		free(tilde_next);
//		free(chi);
//		free(m);
//		return 4;
//	}
//
//	cs_initial_guess_tilde_chi(s, tilde);
//
//	double last_err = INFINITY;
//	int converged = 0;
//	int iters = 0;
//
//	for (int iter = 0; iter < max_iter; ++iter) {
//		double C = 0.0;
//		if (cs_normalize_chi(s, tilde, &C, chi) != 0) {
//			free(tilde);
//			free(tilde_next);
//			free(chi);
//			free(m);
//			return 5;
//		}
//		cs_masses_from_chi(s, chi, m);
//		cs_chi_from_masses(s, m, tilde_next);
//
//		double err = 0.0;
//		if (norm == CS_ERROR_NORM_LINF) {
//			err = cs_error_linf(n, tilde_next, tilde);
//		} else {
//			err = cs_error_l2(n, tilde_next, tilde);
//		}
//
//		if (verbose) {
//			printf("iter %4d  err=%.3e\n", iter, err);
//		}
//
//		/* swap tilde buffers */
//		double *tmp = tilde;
//		tilde = tilde_next;
//		tilde_next = tmp;
//
//		last_err = err;
//		iters = iter + 1;
//		if (err < tol) {
//			converged = 1;
//			break;
//		}
//	}
//
//	/* final normalization for output psi */
//	if (cs_normalize_chi(s, tilde, NULL, chi) != 0) {
//		free(tilde);
//		free(tilde_next);
//		free(chi);
//		free(m);
//		return 6;
//	}
//	for (size_t i = 0; i < n; ++i) {
//		psi_out[i] = 1.0 + chi[i];
//	}
//
//	info_out->iterations = iters;
//	info_out->converged = converged;
//	info_out->last_error = last_err;
//
//	free(tilde);
//	free(tilde_next);
//	free(chi);
//	free(m);
//	return 0;
//}

//void cs_psi_at_points(const conformal_solver_t *s, const double *chi, size_t k,
//					  const double *r_eval_xyz, double *psi_out) {
//	const size_t n = s->n;
//	const double inv4pi = 1.0 / (4.0 * M_PI);
//	const double eps2 = s->softening * s->softening;
//
//	/* Precompute masses from chi */
//	double *m = (double *)malloc(n * sizeof(double));
//	if (!m)
//		return;
//	cs_masses_from_chi(s, chi, m);
//
//	for (size_t i = 0; i < k; ++i) {
//		const double xi = r_eval_xyz[3 * i + 0];
//		const double yi = r_eval_xyz[3 * i + 1];
//		const double zi = r_eval_xyz[3 * i + 2];
//
//		double sum = 0.0;
//		for (size_t j = 0; j < n; ++j) {
//			const double xj = s->r_xyz[3 * j + 0];
//			const double yj = s->r_xyz[3 * j + 1];
//			const double zj = s->r_xyz[3 * j + 2];
//			const double dx = xi - xj;
//			const double dy = yi - yj;
//			const double dz = zi - zj;
//			const double r2 = dx * dx + dy * dy + dz * dz + eps2;
//			sum += m[j] / sqrt(r2);
//		}
//		psi_out[i] = 1.0 - inv4pi * sum;
//	}
//
//	free(m);
//}
