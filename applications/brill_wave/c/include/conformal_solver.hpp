#ifndef CONFORMAL_SOLVER_H
#define CONFORMAL_SOLVER_H

#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stddef.h>
#include <type_traits>
#include <utility>
#include <vector>

#define NDIM 3

using real = double;

typedef enum {
	CS_ERROR_NORM_LINF = 0,
	CS_ERROR_NORM_L2 = 1,
} cs_error_norm_t;

struct cs_convergence_info_t {
	int iterations;
	int converged;
	double last_error;
};

template <typename T>
inline constexpr T inv(T x) {
	return T(1.0) / x;
}

template <typename T>
inline constexpr T sqr(T x) {
	return x * x;
}

struct conformal_solver_t {
	conformal_solver_t(int const n_, double L, double M_, double sigma, double eps, double soft) {
		n = n_;
		M = M_;
		Ngrid = n * n * n;
		r_xyz.resize(Ngrid);
		V.resize(Ngrid);
		vol.resize(Ngrid);

		const double dx = (2.0 * L) / (double)(n - 1);
		const double vol_cell = dx * dx * dx;
		const double inv2sig2 = 1.0 / (2.0 * sigma * sigma);
		int idx = 0;
		for (int i = 0; i < n; ++i) {
			const double x = -L + dx * (double)i;
			for (int j = 0; j < n; ++j) {
				const double y = -L + dx * (double)j;
				for (int k = 0; k < n; ++k) {
					const double z = -L + dx * (double)k;
					r_xyz[idx][0] = x;
					r_xyz[idx][1] = y;
					r_xyz[idx][2] = z;
					const double r2 = x * x + y * y + z * z;
					V[idx] = exp(-r2 * inv2sig2);
					vol[idx] = vol_cell;
					idx++;
				}
			}
		}
		Nkeep = 0;
		for (int i = 0; i < Ngrid; ++i) {
			if (V[i] > eps) Nkeep++;
		}

		if (std::isnan(soft)) soft = 0.5 * dx;
		softening = soft;
	}
	auto getPosition(int i) const {
		return r_xyz[i];
	}
	real getW() const {
		return std::inner_product(vol.begin(), vol.end(), vol.begin(), real(0));
	}
	auto normalize_chi(std::vector<real> const &tilde_chi) const {
		std::vector<real> chi_out(Ngrid);
		real denom = real(0);
		for (size_t i = 0; i < Ngrid; ++i) {
			denom += vol[i] * V[i] * tilde_chi[i];
		}
		assert(denom > 0.0);
		const double C = -(4.0 * M_PI * M + getW()) / denom;
		for (size_t i = 0; i < Ngrid; ++i) {
			chi_out[i] = C * tilde_chi[i];
		}
		return std::pair(C, chi_out);
	}
	std::vector<real> masses_from_chi(const std::vector<real> &chi) const {
		std::vector<real> m_out(Ngrid);
		const double inv4pi = inv(4.0 * M_PI);
		for (size_t i = 0; i < Ngrid; ++i) {
			m_out[i] = vol[i] * V[i] * (real(1) + chi[i]) * inv4pi;
		}
		return m_out;
	}
	std::vector<real> chi_from_masses(const std::vector<real> &m) const {
		std::vector<real> tilde_chi_next_out(Ngrid);
		const real inv4pi = 1.0 / (4.0 * M_PI);
		const real eps2 = softening * softening;

		for (size_t i = 0; i < Ngrid; ++i) {
			const auto &xyzi = r_xyz[i];

			real sum = 0.0;
			for (size_t j = 0; j < Ngrid; ++j) {
				if (exclude_self && eps2 == 0.0 && j == i) continue;

				const auto &xyzj = r_xyz[j];
				auto const r2 =
					eps2 + std::transform_reduce(xyzi.begin(), xyzi.end(), xyzj.begin(), real(0), std::plus<real>{}, [](real x, real y) {
						return sqr(x - y);
					});
				sum += m[j] * inv(std::sqrt(r2));
			}

			tilde_chi_next_out[i] = -inv4pi * sum;
		}
		return tilde_chi_next_out;
	}
	auto solve(real tol, int max_iter, cs_error_norm_t norm, int verbose) const {
		std::vector<real> psi_out(Ngrid);
		cs_convergence_info_t info_out;
		assert(tol > 0.0);
		assert(max_iter > 0);

		std::vector<real> tilde(Ngrid);
		const real invM2 = inv(sqr(M));
		for (size_t i = 0; i < Ngrid; ++i) {
			auto const &xyz = r_xyz[i];
			auto const r2 = std::inner_product(xyz.begin(), xyz.end(), xyz.begin(), real(0));
			tilde[i] = 1.0 / std::sqrt(1.0 + 4.0 * r2 * invM2);
		}

		real last_err = INFINITY;
		int converged = 0;
		int iters = 0;
		for (int iter = 0; iter < max_iter; ++iter) {
			auto const [C, chi] = normalize_chi(tilde);
			auto m = masses_from_chi(chi);
			auto tilde_next = chi_from_masses(m);

			real err = 0.0;
			if (norm == CS_ERROR_NORM_LINF) {
				err = std::transform_reduce(
					tilde_next.begin(), tilde_next.end(), tilde.begin(), real(0),
					[](real a, real b) {
						return std::max(a, b);
					},
					[](real x, real y) {
						return std::abs(x - y);
					});
			} else {
				err = std::sqrt(std::transform_reduce(tilde_next.begin(), tilde_next.end(), tilde.begin(), real(0), std::plus<real>{},
													  [](real x, real y) {
														  return sqr(x - y);
													  }) /
								real(Ngrid));
			}

			if (verbose) {
				printf("iter %4d  err=%.3e\n", iter, err);
			}

			/* swap tilde buffers */
			std::swap(tilde, tilde_next);

			last_err = err;
			iters = iter + 1;
			if (err < tol) {
				converged = 1;
				break;
			}
		}

		/* final normalization for output psi */
		auto const [_, chi] = normalize_chi(tilde);
		for (size_t i = 0; i < Ngrid; ++i) {
			psi_out[i] = 1.0 + chi[i];
		}

		info_out.iterations = iters;
		info_out.converged = converged;
		info_out.last_error = last_err;

		return std::pair(psi_out, info_out);
	}
	int getNkeep() const {
		return Nkeep;
	}
	/* Evaluate psi(r) = 1 - (1/4π) Σ_j m_j / |r - r_j| at arbitrary points,
	   mirroring ConformalFactorSolver.psi_at_points in the Python reference. */
	std::vector<real> psi_at_points(const std::vector<std::array<real, NDIM>> &r_eval, const std::vector<real> &chi) const {
		const size_t K = r_eval.size();
		std::vector<real> psi_out(K);
		auto m = masses_from_chi(chi);
		const real inv4pi = 1.0 / (4.0 * M_PI);
		const real eps2 = softening * softening;
		for (size_t i = 0; i < K; ++i) {
			const auto &xyzi = r_eval[i];
			real sum = 0.0;
			for (size_t j = 0; j < Ngrid; ++j) {
				const auto &xyzj = r_xyz[j];
				real r2 = eps2;
				for (int d = 0; d < NDIM; ++d) {
					const real dd = xyzi[d] - xyzj[d];
					r2 += dd * dd;
				}
				sum += m[j] / std::sqrt(r2);
			}
			psi_out[i] = 1.0 - inv4pi * sum;
		}
		return psi_out;
	}

private:
	size_t n;
	size_t Ngrid;
	int Nkeep;
	/* Arrays owned by the caller; solver does not free them. */
	std::vector<std::array<real, NDIM>> r_xyz; /* length 3*n: [x0,y0,z0, x1,y1,z1, ...] */
	std::vector<real> V;					   /* length n */
	std::vector<real> vol;					   /* length n */
	real M;
	real softening;	   /* >= 0; used as sqrt(|r_i-r_j|^2 + softening^2) */
	bool exclude_self; /* if nonzero and softening==0, skip j==i */
};

/* Initialize a solver view over caller-owned arrays. Returns 0 on success. */
int cs_init(conformal_solver_t *s, size_t n, const double *r_xyz, const double *V, const double *vol, double M, double softening,
			int exclude_self);

/* Default initial guess: tilde_chi[i] = (1 + 4 r^2 / M^2)^(-1/2). */
void cs_initial_guess_tilde_chi(const conformal_solver_t *s, double *tilde_chi_out);

/* Normalize: returns C and writes chi_out (length n). Returns 0 on success. */
int cs_normalize_chi(const conformal_solver_t *s, const double *tilde_chi, double *C_out, double *chi_out);

/* Compute masses: m[i] = vol[i]*V[i]*(1+chi[i])/(4*pi). */
void cs_masses_from_chi(const conformal_solver_t *s, const double *chi, double *m_out);

/* Brute-force chi from masses at interpolation points:
   tilde_chi_next[i] = -(1/4*pi) * sum_j m[j] / |r_i - r_j|  (with options)
*/
void cs_chi_from_masses(const conformal_solver_t *s, const double *m, double *tilde_chi_next_out);

/* Solve steps 4-11. On success, writes psi_out (length n). */
int cs_solve(const conformal_solver_t *s, double tol, int max_iter, cs_error_norm_t norm, int verbose, double *psi_out,
			 cs_convergence_info_t *info_out);

/* Evaluate psi at arbitrary points r_eval (K points, packed xyz).
   Requires chi at interpolation points. */
void cs_psi_at_points(const conformal_solver_t *s, const double *chi, size_t k, const double *r_eval_xyz, double *psi_out);

#endif /* CONFORMAL_SOLVER_H */
