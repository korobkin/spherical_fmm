#ifndef CONFORMAL_SOLVER_H
#define CONFORMAL_SOLVER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CS_ERROR_NORM_LINF = 0,
    CS_ERROR_NORM_L2 = 1,
} cs_error_norm_t;

typedef struct {
    int iterations;
    int converged;
    double last_error;
} cs_convergence_info_t;

typedef struct {
    size_t n;
    /* Arrays owned by the caller; solver does not free them. */
    const double *r_xyz;   /* length 3*n: [x0,y0,z0, x1,y1,z1, ...] */
    const double *V;       /* length n */
    const double *vol;     /* length n */

    double M;
    double softening;      /* >= 0; used as sqrt(|r_i-r_j|^2 + softening^2) */
    int exclude_self;      /* if nonzero and softening==0, skip j==i */

    double W;              /* cached W = sum vol_a * V_a */
} conformal_solver_t;

/* Initialize a solver view over caller-owned arrays. Returns 0 on success. */
int cs_init(
    conformal_solver_t *s,
    size_t n,
    const double *r_xyz,
    const double *V,
    const double *vol,
    double M,
    double softening,
    int exclude_self
);

/* Default initial guess: tilde_chi[i] = (1 + 4 r^2 / M^2)^(-1/2). */
void cs_initial_guess_tilde_chi(const conformal_solver_t *s, double *tilde_chi_out);

/* Normalize: returns C and writes chi_out (length n). Returns 0 on success. */
int cs_normalize_chi(
    const conformal_solver_t *s,
    const double *tilde_chi,
    double *C_out,
    double *chi_out
);

/* Compute masses: m[i] = vol[i]*V[i]*(1+chi[i])/(4*pi). */
void cs_masses_from_chi(const conformal_solver_t *s, const double *chi, double *m_out);

/* Brute-force chi from masses at interpolation points:
   tilde_chi_next[i] = -(1/4*pi) * sum_j m[j] / |r_i - r_j|  (with options)
*/
void cs_chi_from_masses(const conformal_solver_t *s, const double *m, double *tilde_chi_next_out);

/* Solve steps 4-11. On success, writes psi_out (length n). */
int cs_solve(
    const conformal_solver_t *s,
    double tol,
    int max_iter,
    cs_error_norm_t norm,
    int verbose,
    double *psi_out,
    cs_convergence_info_t *info_out
);

/* Evaluate psi at arbitrary points r_eval (K points, packed xyz).
   Requires chi at interpolation points. */
void cs_psi_at_points(
    const conformal_solver_t *s,
    const double *chi,
    size_t k,
    const double *r_eval_xyz,
    double *psi_out
);

#ifdef __cplusplus
}
#endif

#endif /* CONFORMAL_SOLVER_H */
