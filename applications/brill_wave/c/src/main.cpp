#include "conformal_solver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

static void usage(const char *prog) {
    fprintf(
        stderr,
        "Usage: %s [options]\n\n"
        "Demo: builds a uniform grid, a Gaussian V(r), filters V>eps, then runs the\n"
        "conformal-factor iteration with brute-force summation.\n\n"
        "Options:\n"
        "  --n <int>         grid points per axis (default 14)\n"
        "  --L <double>      half-size of box [-L, L] (default 2.0)\n"
        "  --sigma <double>  Gaussian sigma (default 0.5)\n"
        "  --eps <double>    drop points where V<=eps (default 1e-6)\n"
        "  --M <double>      parameter M in initial guess and normalization (default 1.0)\n"
        "  --tol <double>    convergence tolerance (default 1e-11)\n"
        "  --max_iter <int>  max iterations (default 200)\n"
        "  --soft <double>   softening length (default dx/2)\n"
        "  --out <file>      write (x y z psi) to file (default output.dat)\n"
        "  --no_self         do NOT exclude self-interactions (default excludes if soft==0)\n"
        "  --l2              use L2 error norm (default Linf)\n"
        "  --quiet           no per-iteration output\n",
        prog
    );
}

static int parse_int(const char *s, int *out) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!end || *end != '\0') return 1;
    *out = (int)v;
    return 0;
}

static int parse_double(const char *s, double *out) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (!end || *end != '\0') return 1;
    *out = v;
    return 0;
}

int main(int argc, char **argv) {
    int n = 14;
    double L = 2.0;
    double sigma = 0.5;
    double eps = 1e-6;
    double M = 1.0;
    double tol = 1e-11;
    int max_iter = 200;
    double soft = NAN; /* default: dx/2 */
    int exclude_self = 1;
    cs_error_norm_t norm = CS_ERROR_NORM_LINF;
    int verbose = 1;
    const char *out_path = "output.dat";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &n)) return 2;
        } else if (strcmp(argv[i], "--L") == 0 && i + 1 < argc) {
            if (parse_double(argv[++i], &L)) return 2;
        } else if (strcmp(argv[i], "--sigma") == 0 && i + 1 < argc) {
            if (parse_double(argv[++i], &sigma)) return 2;
        } else if (strcmp(argv[i], "--eps") == 0 && i + 1 < argc) {
            if (parse_double(argv[++i], &eps)) return 2;
        } else if (strcmp(argv[i], "--M") == 0 && i + 1 < argc) {
            if (parse_double(argv[++i], &M)) return 2;
        } else if (strcmp(argv[i], "--tol") == 0 && i + 1 < argc) {
            if (parse_double(argv[++i], &tol)) return 2;
        } else if (strcmp(argv[i], "--max_iter") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &max_iter)) return 2;
        } else if (strcmp(argv[i], "--soft") == 0 && i + 1 < argc) {
            if (parse_double(argv[++i], &soft)) return 2;
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--no_self") == 0) {
            exclude_self = 0;
        } else if (strcmp(argv[i], "--l2") == 0) {
            norm = CS_ERROR_NORM_L2;
        } else if (strcmp(argv[i], "--quiet") == 0) {
            verbose = 0;
        } else {
            fprintf(stderr, "Unknown/invalid option: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    if (n <= 1) {
        fprintf(stderr, "n must be > 1\n");
        return 2;
    }

    /* Build uniform grid */
    const int Ngrid = n*n*n;
    double *r_xyz = (double *)malloc((size_t)3*(size_t)Ngrid*sizeof(double));
    double *V = (double *)malloc((size_t)Ngrid*sizeof(double));
    double *vol = (double *)malloc((size_t)Ngrid*sizeof(double));
    if (!r_xyz || !V || !vol) {
        fprintf(stderr, "allocation failed\n");
        free(r_xyz);
        free(V);
        free(vol);
        return 3;
    }

    const double dx = (2.0*L)/(double)(n - 1);
    const double vol_cell = dx*dx*dx;
    const double inv2sig2 = 1.0/(2.0*sigma*sigma);

    int idx = 0;
    for (int i = 0; i < n; ++i) {
        const double x = -L + dx*(double)i;
        for (int j = 0; j < n; ++j) {
            const double y = -L + dx*(double)j;
            for (int k = 0; k < n; ++k) {
                const double z = -L + dx*(double)k;
                r_xyz[3*idx + 0] = x;
                r_xyz[3*idx + 1] = y;
                r_xyz[3*idx + 2] = z;

                const double r2 = x*x + y*y + z*z;
                V[idx] = exp(-r2*inv2sig2);
                vol[idx] = vol_cell;
                idx++;
            }
        }
    }

    /* Filter points where V > eps */
    int Nkeep = 0;
    for (int i = 0; i < Ngrid; ++i) {
        if (V[i] > eps) Nkeep++;
    }

    double *r2_xyz = (double *)malloc((size_t)3*(size_t)Nkeep*sizeof(double));
    double *V2 = (double *)malloc((size_t)Nkeep*sizeof(double));
    double *vol2 = (double *)malloc((size_t)Nkeep*sizeof(double));
    if (!r2_xyz || !V2 || !vol2) {
        fprintf(stderr, "allocation failed\n");
        free(r_xyz);
        free(V);
        free(vol);
        free(r2_xyz);
        free(V2);
        free(vol2);
        return 3;
    }

    int w = 0;
    for (int i = 0; i < Ngrid; ++i) {
        if (V[i] <= eps) continue;
        r2_xyz[3*w + 0] = r_xyz[3*i + 0];
        r2_xyz[3*w + 1] = r_xyz[3*i + 1];
        r2_xyz[3*w + 2] = r_xyz[3*i + 2];
        V2[w] = V[i];
        vol2[w] = vol[i];
        w++;
    }

    free(r_xyz);
    free(V);
    free(vol);

    if (isnan(soft)) soft = 0.5*dx;

    conformal_solver_t solver;
    if (cs_init(&solver, (size_t)Nkeep, r2_xyz, V2, vol2, M, soft, exclude_self) != 0) {
        fprintf(stderr, "solver init failed\n");
        free(r2_xyz);
        free(V2);
        free(vol2);
        return 4;
    }

    double *psi = (double *)malloc((size_t)Nkeep*sizeof(double));
    if (!psi) {
        fprintf(stderr, "allocation failed\n");
        free(r2_xyz);
        free(V2);
        free(vol2);
        return 3;
    }

    cs_convergence_info_t info;
    int rc = cs_solve(&solver, tol, max_iter, norm, verbose, psi, &info);
    if (rc != 0) {
        fprintf(stderr, "solve failed (code %d)\n", rc);
        free(r2_xyz);
        free(V2);
        free(vol2);
        free(psi);
        return 5;
    }

    double pmin = psi[0], pmax = psi[0];
    for (int i = 1; i < Nkeep; ++i) {
        if (psi[i] < pmin) pmin = psi[i];
        if (psi[i] > pmax) pmax = psi[i];
    }

    printf("Ngrid=%d  Nkeep=%d  dx=%.6g\n", Ngrid, Nkeep, dx);
    printf("W=%.16e\n", solver.W);
    printf("converged=%d  iterations=%d  last_error=%.3e\n", info.converged, info.iterations, info.last_error);
    printf("psi_min=%.16e  psi_max=%.16e\n", pmin, pmax);

    FILE *fp = fopen(out_path, "w");
    if (!fp) {
        fprintf(stderr, "failed to open output file: %s\n", out_path);
        free(r2_xyz);
        free(V2);
        free(vol2);
        free(psi);
        return 6;
    }
    fprintf(fp, "# 1:x 2:y 3:z 4:psi\n");
    for (int i = 0; i < Nkeep; ++i) {
        const double x = r2_xyz[3*i + 0];
        const double y = r2_xyz[3*i + 1];
        const double z = r2_xyz[3*i + 2];
        fprintf(fp, "%14.7e %14.7e %14.7e %14.7e\n", x, y, z, psi[i]);
    }
    fclose(fp);
    printf("wrote %d rows to %s\n", Nkeep, out_path);

    free(r2_xyz);
    free(V2);
    free(vol2);
    free(psi);
    return 0;
}
