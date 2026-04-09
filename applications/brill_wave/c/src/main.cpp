#include "conformal_solver.hpp"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

static void usage(const char *prog) {
	fprintf(stderr,
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
			prog);
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
#include <fenv.h>


int main(int argc, char **argv) {
	feenableexcept(FE_DIVBYZERO);
	int n = 15;
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

	const double dx = (2.0 * L) / (double)(n - 1);
	//	const double vol_cell = dx * dx * dx;
	//	const double inv2sig2 = 1.0 / (2.0 * sigma * sigma);

	int const Ngrid = n * n * n;

	/* Filter points where V > eps */

	conformal_solver_t solver(n, L, M, sigma, eps, soft);
	auto const [psi, info] = solver.solve(tol, max_iter, norm, verbose);
	double pmin = psi[0], pmax = psi[0];
	int const Nkeep = solver.getNkeep();
	for (int i = 1; i < Nkeep; ++i) {
		if (psi[i] < pmin) pmin = psi[i];
		if (psi[i] > pmax) pmax = psi[i];
	}
	printf("Ngrid=%d  Nkeep=%d  dx=%.6g\n", Ngrid, Nkeep, dx);
	printf("W=%.16e\n", solver.getW());
	printf("converged=%d  iterations=%d  last_error=%.3e\n", info.converged, info.iterations, info.last_error);
	printf("psi_min=%.16e  psi_max=%.16e\n", pmin, pmax);

	FILE *fp = fopen(out_path, "w");
	fprintf(fp, "# 1:x 2:y 3:z 4:psi\n");
	for (int i = 0; i < Nkeep; ++i) {
		const double x = solver.getPosition(i)[0];
		const double y = solver.getPosition(i)[1];
		const double z = solver.getPosition(i)[2];
		fprintf(fp, "%14.7e %14.7e %14.7e %14.7e\n", x, y, z, psi[i]);
	}
	fclose(fp);
	printf("wrote %d rows to %s\n", Nkeep, out_path);

	/* Also output a 1D cut of psi along the x axis, computed via psi_at_points
	   (brute-force summation over the masses), matching the Python notebook's
	   psi_line computation — this is the true conformal factor, not the
	   C*tilde_chi quantity that solve() returns on the grid. */
	{
		char xcut_path[1024];
		const char *dot = strrchr(out_path, '.');
		if (dot && dot != out_path) {
			const int base_len = (int)(dot - out_path);
			snprintf(xcut_path, sizeof(xcut_path), "%.*s_xcut%s", base_len, out_path, dot);
		} else {
			snprintf(xcut_path, sizeof(xcut_path), "%s_xcut", out_path);
		}

		const int K = 200;
		const double xmin = -3.0, xmax = 3.0;
		std::vector<std::array<double, 3>> r_eval(K);
		for (int i = 0; i < K; ++i) {
			const double x = xmin + (xmax - xmin) * (double)i / (double)(K - 1);
			r_eval[i] = {x, 0.0, 0.0};
		}

		std::vector<double> chi(Ngrid);
		for (int i = 0; i < Ngrid; ++i) chi[i] = psi[i] - 1.0;

		auto psi_line = solver.psi_at_points(r_eval, chi);

		FILE *fx = fopen(xcut_path, "w");
		if (!fx) {
			fprintf(stderr, "failed to open %s for writing\n", xcut_path);
		} else {
			fprintf(fx, "# 1:x 2:y 3:z 4:psi  (x-axis cut via psi_at_points, K=%d)\n", K);
			for (int i = 0; i < K; ++i) {
				fprintf(fx, "%14.7e %14.7e %14.7e %14.7e\n", r_eval[i][0], 0.0, 0.0, psi_line[i]);
			}
			fclose(fx);
			printf("wrote %d rows (x-axis cut) to %s\n", K, xcut_path);
		}
	}

	return 0;
}
