#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <complex>
#include <cstring>
#include <functional>
#include <set>
#include <stack>
#include <stdio.h>
#include <unordered_map>
#include <utility>
#include <vector>

// #define USE_INTEL

#define CHECK_NAN

#if USE_DOUBLE_FLAG == 1
#define USE_DOUBLE
#endif
#if USE_FLOAT_FLAG == 1
#define USE_FLOAT
#endif
#if USE_SIMD_FLAG == 1
#define USE_SIMD
#endif
#if USE_CUDA_FLAG == 1
#define USE_CUDA
#endif

// #define USE_SCALED

struct entry_t {
	int l;
	int m;
	int o;
};

int L2L_allrot(int P, int Q, int rot);
int greens_ewald2(int P, int Q, double alpha);

const auto cmp = [](entry_t a, entry_t b) {
	if (a.m < b.m) {
		return true;
	} else if (a.m > b.m) {
		return false;
	} else {
		return a.l < b.l;
	}
};

enum stage_t { PRE1, PRE2, POST1, POST2, XZ1, XZ2, FULL };

// #define NO_DIPOLE
#define USE_PERIODIC

struct flops_t {
	int r;
	int i;
	int fma;
	int rdiv;
	int idiv;
	int con;
	int icmp;
	int rcmp;
	int asgn;
	flops_t() {
		reset();
	}
	flops_t(const flops_t &) = default;
	flops_t(flops_t &&) = default;
	flops_t &operator=(const flops_t &) = default;
	flops_t &operator=(flops_t &&) = default;
	void reset() {
		r = 0;
		i = 0;
		icmp = 0;
		rcmp = 0;
		fma = 0;
		rdiv = 0;
		idiv = 0;
		con = 0;
		asgn = 0;
	}
	flops_t &operator+=(const flops_t &other) {
		r += other.r;
		i += other.i;
		fma += other.fma;
		rdiv += other.rdiv;
		idiv += other.idiv;
		con += other.con;
		asgn += other.asgn;
		icmp += other.icmp;
		rcmp += other.rcmp;
		return *this;
	}
	flops_t &operator*=(int a) {
		r *= a;
		i *= a;
		fma *= a;
		rdiv *= a;
		idiv *= a;
		con *= a;
		asgn *= a;
		icmp *= a;
		rcmp *= a;
		return *this;
	}
	int load() const;
};

using complex = std::complex<double>;

static int ntab = 0;
static int tprint_on = true;
static std::string *tprint_str;

#define TAB0()                                                                                                                             \
	if (ntab != 0) {                                                                                                                       \
		printf("Tab mismatch (%i) %s %i\n", ntab, __FILE__, __LINE__);                                                                     \
		ntab = 0;                                                                                                                          \
	}

#ifdef NO_DIPOLE
constexpr static int nodip = true;
#else
constexpr static int nodip = false;
#endif
#ifdef USE_PERIODIC
constexpr static int periodic = 1;
#else
constexpr static int periodic = 0;
#endif
#ifdef USE_SCALED
constexpr static int scaled = 1;
#else
constexpr static int scaled = 0;
#endif

std::array<std::unordered_map<std::string, flops_t>, PMAX + 1> flops_map;

// #define FLOAT
// #define DOUBLE
// #define CUDA_FLOAT
// #define CUDA_DOUBLE
// #define VEC_DOUBLE
// #define VEC_FLOAT
// #define VEC_DOUBLE_SIZE 2
// #define VEC_FLOAT_SIZE 8
std::vector<int> precision;
std::vector<int> simd;
std::vector<int> simd_size;
std::vector<std::string> base_rtype;
std::vector<std::string> rtype;
std::vector<std::string> base_itype;
std::vector<std::string> itype;
std::vector<std::string> base_uitype;
std::vector<std::string> uitype;
std::vector<std::string> fixed_type;
void regular_harmonic_full(int);

static std::string root_dir = std::string(ROOT_DIR) + "/include/";

#define ASPRINTF(...)                                                                                                                      \
	if (asprintf(__VA_ARGS__) == 0) {                                                                                                      \
		printf("ASPRINTF error %s %i\n", __FILE__, __LINE__);                                                                              \
		abort();                                                                                                                           \
	}
#define SYSTEM(...)                                                                                                                        \
	if (system(__VA_ARGS__) != 0) {                                                                                                        \
		printf("SYSTEM error %s %i\n", __FILE__, __LINE__);                                                                                \
		abort();                                                                                                                           \
	}

#define DEBUGNAN

static bool fmaops = true;
static int pmin = PMIN;
static int pmax = PMAX;
static std::string vtype = "simd_f32";
static std::string type;
static int typenum = -1;

static const int divops = 4;
static const char *prefix = "";
static std::string detail_header;
static std::string detail_header_vec;
static std::vector<std::string> lines[2];

static std::string header = "sfmm.hpp";
bool enable_scaled = true;
static std::string full_header = std::string("./generated_code/include/") + header;
// static std::string full_detail_header = std::string("./generated_code/include/detail/") + header;

static const char *period_name() {
	return periodic ? "_periodic" : "";
}

static const char *scaled_name() {
	return scaled ? "_scaled" : "";
}

std::string random_macro() {
	static std::set<int> used;
	char *macro;
	int num;
	do {
		num = rand();
	} while (used.find(num) != used.end());
	ASPRINTF(&macro, "SFMM_MACRO_%0i_%s", (unsigned)num, type.c_str());
	used.insert(num);
	std::string result = macro;
	free(macro);
	return result;
}

std::string type_macro() {
	return "SFMM_MACRO_" + type;
}

static const char *dip_name() {
	return nodip ? "_wo_dipole" : "";
}

template <class... Args>
std::string print2str(const char *fstr, Args &&...args) {
	std::string result;
	char *str;
	ASPRINTF(&str, fstr, std::forward<Args>(args)...);
	result = str;
	free(str);
	return result;
}

template <class... Args>
std::string print2str(const char *fstr) {
	std::string result = fstr;
	return result;
}

static bool is_float(std::string str) {
	return precision[typenum] == 1;
}

static bool is_double(std::string str) {
	return precision[typenum] == 2;
}

static bool is_vec(std::string str) {
	return simd[typenum];
}

const double ewald_r2 = (2.6 + 0.5 * sqrt(3));
const int ewald_h2 = 8;

static FILE *fp = nullptr;

int xyexp_sz(int P) {
	int index = 0;
	for (int l = 0; l <= P; l++) {
		for (int m = -l; m <= l; m++) {
			if (l % 2 != abs(m) % 2) {
				continue;
			}
			index++;
		}
	}
	return index;
}

int exp_sz(int P) {
	return (P + 1) * (P + 1);
}

int half_exp_sz(int P) {
	return (P + 2) * (P + 1) / 2;
}

int mul_sz(int P) {
	if (nodip) {
		if (P <= 1) {
			return 1;
		} else {
			return P * P - 3;
		}
	} else {
		return P * P;
	}
}

static flops_t greens_flops;

double tiny() {
	if (is_float(type)) {
		return 10.0 * std::numeric_limits<float>::min();
	} else {
		return 10.0 * std::numeric_limits<double>::min();
	}
}
double huge() {
	if (is_float(type)) {
		return 0.1 * std::numeric_limits<float>::max();
	} else {
		return 0.1 * std::numeric_limits<double>::max();
	}
}

double factorial(int n) {
	return n == 0 ? 1.0 : n * factorial(n - 1);
}

flops_t rescale_flops(int P) {
	flops_t fps;
	fps.asgn++;
	fps.r += (P + 1) * (P + 2) + 1;
	fps.rdiv++;
	return fps;
}

flops_t init_flops(int P) {
	flops_t fps;
	fps.asgn += (P + 1) * (P + 1) + 2;
	return fps;
}

flops_t copy_flops(int P) {
	flops_t fps;
	fps.asgn += (P + 1) * (P + 1) + 2;
	return fps;
}

flops_t accumulate_flops(int P) {
	flops_t fps;
	fps = rescale_flops(P);
	fps.r += (P + 1) * (P + 1) + 1;
	// fps.a++;
	return fps;
}

flops_t rsqrt_flops() {
	flops_t fps;
	fps.r += 4;
	fps.i += 2;
	fps.asgn += 2;
	fps.fma += 3;
	if (!is_float(type)) {
		fps.fma += 1;
	}
	return fps;
}

flops_t sqrt_flops() {
	flops_t fps = rsqrt_flops();
	fps.rdiv++;
	return fps;
}

flops_t sincos_flops() {
	flops_t fps;
	fps.r += 8;
	fps.fma += 10;
	fps.asgn += 3;
	fps.i += 8;
	fps.con += 2;
	if (!is_float(type)) {
		fps.fma += 10;
	}
	return fps;
}

flops_t erfcexp_flops() {
	flops_t fps;
	fps.r += 5;
	fps.rdiv += 1;
	fps.con += 2;
	fps.rcmp++;
	fps.i += 2;
	fps.fma += 7;
	fps.asgn++;
	if (!is_float(type)) {
		fps.fma += 11;
	}
	if (is_float(type)) {
		if (is_vec(type)) {
			fps.r += 14;
			fps.rcmp++;
			fps.con++;
			fps.rdiv += 2;
			fps.asgn += 2;
			fps.fma += 34;
		} else {
			fps.r += 7;
			fps.rcmp++;
			fps.asgn++;
			fps.fma += 25;
		}
	} else {
		if (is_vec(type)) {
			fps.r += 17;
			fps.rcmp += 4;
			fps.asgn += 5;
			fps.rdiv += 2;
			fps.fma += 76;
		} else {
			fps.r += 5;
			fps.rcmp += 2;
			fps.asgn += 2;
			fps.rdiv++;
			fps.fma += 36;
		}
	}
	return fps;
}

// #define COUNT_POT_FLOPS
flops_t parse_flops(const char *line);

flops_t count_flops(std::string fname) {
	FILE *fp = fopen(fname.c_str(), "rt");
	if (!fp) {
		printf("Unable to open %s\n", fname.c_str());
		abort();
	}
	constexpr int N = 1000;
	char line[N];
	flops_t fps;
	int ln = 0;
	while (fgets(line, N - 2, fp)) {
		if (line[0] != '#') {
			int j = 0;
			fps += parse_flops(line);
		}
		//		printf( "%i, %i : %s", fps.r, cnt, line);
	}

	//	printf( "%i\n", ln);
	fclose(fp);
	return fps;
}

void set_file(std::string file) {
	if (fp != nullptr) {
		fclose(fp);
	}
	fp = fopen(file.c_str(), "at");
	if (fp == nullptr) {
		printf("unable to open %s for writing\n", file.c_str());
		exit(0xFF);
	}
}

struct closer {
	~closer() {
		if (fp != nullptr) {
			fclose(fp);
		}
	}
};

static closer closer_instance;

void indent() {
	ntab++;
}

void deindent() {
	ntab--;
}
constexpr double dfactorial(int n) {
	if (n == 1 || n == 0) {
		return 1;
	} else {
		return n * dfactorial(n - 2);
	}
}

flops_t running_flops;

void reset_running_flops() {
	running_flops.reset();
}

flops_t get_running_flops(bool simd = true) {
	flops_t r = running_flops;
	// running_flops.reset();
	return r;
}

template <class... Args>
void tprint(const char *str, Args &&...args) {
	if (fp == nullptr) {
		return;
	}
	auto line = print2str(str, std::forward<Args>(args)...);
	running_flops += parse_flops(line.c_str());
	if (tprint_on) {
		if (tprint_str) {
			for (int i = 0; i < ntab; i++) {
				*tprint_str += "\t";
			}
			*tprint_str += line;
		} else {
			for (int i = 0; i < ntab; i++) {
				fprintf(fp, "\t");
			}
			fprintf(fp, "%s", line.c_str());
		}
	}
}

void tprint(const char *str) {
	tprint("%s", str);
}

constexpr int nchains = 3;
static int current_chain;
static std::vector<std::vector<std::string>> inschains(nchains);

void tprint_new_chain() {
	int best = -1, smallest = 1000000000;
	for (int i = 0; i < inschains.size(); i++) {
		if (inschains[i].size() < smallest) {
			smallest = inschains[i].size();
			best = i;
		}
	}
	current_chain = best;
}

template <class... Args>
void tprint_chain(const char *fstr, Args &&...args) {
	std::string str;
	for (int i = 0; i < ntab; i++) {
		str += "\t";
	}
	char *buf;
	ASPRINTF(&buf, fstr, std::forward<Args>(args)...)
	str += buf;
	free(buf);
	//	printf( "%i\n", current_chain);
	inschains[current_chain].push_back(str);
}

void tprint_chain(const char *fstr) {
	std::string str;
	for (int i = 0; i < ntab; i++) {
		str += "\t";
	}
	str += fstr;
	inschains[current_chain].push_back(str);
}

void tprint_flush_chains() {
	int n[nchains];
	for (int i = 0; i < nchains; i++) {
		n[i] = 0;
	}
	const auto done = [&n]() {
		for (int i = 0; i < nchains; i++) {
			if (n[i] < inschains[i].size()) {
				return false;
			}
		}
		return true;
	};
	while (!done()) {
		int best = -1;
		double largest = 0;
		for (int i = 0; i < inschains.size(); i++) {
			if (n[i] < inschains[i].size()) {
				double value = (double)(inschains[i].size() - n[i]) / (inschains[i].size() + 1);
				if (value >= largest) {
					largest = value;
					best = i;
				}
			}
		}
		if (best == -1) {
			abort();
		}
		if (fp) {
			running_flops += parse_flops(inschains[best][n[best]].c_str());
			fprintf(fp, "%s", inschains[best][n[best]].c_str());
		}
		n[best]++;
	}
	inschains = decltype(inschains)(nchains);
}

double nonepow(int i) {
	return i % 2 == 0 ? 1.0 : -1.0;
}

template <class T>
void ewald_limits(int &r2, int &h2, double alpha) {
	r2 = 0;
	h2 = 0;
	const auto threesqr = [](int i) {
		for (int x = 0; x <= i; x++) {
			for (int y = 0; y <= i; y++) {
				for (int z = 0; z <= i; z++) {
					if (x * x + y * y + z * z == i) {
						return true;
					}
				}
			}
		}
		return false;
	};

	double dx = 0.01;
	int R2 = 1;
	while (1) {
		double m = double(0);
		const int e = sqrt(R2);
		for (double x0 = 0; x0 < 0.5; x0 += dx) {
			for (double y0 = x0; y0 < 0.5; y0 += dx) {
				for (double z0 = y0; z0 < 0.5; z0 += dx) {
					for (int xi = 0; xi <= e; xi++) {
						for (int yi = xi; yi <= e; yi++) {
							for (int zi = yi; zi <= e; zi++) {
								if (xi * xi + yi * yi + zi * zi != R2) {
									continue;
								}
								const double x = x0 - xi;
								const double y = y0 - yi;
								const double z = z0 - zi;
								const double r0 = sqrt(x0 * x0 + y0 * y0 + z0 * z0);
								if (r0 != double(0)) {
									const double r = sqrt(x * x + y * y + z * z);
									double a = r0 * erfc(alpha * r) / r;
									m = std::max(m, fabs(a));
								}
							}
						}
					}
				}
			}
		}
		if (m < std::numeric_limits<T>::epsilon()) {
			break;
		}
		while (!threesqr(++R2))
			;
	}
	while (!threesqr(--R2))
		;
	r2 = R2;
	R2 = 1;
	while (1) {
		double m = double(0);
		const int e = sqrt(R2);
		int n = 0;
		for (double x0 = 0; x0 < 0.5; x0 += dx) {
			for (double y0 = x0; y0 < 0.5; y0 += dx) {
				for (double z0 = y0; z0 < 0.5; z0 += dx) {
					for (int xi = 0; xi <= e; xi++) {
						for (int yi = xi; yi <= e; yi++) {
							for (int zi = yi; zi <= e; zi++) {
								if (xi * xi + yi * yi + zi * zi != R2) {
									continue;
								}
								const double x = x0 - xi;
								const double y = y0 - yi;
								const double z = z0 - zi;
								const double h2 = xi * xi + yi * yi + zi * zi;
								const double hdotx = x * xi + y * yi + z * zi;
								const double r = sqrt(x * x + y * y + z * z);
								double a = r * fabs(cos(2.0 * M_PI * hdotx) * exp(-M_PI * M_PI * h2 / (alpha * alpha)) / (h2 * sqrt(M_PI)));
								m = std::max(m, fabs(a));
							}
						}
					}
				}
			}
		}
		if (m < std::numeric_limits<T>::epsilon()) {
			break;
		}
		while (!threesqr(++R2))
			;
	}
	while (!threesqr(--R2))
		;
	h2 = R2;
}

flops_t parse_flops(const char *line) {
	flops_t fps0;
	int j = 0;
	while (j < strlen(line)) {
		if (strncmp(line + j, " += ", 4) == 0) {
			fps0.r++;
			j += 4;
		} else if (strncmp(line + j, " -= ", 4) == 0) {
			fps0.r++;
			j += 4;
		} else if (strncmp(line + j, " = ", 3) == 0) {
			fps0.asgn++;
			j += 2;
		} else if (strncmp(line + j, " *= ", 4) == 0) {
			fps0.r++;
			j += 4;
		} else if (strncmp(line + j, " /= ", 4) == 0) {
			fps0.rdiv++;
			j += 4;
		} else if (strncmp(line + j, " + ", 3) == 0) {
			fps0.r++;
			j += 3;
		} else if (strncmp(line + j, " - ", 3) == 0) {
			fps0.r++;
			j += 3;
		} else if (strncmp(line + j, " -", 2) == 0) {
			fps0.r++;
			j += 2;
		} else if (strncmp(line + j, " * ", 3) == 0) {
			fps0.r++;
			j += 3;
		} else if (strncmp(line + j, " / ", 3) == 0) {
			fps0.rdiv++;
			j += 3;
		} else if (strncmp(line + j, " < ", 3) == 0) {
			fps0.rcmp++;
			j += 3;
		} else if (strncmp(line + j, " > ", 3) == 0) {
			fps0.rcmp++;
			j += 3;
		} else if (strncmp(line + j, "CONVERT", 7) == 0) {
			fps0.con++;
			j += 7;
		} else if (strncmp(line + j, "fma", 3) == 0) {
			fps0.fma++;
			j += 3;
		} else if (strncmp(line + j, "rsqrt", 5) == 0) {
			fps0 += rsqrt_flops();
			j += 5;
		} else if (strncmp(line + j, "sqrt", 4) == 0) {
			fps0 += sqrt_flops();
			j += 4;
		} else if (strncmp(line + j, "erfcexp", 7) == 0) {
			fps0 += erfcexp_flops();
			j += 7;
		} else if (strncmp(line + j, "sincos", 6) == 0) {
			fps0 += sincos_flops();
			j += 6;
		} else if (strncmp(line + j, "greens(", 7) == 0) {
			fps0 += greens_flops;
			j += 17;
		} else {
			j++;
		}
	}
	if (fps0.asgn) {
		if (fps0.con || fps0.fma || fps0.i || fps0.icmp || fps0.idiv || fps0.r || fps0.rcmp || fps0.rdiv) {
			fps0.asgn--;
		}
	}
	return fps0;
}

enum arg_type { LIT, PTR, CPTR, EXP, HEXP, XYEXP, MUL, CEXP, CMUL, FORCE, VEC3 };

void init_real(std::string var) {
	if (simd[typenum]) {
		tprint("T %s;\n", var.c_str());
#ifdef CHECK_NAN
		fprintf(fp, "#if !defined(NDEBUG) && !defined(__CUDA_ARCH__)\n");
		tprint("%s.set_NaN();\n", var.c_str());
		fprintf(fp, "#endif /* NDEBUG */ \n");
#endif
	} else {
		tprint("T %s;\n", var.c_str());
#ifdef CHECK_NAN
		fprintf(fp, "#if !defined(NDEBUG) && !defined(__CUDA_ARCH__)\n");
		tprint("%s=std::numeric_limits<%s>::signaling_NaN();\n", var.c_str(), base_rtype[typenum].c_str());
		fprintf(fp, "#endif /* NDEBUG */ \n");
#endif
	}
}

void init_reals(std::string var, int cnt) {
	if (simd[typenum]) {
		tprint("T %s [%i];\n", var.c_str(), cnt);
#ifdef CHECK_NAN
		fprintf(fp, "#if !defined(NDEBUG) && !defined(__CUDA_ARCH__)\n");
		tprint("for (int i = 0; i < %i; i++) {\n", cnt);
		indent();
		tprint("%s[i].set_NaN();\n", var.c_str());
		deindent();
		tprint("}\n");
		fprintf(fp, "#endif /* NDEBUG */ \n");
#endif
	} else {
#ifdef CHECK_NAN
		fprintf(fp, "#if !defined(NDEBUG) && !defined(__CUDA_ARCH__)\n");
		tprint("T %s[%i]={", var.c_str(), cnt);
		int ontab = ntab;
		ntab = 0;
		for (int n = 0; n < cnt; n++) {
			tprint("std::numeric_limits<%s>::signaling_NaN()", base_rtype[typenum].c_str());
			if (n != cnt - 1) {
				tprint(",");
			}
		}
		tprint("};\n");
		ntab = ontab;
		fprintf(fp, "#else /* NDEBUG */ \n");
#endif
		tprint("T %s[%i];\n", var.c_str(), cnt);
#ifdef CHECK_NAN
		fprintf(fp, "#endif /* NDEBUG */ \n");
#endif
	}
}

template <class... Args>
std::string func_args(int P, const char *arg, arg_type atype, int term) {
	std::string str;
	if (atype == VEC3) {
		str += std::string("vec3<") + type + "> " + arg;
	} else if (atype == EXP) {
		str += std::string("expansion") + "<" + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == XYEXP) {
		str += std::string("detail::expansion_xy") + "<" + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == HEXP) {
		str += std::string("detail::expansion_xz") + "<" + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == MUL) {
		str += std::string("multipole") + "<" + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == CEXP) {
		str += std::string("const expansion") + "<" + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == CMUL) {
		str += std::string("const multipole") + "<" + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == FORCE) {
		str += std::string("force_type<") + type + ">& " + arg;
	} else {
		if (atype == CPTR) {
			str += std::string("const ");
		}
		str += type + " ";
		if (atype == PTR || atype == CPTR) {
			str += "*";
		}
		str += std::string(arg);
	}
	return str;
}

template <class... Args>
std::string func_args(int P, const char *arg, arg_type atype, Args &&...args) {
	auto str = func_args(P, arg, atype, 1);
	str += std::string(", ");
	str += func_args(P, std::forward<Args>(args)...);
	return str;
}
template <class... Args>
std::string func_args_sig(int P, const char *arg, arg_type atype, int term) {
	std::string str;
	if (atype == VEC3) {
		str += std::string("vec3<") + type + ">";
	} else if (atype == EXP) {
		str += std::string("expansion") + "<" + type + ", " + std::to_string(P) + ">&";
	} else if (atype == XYEXP) {
		str += std::string("detail::expansion_xy") + "<" + type + ", " + std::to_string(P) + ">&";
	} else if (atype == HEXP) {
		str += std::string("detail::expansion_xz") + "<" + type + ", " + std::to_string(P) + ">&";
	} else if (atype == MUL) {
		str += std::string("multipole") + "<" + type + ", " + std::to_string(P) + ">&";
	} else if (atype == CEXP) {
		str += std::string("const expansion") + "<" + type + ", " + std::to_string(P) + ">&";
	} else if (atype == CMUL) {
		str += std::string("const multipole") + "<" + type + ", " + std::to_string(P) + ">&";
	} else if (atype == FORCE) {
		str += std::string("force_type<") + type + ">&";
	} else {
		if (atype == CPTR) {
			str += std::string("const ");
		}
		str += type + "";
		if (atype == PTR || atype == CPTR) {
			str += "*";
		}
	}
	return str;
}

template <class... Args>
std::string func_args_sig(int P, const char *arg, arg_type atype, Args &&...args) {
	auto str = func_args_sig(P, arg, atype, 1);
	str += std::string(", ");
	str += func_args_sig(P, std::forward<Args>(args)...);
	return str;
}

template <class... Args>
std::string func_args_call(int P, const char *arg, arg_type atype, int term) {
	std::string str;
	if (atype == EXP) {
		str += std::string(arg) + "_st";
	} else if (atype == HEXP) {
		str += std::string(arg) + "_st";
	} else if (atype == XYEXP) {
		str += std::string(arg) + "_st";
	} else if (atype == MUL) {
		str += std::string(arg) + "_st";
	} else if (atype == CEXP) {
		str += std::string(arg) + "_st";
	} else if (atype == CMUL) {
		str += std::string(arg) + "_st";
	} else if (atype == FORCE) {
		str += std::string(arg);
	} else {
		str += std::string(arg);
	}
	return str;
}

template <class... Args>
std::string func_args_call(int P, const char *arg, arg_type atype, Args &&...args) {
	auto str = func_args_call(P, arg, atype, 1);
	str += std::string(", ");
	str += func_args_call(P, std::forward<Args>(args)...);
	return str;
}

template <class... Args>
void func_args_cover(int P, const char *arg, arg_type atype, int term) {
	std::string str;
	if (atype == EXP) {
		tprint("T* %s(%s_st.data());\n", arg, arg);
	} else if (atype == HEXP) {
		tprint("T* %s(%s_st.data());\n", arg, arg);
	} else if (atype == XYEXP) {
		tprint("T* %s(%s_st.data());\n", arg, arg);
	} else if (atype == CEXP) {
		tprint("const T* %s(%s_st.data());\n", arg, arg);
	} else if (atype == MUL) {
		tprint("T* %s(%s_st.data());\n", arg, arg);
	} else if (atype == CMUL) {
		tprint("const T* %s(%s_st.data());\n", arg, arg);
	}
}

template <class... Args>
void func_args_cover(int P, const char *arg, arg_type atype, Args &&...args) {
	func_args_cover(P, arg, atype, 1);
	func_args_cover(P, std::forward<Args>(args)...);
}

template <class... Args>
void func_args_dummies(int P, const char *arg, arg_type atype, int term) {
	std::string str;
	if (atype == EXP) {
		tprint("static expansion<%s, %i> %s_dummy;\n", type.c_str(), P, arg);
	} else if (atype == HEXP) {
		tprint("static expansion_xz<%s, %i> %s_dummy;\n", type.c_str(), P, arg);
	} else if (atype == XYEXP) {
		tprint("static expansion_xy<%s, %i> %s_dummy;\n", type.c_str(), P, arg);
	} else if (atype == CEXP) {
		tprint("static expansion<%s, %i> %s_dummy;\n", type.c_str(), P, arg);
	} else if (atype == MUL) {
		tprint("static multipole<%s, %i> %s_dummy;\n", type.c_str(), P, arg);
	} else if (atype == CMUL) {
		tprint("static multipole<%s, %i> %s_dummy;\n", type.c_str(), P, arg);
	} else if (atype == FORCE) {
		tprint("static force_type<%s> %s_dummy;\n", type.c_str(), arg);
	} else if (atype == VEC3) {
		tprint("static vec3<%s> %s_dummy;\n", type.c_str(), arg);
	}
}

template <class... Args>
void func_args_dummies(int P, const char *arg, arg_type atype, Args &&...args) {
	func_args_dummies(P, arg, atype, 1);
	func_args_dummies(P, std::forward<Args>(args)...);
}

void include(std::string filename) {
	constexpr int N = 1024;
	char buffer[N];
	filename = root_dir + filename;
	FILE *fp0 = fopen(filename.c_str(), "rt");
	if (!fp0) {
		printf("Unable to open %s\n", filename.c_str());
		abort();
	}
	while (!feof(fp0)) {
		if (!fgets(buffer, N, fp0)) {
			break;
		}
		fprintf(fp, "%s", buffer);
	}
	fclose(fp0);
}

std::string timing_body;
std::string current_sig;
int timing_cnt = 0;

template <class... Args>
std::string func_header(const char *func, int P, bool pub, bool calcpot, bool timing, bool flops, bool vec, std::string head,
						Args &&...args) {
	static std::set<std::string> igen;
	reset_running_flops();
	std::string func_name = std::string(func);
	std::string func_ptr = print2str("int(*)(");
	func_ptr += func_args_sig(P, std::forward<Args>(args)..., 0);
	func_ptr += ", int";
	func_ptr += ")";
	current_sig = func_ptr;
	if (timing) {
		if (timing_body != "") {
			timing_body += ",\n";
		}
		timing_body += print2str("\t{(void*)((%s) &%s), \"%s\", ", func_ptr.c_str(), func_name.c_str(), type.c_str());
		timing_cnt++;
	}
	std::string file_name = "./generated_code/include/detail/sfmm_detail.hpp";
	func_name = "int " + func_name;
	func_name += "(" + func_args(P, std::forward<Args>(args)..., 0);
	auto func_name2 = func_name;
	func_name += ", int flags = sfmmDefaultFlags)";
	func_name2 += ", int flags)";
	set_file(full_header.c_str());
	static std::set<std::string> already_printed;
	if (already_printed.find(func_name) == already_printed.end()) {
		already_printed.insert(func_name);
		if (prefix[0]) {
			func_name = std::string(prefix) + " " + func_name;
			func_name2 = std::string(prefix) + " " + func_name2;
		}
		if (simd[typenum]) {
			tprint("#ifndef __CUDACC__\n");
		}
		if (pub) {
			tprint("%s;\n", func_name.c_str());
		} else {
			if (is_vec(type)) {
				detail_header_vec += func_name + ";\n";
			} else {
				detail_header += func_name + ";\n";
			}
		}
		if (simd[typenum]) {
			tprint("#endif\n");
		}
	} else {
		if (prefix[0]) {
			func_name = std::string(prefix) + " " + func_name;
			func_name2 = std::string(prefix) + " " + func_name2;
		}
	}
	std::string func1 = std::string(func) + std::string("_") + type;
	// file_name = dir + "/" + file_name;
	set_file(file_name);
	//	tprint("#include \"%s\"\n", header.c_str());
	//	tprint("#include \"typecast_%s.hpp\"\n", type.c_str());
	tprint("\n");
	tprint("namespace sfmm {\n");
	tprint("#ifndef __CUDACC__\n");
	tprint("using namespace simd;\n");
	tprint("#endif\n");
	if (!pub) {
		tprint("namespace detail {\n");
	}
	if (head != "") {
		tprint("\n");
		tprint("%s\n", head.c_str());
	}
	tprint("\n");
	tprint("inline %s {\n", func_name2.c_str());
	indent();
	tprint("using T = %s;\n", rtype[typenum].c_str());
	tprint("using V = %s;\n", itype[typenum].c_str());
	tprint("using TCONVERT = T;\n");
	tprint("using VCONVERT = V;\n");
	tprint("const static auto TCAST = [](%s a) { return %s(%s(a));};\n", base_rtype[typenum].c_str(), rtype[typenum].c_str(),
		   base_rtype[typenum].c_str());
	tprint("const static auto VCAST = [](%s a) { return %s(%s(a));};\n", base_itype[typenum].c_str(), itype[typenum].c_str(),
		   base_itype[typenum].c_str());
	if (flops) {
		tprint("if( !(flags & sfmmFLOPsOnly) ) {\n");
		indent();
	}
	if (calcpot) {
		func_args_cover(P, std::forward<Args>(args)..., 0);
		if (vec) {
			tprint("T& x=dx[0];\n");
			tprint("T& y=dx[1];\n");
			tprint("T& z=dx[2];\n");
		}
	}
	return file_name;
}

void fixed_point_covers() {
	if (simd[typenum]) {
	}
	constexpr int N = 8;
	const char *protos[N] = {
		"SFMM_PREFIX inline int M2L%s(expansion<%s,P>& L, const multipole<%s,P>& M, vec3<%s> x0, vec3<%s> x1, int flags = sfmmDefaultFlags "
		") {\n",
		"SFMM_PREFIX inline int M2P%s(force_type<%s>& f, const multipole<%s,P>& M, vec3<%s> x0, vec3<%s> x1, int flags = sfmmDefaultFlags "
		") {\n",
		"SFMM_PREFIX inline int P2L%s(expansion<%s,P>& L, %s m, vec3<%s> x0, vec3<%s> x1, int flags = sfmmDefaultFlags ) {\n",
		"SFMM_PREFIX inline int P2P%s(force_type<%s>& f, %s m, vec3<%s> x0, vec3<%s> x1, int flags = sfmmDefaultFlags ) {\n",
		"SFMM_PREFIX inline int L2L%s(expansion<%s,P>& L, vec3<%s> x0, vec3<%s> x1, int flags = sfmmDefaultFlags ) {\n",
		"SFMM_PREFIX inline int L2P%s(force_type<%s>& f, expansion<%s,P>& L, vec3<%s> x0, vec3<%s> x1, int flags = sfmmDefaultFlags ) {\n",
		"SFMM_PREFIX inline int P2M%s(multipole<%s,P>& M, %s m, vec3<%s> x0, vec3<%s> x1, int flags = sfmmDefaultFlags ) {\n",
		"SFMM_PREFIX inline int M2M%s(multipole<%s,P>& M, vec3<%s> x0, vec3<%s> x1, int flags = sfmmDefaultFlags ) {\n"};

	const char *calls[N] = {"return M2L%s(L, M, dx, flags) + %i;", "return M2P%s(f, M, dx, flags) + %i;",
							"return P2L%s(L, m, dx, flags) + %i;", "return P2P%s(f, m, dx, flags) + %i;",
							"return L2L%s(L, dx, flags) + %i;",	   "return L2P%s(f, L, dx, flags) + %i;",
							"return P2M%s(M, m, dx, flags) + %i;", "return M2M%s(M, dx, flags) + %i;"};

	tprint("SFMM_PREFIX inline %s distance(const %s& a, const %s& b) {\n", type.c_str(), type.c_str(), type.c_str());
	indent();
	tprint("const %s c = a - b;\n", type.c_str());
	tprint("const %s absc = abs(c);\n", type.c_str());
	tprint("return copysign(fmin(absc, %s(1) - absc), c * (%s(0.5) - absc));\n", type.c_str(), type.c_str());
	deindent();
	tprint("}\n");
	const int nparams[N] = {2, 2, 2, 2, 1, 2, 2, 1};
	for (int i = 0; i < N; i++) {
		for (int k = 0; k < 2; k++) {
			if (i >= 4 && k == 1) {
				continue;
			}
			const char *estr = k == 0 ? "" : "_ewald";
			for (int j = 0; j < 2; j++) {
				if ((i == 7 && j == 1)) {
					continue;
				}
				reset_running_flops();
				tprint("\n");
				auto vtype = j == 0 ? type : fixed_type[typenum];
				if (i != 3) {
					tprint("template<int P>\n");
				}
				if (nparams[i] == 1) {
					tprint(protos[i], estr, type.c_str(), vtype.c_str(), vtype.c_str());
				} else {
					tprint(protos[i], estr, type.c_str(), type.c_str(), vtype.c_str(), vtype.c_str());
				}
				indent();
				tprint("vec3<%s> dx;\n", type.c_str());
				for (int dim = 0; dim < 3; dim++) {
					if (i < 4) {
						tprint("dx[%i] = distance(x1[%i], x0[%i]);\n", dim, dim, dim);
					} else {
						tprint("dx[%i] = x0[%i] - x1[%i];\n", dim, dim, dim);
					}
				}
				auto flops = get_running_flops();
				if (i < 4) {
					if (j != 0) {
						flops.r += 3;
					} else {
						flops.r += 21;
					}
				}
				tprint(calls[i], estr, flops.load());
				deindent();
				tprint("\n}\n");
				reset_running_flops();
			}
		}
	}
	if (simd[typenum]) {
	}
}

void create_func_data_ptr(std::string fname) {
	return;
	tprint("static auto* const func_data_ptr = detail::operator_initialize((void*)((%s) &%s));\n", current_sig.c_str(), fname.c_str());
}

void open_timer(std::string fname) {
	return;
	tprint("static auto* func_data_ptr = detail::operator_initialize((void*)((%s) &%s));\n", current_sig.c_str(), fname.c_str());
	tprint("timer tm;\n");
	tprint("if( flags & sfmmProfilingOn ) {\n");
	indent();
	tprint("tm.start();\n");
	deindent();
	tprint("}\n");
}

void close_timer() {
	return;
	tprint("if( flags & sfmmProfilingOn ) {\n");
	indent();
	tprint("tm.stop();\n");
	tprint("detail::operator_update_timing(func_data_ptr, tm.read());\n");
	deindent();
	tprint("}\n");
}

void set_tprint(bool c) {
	tprint_on = c;
}

int lindex(int l, int m) {
	return l * (l + 1) + m;
}

int xyindex(int l0, int m0) {
	int index = 0;
	for (int l = 0;; l++) {
		for (int m = -l; m <= l; m++) {
			if (l % 2 != abs(m) % 2) {
				if (l == l0 && m == m0) {
					abort();
				} else {
					continue;
				}
			}
			if (l == l0 && m == m0) {
				return index;
			}
			index++;
		}
	}
}

int mindex(int l, int m) {
	if (nodip) {
		if (l == 1) {
			abort();
		}
		return std::max(0, lindex(l, m) - 3);
	} else {
		return lindex(l, m);
	}
}

constexpr int cindex(int l, int m) {
	return l * (l + 1) / 2 + m;
}

template <class T>
T nonepow(int m) {
	return m % 2 == 0 ? double(1) : double(-1);
}

template <int P>
struct spherical_expansion : public std::array<complex, (P + 1) * (P + 2) / 2> {
	inline complex operator()(int n, int m) const {
		if (m >= 0) {
			return (*this)[cindex(n, m)];
		} else {
			return (*this)[cindex(n, -m)].conj() * nonepow<double>(m);
		}
	}
	void print() const {
		for (int l = 0; l <= P; l++) {
			for (int m = 0; m <= l; m++) {
				printf("%e + i %e  ", (*this)(l, m).real(), (*this)(l, m).imag());
			}
			printf("\n");
		}
	}
};

struct hash {
	size_t operator()(std::array<int, 3> i) const noexcept {
		return i[0] * 12345 + i[1] * 42 + i[2];
	}
};

double Brot(int n, int m, int l) {
	static std::unordered_map<std::array<int, 3>, double, hash> values;
	std::array<int, 3> key;
	key[0] = n;
	key[1] = m;
	key[2] = l;
	if (values.find(key) != values.end()) {
		return values[key];
	} else {
		double v;
		if (n == 0 && m == 0 && l == 0) {
			v = 1.0;
		} else if (abs(l) > n) {
			v = 0.0;
		} else if (m == 0) {
			v = 0.5 * (Brot(n - 1, m, l - 1) - Brot(n - 1, m, l + 1));
		} else if (m > 0) {
			v = 0.5 * (Brot(n - 1, m - 1, l - 1) + Brot(n - 1, m - 1, l + 1) + 2.0 * Brot(n - 1, m - 1, l));
		} else {
			v = 0.5 * (Brot(n - 1, m + 1, l - 1) + Brot(n - 1, m + 1, l + 1) - 2.0 * Brot(n - 1, m + 1, l));
		}
		values[key] = v;
		return v;
	}
}

bool close21(double a) {
	return std::abs(1.0 - a) < 1.0e-20;
}

void z_rot2(int P, const char *dst, const char *src, stage_t stage, std::string opname, bool init, bool multipole,
			std::pair<int, int> range = std::make_pair(-1, -1)) {
	if (range.first == -1) {
		range.first = 1;
		range.second = P;
	}
	if (init) {
		tprint("r0[0] = cosphi;\n");
		tprint("ip[0] = sinphi;\n");
		tprint("in[0] = -sinphi;\n");
		for (int m = 1; m < P; m++) {
			tprint("r0[%i] = in[%i] * sinphi;\n", m, m - 1);
			tprint("ip[%i] = ip[%i] * cosphi;\n", m, m - 1);
			tprint("r0[%i] = fma(r0[%i], cosphi, r0[%i]);\n", m, m - 1, m);
			tprint("ip[%i] = fma(r0[%i], sinphi, ip[%i]);\n", m, m - 1, m);
			tprint("in[%i] = -ip[%i];\n", m, m);
		}
	}
	int mmin = 1;
	bool initR = true;
	std::function<int(int, int)> index;
	if (multipole) {
		index = mindex;
	} else {
		index = lindex;
	}
	std::vector<int> set_nan;
	using cmd_t = std::pair<int, std::string>;
	std::vector<cmd_t> cmds;
	cmds.push_back(std::make_pair(0, print2str("%s[0] = %s[0];\n", dst, src)));
	for (int m = range.first; m <= range.second; m++) {
		if (!(nodip && m == 1 && multipole)) {
			cmds.push_back(std::make_pair(index(m, 0), print2str("%s[%i] = %s[%i];\n", dst, index(m, 0), src, index(m, 0))));
		}
		for (int l = m; l <= P; l++) {
			if (multipole && nodip && l == 1) {
				continue;
			}
			if (stage == POST1) {
				if (l == P) {
					if (l % 2 != m % 2) {
						continue;
					}
				} else if (nodip && l == P - 1) {
					if (l % 2 != m % 2) {
						continue;
					}
				}
			}
			bool read_ionly = false;
			bool read_ronly = false;
			bool write_ronly = false;
			if (stage == POST2) {
				if (l == P) {
					read_ionly = m % 2;
				} else if (nodip && l == P - 1) {
					read_ionly = m % 2;
				}
				if (l == P) {
					read_ronly = !(m % 2);
				} else if (nodip && l == P - 1) {
					read_ronly = !(m % 2);
				}
			} else if (stage == PRE2) {
				if (l == P || opname == "M2P" || opname == "L2P") {
					write_ronly = m % 2 != l % 2;
				}
			}
			read_ronly = read_ronly || (stage == XZ1 && l >= P);
			read_ronly = read_ronly || (stage == XZ2 && l >= P - 1);
			bool sw = false;
			if (stage == POST1 && (l >= P - 1)) {
				if (nodip) {
					sw = true;
				} else {
					sw = m % 2 == P % 2;
				}
			}
			read_ronly = read_ronly || sw;
			if (read_ionly) {
				cmds.push_back(
					std::make_pair(index(l, -m), print2str("%s[%i] = %s[%i] * in[%i];\n", dst, index(l, m), src, index(l, -m), m - 1)));
				cmds.push_back(
					std::make_pair(index(l, -m), print2str("%s[%i] = %s[%i] * r0[%i];\n", dst, index(l, -m), src, index(l, -m), m - 1)));
			} else if (read_ronly) {
				cmds.push_back(
					std::make_pair(index(l, m), print2str("%s[%i] = %s[%i] * ip[%i];\n", dst, index(l, -m), src, index(l, m), m - 1)));
				cmds.push_back(
					std::make_pair(index(l, m), print2str("%s[%i] = %s[%i] * r0[%i];\n", dst, index(l, m), src, index(l, m), m - 1)));
			} else if (write_ronly) {
				cmds.push_back(
					std::make_pair(index(l, -m), print2str("%s[%i] = %s[%i] * in[%i];\n", dst, index(l, m), src, index(l, -m), m - 1)));
				cmds.push_back(std::make_pair(index(l, m), print2str("%s[%i] = fma(%s[%i], r0[%i], %s[%i]);\n", dst, index(l, m), src,
																	 index(l, m), m - 1, dst, index(l, m))));
				set_nan.push_back(index(l, -m));
			} else {
				cmds.push_back(
					std::make_pair(index(l, -m), print2str("%s[%i] = %s[%i] * in[%i];\n", dst, index(l, m), src, index(l, -m), m - 1)));
				cmds.push_back(
					std::make_pair(index(l, -m), print2str("%s[%i] = %s[%i] * r0[%i];\n", dst, index(l, -m), src, index(l, -m), m - 1)));
				cmds.push_back(std::make_pair(index(l, m), print2str("%s[%i] = fma(%s[%i], r0[%i], %s[%i]);\n", dst, index(l, m), src,
																	 index(l, m), m - 1, dst, index(l, m))));
				cmds.push_back(std::make_pair(index(l, m), print2str("%s[%i] = fma(%s[%i], ip[%i], %s[%i]);\n", dst, index(l, -m), src,
																	 index(l, m), m - 1, dst, index(l, -m))));
			}
		}
	}
	std::sort(cmds.begin(), cmds.end(), [](const cmd_t &a, const cmd_t &b) {
		return a.first < b.first;
	});
	for (auto cmd : cmds) {
		tprint("%s", cmd.second.c_str());
	}
#ifdef CHECK_NAN
	if (set_nan.size()) {
		fprintf(fp, "#ifndef NDEBUG\n");
		fprintf(fp, "#if !defined(NDEBUG) && !defined(__CUDA_ARCH__)\n");
		for (auto i : set_nan) {
			tprint("%s[%i]=std::numeric_limits<%s>::signaling_NaN();\n", dst, i, type.c_str());
		}
		fprintf(fp, "#endif\n");
		fprintf(fp, "#endif /* NDEBUG */ \n");
	}
#endif
}

void xz_swap2(int P, const char *dst, const char *src, bool inv, stage_t stage, const char *opname = "") {
	auto brot = [inv](int n, int m, int l) {
		if (inv) {
			return Brot(n, m, l);
		} else {
			return Brot(n, l, m);
		}
	};
	std::function<int(int, int)> index;
	if (dst[0] == 'M') {
		index = mindex;
	} else {
		index = lindex;
	}
	tprint("%s[0] = %s[0];\n", dst, src);
	std::vector<int> set_nan;
	for (int n = 1; n <= P; n++) {
		if (dst[0] == 'M' && nodip && n == 1) {
			continue;
		}
		int lmax = n;
		if (stage == POST1) {
			lmax = std::min(n, P - n);
			if (nodip && n == P - 1) {
				lmax = 0;
			}
		}
		std::vector<std::vector<std::pair<double, int>>> ops(2 * n + 1);
		int mmax = n;
		if (stage == PRE2) {
			mmax = std::min((P + 1) - n, n);
			if (std::string(opname) == std::string("M2P") || std::string(opname) == std::string("L2P")) {
				mmax = std::min(1, mmax);
			}
		}
		for (int m = 0; m <= n; m++) {
			if (m <= mmax) {
				for (int l = 0; l <= lmax; l++) {
					bool flag = false;
					if (stage == POST2) {
						if (P == n && n % 2 != abs(l) % 2) {
							continue;
						} else if (nodip && P - 1 == n && n % 2 != abs(l) % 2) {
							continue;
						}
					}
					double r = l == 0 ? brot(n, m, 0) : brot(n, m, l) + nonepow<double>(l) * brot(n, m, -l);
					double i = l == 0 ? 0.0 : brot(n, m, l) - nonepow<double>(l) * brot(n, m, -l);
					if (r != 0.0) {
						ops[n + m].push_back(std::make_pair(r, l));
					}
					if (i != 0.0 && m != 0) {
						ops[n - m].push_back(std::make_pair(i, -l));
					}
				}
			} else {
				set_nan.push_back(index(n, m));
				set_nan.push_back(index(n, -m));
			}
		}
		for (int m = 0; m < 2 * n + 1; m++) {
			std::sort(ops[m].begin(), ops[m].end(), [n, index](std::pair<double, int> a, std::pair<double, int> b) {
				return index(n, a.second) < index(n, b.second);
			});
		}
		for (int ppp = 0; ppp < (P + 1) * (P + 1); ppp++) {
			for (int m = 0; m < 2 * n + 1; m++) {
				for (int l = 0; l < ops[m].size(); l++) {
					int len = 1;
					while (len + l < ops[m].size()) {
						if (ops[m][len + l].first == ops[m][l].first && !close21(ops[m][l].first)) {
							len++;
						} else {
							break;
						}
					}
					if (index(n, ops[m][l].second) != ppp) {
						continue;
					}
					if (close21(ops[m][l].first)) {
						tprint("%s[%i] %s= %s[%i];\n", dst, index(n, m - n), l == 0 ? "" : "+", src, index(n, ops[m][l].second));
					} else {
						if (l == 0) {
							tprint("%s[%i] = TCAST(%.20e) * %s[%i];\n", dst, index(n, m - n), ops[m][l].first, src,
								   index(n, ops[m][l].second));
						} else {
							tprint("%s[%i] = fma(TCAST(%.20e), %s[%i], %s[%i]);\n", dst, index(n, m - n), ops[m][l].first, src,
								   index(n, ops[m][l].second), dst, index(n, m - n));
						}
					}
					l += len - 1;
				}
			}
		}
	}
	if (set_nan.size()) {
#ifdef CHECK_NAN
		fprintf(fp, "#if !defined(NDEBUG) && !defined(__CUDA_ARCH__)\n");
		fprintf(fp, "#ifndef NDEBUG\n");
		for (auto i : set_nan) {
			tprint("%s[%i]=std::numeric_limits<%s>::signaling_NaN();\n", dst, i, type.c_str());
		}
		fprintf(fp, "#endif /* NDEBUG */ \n");
		fprintf(fp, "#endif\n");
#endif
	}
}

void greens_body(int P, const char *M = nullptr) {
	tprint("r2 = fma(x, x, fma(y, y, z * z));\n");
	tprint("r2inv = TCAST(1) / r2;\n");
	tprint("O[0] = -rsqrt(r2);\n");
	if (M) {
		tprint("O[0] *= %s;\n", M);
	}
	tprint("x *= -r2inv;\n");
	tprint("y *= -r2inv;\n");
	tprint("z *= -r2inv;\n");
	auto index = lindex;
	tprint("O[%i] = x * O[0];\n", index(1, 1));
	tprint("O[%i] = y * O[0];\n", index(1, -1));
	for (int m = 2; m <= P; m++) {
		tprint("ax0 = O[%i] * TCAST(%i);\n", index(m - 1, m - 1), 2 * m - 1);
		tprint("ay0 = O[%i] * TCAST(%i);\n", index(m - 1, -(m - 1)), 2 * m - 1);
		tprint("O[%i] = x * ax0 - y * ay0;\n", index(m, m));
		tprint("O[%i] = fma(y, ax0, x * ay0);\n", index(m, -m));
	}
	if (2 <= P) {
		tprint("O[%i] = -r2inv * O[0];\n", lindex(2, 0), lindex(0, 0));
	}
	if (1 <= P) {
		tprint("O[%i] = z * O[0];\n", lindex(1, 0));
	}
	for (int n = 1; n < P; n++) {
		const double c0 = -(double((n + 1) * (n + 1)));
		const double c1 = double(2 * n + 1);
		int mmax = n - 1;
		tprint("ax0 = TCAST(%.20e) * z;\n", c1);
		tprint("O[%i] = ax0 * O[%i];\n", lindex(n + 1, -n), lindex(n, -n));
		for (int m = -mmax; m <= mmax; m++) {
			tprint("O[%i] = fma(ax0, O[%i], O[%i]);\n", lindex(n + 1, m), lindex(n, m), lindex(n + 1, m));
		}
		tprint("O[%i] = ax0 * O[%i];\n", lindex(n + 1, n), lindex(n, n));
		if (n != P - 1) {
			tprint("O[%i] = TCAST(%.20e) * r2inv * O[%i];\n", lindex(n + 2, 0), c0, lindex(n, 0));
			for (int m = 1; m <= n; m++) {
				const double c0 = -(double((n + 1) * (n + 1)) - double(m * m));
				tprint("ax%i = TCAST(%.20e) * r2inv;\n", m % 3, c0);
				tprint("O[%i] = ax%i * O[%i];\n", lindex(n + 2, m), m % 3, lindex(n, m));
				tprint("O[%i] = ax%i * O[%i];\n", lindex(n + 2, -m), m % 3, lindex(n, -m));
			}
		}
	}
}

void mg2l_body(int P, int Q, bool ronly = false, std::function<int(int, int)> oindex = lindex, bool acc = false) {
	std::vector<entry_t> pos;
	std::vector<entry_t> neg;
	for (int n = 0; n <= Q; n++) {
		for (int m = 0; m <= n; m++) {
			bool iload = true;
			bool rload = true;
			tprint_new_chain();
			const int kmax = std::min(P - n, P - 1);
			for (int k = 0; k <= kmax; k++) {
				if (nodip && k == 1) {
					continue;
				}
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					bool greal = false;
					bool mreal = false;
					int gxsgn = 1;
					int gysgn = 1;
					int mxsgn = 1;
					int mysgn = 1;
					int gxstr = -1;
					int gystr = -1;
					int mxstr = -1;
					int mystr = -1;
					if (m + l > 0) {
						gxstr = oindex(n + k, m + l);
						gystr = oindex(n + k, -m - l);
					} else if (m + l < 0) {
						if (abs(m + l) % 2 == 0) {
							gxstr = oindex(n + k, -m - l);
							gystr = oindex(n + k, m + l);
							gysgn = -1;
						} else {
							gxstr = oindex(n + k, -m - l);
							gystr = oindex(n + k, m + l);
							gxsgn = -1;
						}
					} else {
						greal = true;
						gxstr = oindex(n + k, 0);
					}
					if (l > 0) {
						mxstr = mindex(k, l);
						mystr = mindex(k, -l);
						mysgn = -1;
					} else if (l < 0) {
						if (l % 2 == 0) {
							mxstr = mindex(k, -l);
							mystr = mindex(k, l);
						} else {
							mxstr = mindex(k, -l);
							mystr = mindex(k, l);
							mxsgn = -1;
							mysgn = -1;
						}
					} else {
						mreal = true;
						mxstr = mindex(k, 0);
					}
					const auto csgn = [](int i) {
						return i > 0 ? '+' : '-';
					};
					const auto add_work = [n, ronly, &gystr, &pos, &neg](int sgn, int m, int mstr, int gstr) {
						if (ronly && gystr == gstr) {
							if (gystr == 0 && mstr == 0) {
								//			abort();
							}
							//		printf( " %i %i\n", mstr, gstr);
							return;
						}
						entry_t entry;
						entry.l = lindex(n, m);
						entry.m = mstr;
						entry.o = gstr;
						if (sgn == 1) {
							pos.push_back(entry);
						} else {
							neg.push_back(entry);
						}
					};
					if (!mreal) {
						if (!greal) {
							add_work(mxsgn * gxsgn, m, mxstr, gxstr);
							add_work(-mysgn * gysgn, m, mystr, gystr);
							if (m > 0) {
								add_work(mysgn * gxsgn, -m, mystr, gxstr);
								add_work(mxsgn * gysgn, -m, mxstr, gystr);
							}
						} else {
							add_work(mxsgn * gxsgn, m, mxstr, gxstr);
							if (m > 0) {
								add_work(mysgn * gxsgn, -m, mystr, gxstr);
							}
						}
					} else {
						if (!greal) {
							add_work(mxsgn * gxsgn, m, mxstr, gxstr);
							if (m > 0) {
								add_work(mxsgn * gysgn, -m, mxstr, gystr);
							}
						} else {
							add_work(mxsgn * gxsgn, m, mxstr, gxstr);
						}
					}
				}
			}
		}
	}
	std::sort(neg.begin(), neg.end(), cmp);
	std::sort(pos.begin(), pos.end(), cmp);
	bool first[(P + 1) * (P + 1)];
	for (int n = 0; n < (P + 1) * (P + 1); n++) {
		first[n] = true;
	}
	if (acc) {
		for (int i = 0; i < neg.size(); i++) {
			if (!first[neg[i].l]) {
				tprint("L[%i] = -L[%i];\n", neg[i].l, neg[i].l);
				first[neg[i].l] = false;
			}
		}
	}
	for (int i = 0; i < neg.size(); i++) {
		if (first[neg[i].l]) {
			tprint("L[%i] = M[%i] * O[%i];\n", neg[i].l, neg[i].m, neg[i].o);
			first[neg[i].l] = false;
		} else {
			tprint("L[%i] = fma(M[%i], O[%i], L[%i]);\n", neg[i].l, neg[i].m, neg[i].o, neg[i].l);
		}
	}
	for (int n = 0; n < (P + 1) * (P + 1); n++) {
		if (!first[n]) {
			tprint("L[%i] = -L[%i];\n", n, n);
		}
	}
	for (int i = 0; i < pos.size(); i++) {
		if (first[pos[i].l] && !acc) {
			first[pos[i].l] = false;
			tprint("L[%i] = M[%i] * O[%i];\n", pos[i].l, pos[i].m, pos[i].o);
		} else {
			tprint("L[%i] = fma(M[%i], O[%i], L[%i]);\n", pos[i].l, pos[i].m, pos[i].o, pos[i].l);
		}
	}
}

std::vector<complex> spherical_singular_harmonic(int P, double x, double y, double z) {
	const double r2 = x * x + y * y + z * z;
	const double r2inv = double(1) / r2;
	complex R = complex(x, y);
	std::vector<complex> O(exp_sz(P));
	O[cindex(0, 0)] = complex(sqrt(r2inv), double(0));
	R *= r2inv;
	z *= r2inv;
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
			O[cindex(m, m)] = O[cindex(m - 1, m - 1)] * R * double(2 * m - 1);
		}
		if (m + 1 <= P) {
			O[cindex(m + 1, m)] = double(2 * m + 1) * z * O[cindex(m, m)];
		}
		for (int n = m + 2; n <= P; n++) {
			O[cindex(n, m)] =
				(double(2 * n - 1) * z * O[cindex(n - 1, m)] - double((n - 1) * (n - 1) - m * m) * r2inv * O[cindex(n - 2, m)]);
		}
	}
	return O;
}

bool close2zero(double a) {
	return std::abs(a) < 1e-10;
}

std::string P2L(int P) {
	auto fname = func_header("P2L", P, true, true, true, true, true, "", "L", EXP, "m", LIT, "dx", VEC3);
	init_real("tmp1");
	init_real("r2");
	init_real("r2inv");
	init_real("ax0");
	init_real("ay0");
	init_real("ax1");
	init_real("ax2");
	reset_running_flops();
	if (scaled) {
		tprint("%s scale;\n", base_rtype[typenum].c_str());
		tprint("expansion<T,%i> O_st(L_st.scale());\n", P);
		tprint("T* O=O_st.data();\n");
		tprint("tmp1 = TCAST(1) / L_st.scale();\n");
		tprint("x *= tmp1;\n");
		tprint("y *= tmp1;\n");
		tprint("z *= tmp1;\n");
	} else {
		tprint("expansion<T,%i> O_st;\n", P);
		tprint("T* O=O_st.data();\n");
	}
	greens_body(P, "m");
	for (int n = 0; n < exp_sz(P); n++) {
		tprint("L[%i] += O[%i];\n", n, n);
	}
	deindent();
	tprint("}\n");
	tprint("return %i;\n", get_running_flops().load());
	timing_body += print2str("\"P2L\", %i, 0, %i, 0.0, 0}", P, get_running_flops(false).load() + flops_map[P][type].load());
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	TAB0();
	return fname;
}

flops_t greens(int P) {
	auto fname = func_header("greens", P, true, true, false, true, true, "", "O", EXP, "dx", VEC3);
	init_real("r2");
	init_real("r2inv");
	init_real("ax0");
	init_real("ay0");
	init_real("ax1");
	init_real("ax2");
	reset_running_flops();
	greens_body(P);
	deindent();
	tprint("}\n");
	tprint("return %i;\n", get_running_flops().load());
	flops_map[P][type + "greens"] = get_running_flops();
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	TAB0();
	return get_running_flops();
}

void MG2L(int P) {
	func_header("MG2L", P, true, true, false, true, false, "", "Lout", EXP, "M", CMUL, "O", EXP);
	reset_running_flops();
	tprint("expansion<T,%i> L_st;\n", P);
	tprint("T* L=L_st.data();\n");
	mg2l_body(P, P, false, lindex, true);
	if (P > 2 && periodic) {
		tprint("L[%i] = fma(TCAST(-0.5) * O_st.trace2(), M_st.trace2(), L[%i]);\n", lindex(0, 0), lindex(0, 0));
	}
	if (P > 1 && periodic) {
		if (!nodip) {
			tprint("L[%i] = fma(TCAST(-2) * O_st.trace2(), M[%i], L[%i]);\n", lindex(1, -1), mindex(1, -1), lindex(1, -1));
			tprint("L[%i] -= O_st.trace2() * M[%i];\n", lindex(1, +0), lindex(1, +0), mindex(1, +0));
			tprint("L[%i] = fma(TCAST(-2) * O_st.trace2(), M[%i], L[%i]);\n", lindex(1, +1), mindex(1, +1), lindex(1, +1));
		}
		tprint("L_st.trace2() = O_st.trace2() * M[%i];\n", mindex(0, 0));
	}
	for (int n = 0; n < exp_sz(P); n++) {
		tprint("Lout[%i] += L[%i];\n", n, n);
	}
	if (periodic) {
		tprint("Lout_st.trace2() += L_st.trace2();\n");
	}
	deindent();
	tprint("}\n");
	tprint("return %i;\n", get_running_flops().load());
	flops_map[P][type + "MG2L"] = get_running_flops();
	deindent();
	tprint("}\n");

	tprint("\n");
	tprint("}\n");
	tprint("\n");
	TAB0();
}

using etype = long double;

std::vector<std::complex<etype>> accurate_greens(int P, etype x, etype y, etype z) {
	const etype r2 = x * x + y * y + z * z;
	const etype r2inv = etype(1) / r2;
	std::complex<etype> R = std::complex<etype>(x, y);
	std::vector<std::complex<etype>> O(exp_sz(P));
	O[cindex(0, 0)] = std::complex<etype>(-sqrt(r2inv), etype(0));
	R *= r2inv;
	z *= r2inv;
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
			O[cindex(m, m)] = O[cindex(m - 1, m - 1)] * R * etype(2 * m - 1);
		}
		if (m + 1 <= P) {
			O[cindex(m + 1, m)] = etype(2 * m + 1) * z * O[cindex(m, m)];
		}
		for (int n = m + 2; n <= P; n++) {
			O[cindex(n, m)] = (etype(2 * n - 1) * z * O[cindex(n - 1, m)] - etype((n - 1) * (n - 1) - m * m) * r2inv * O[cindex(n - 2, m)]);
		}
	}
	return O;
}

void ewald_zero(std::complex<etype> *G, int P) {
	constexpr int R = 5;
	constexpr int H = 3;
	for (int i = 0; i < exp_sz(P); i++) {
		G[i] = 0.0;
	}
	const etype pi = 4.0L * atanl(1.0L);
	G[0].real(4.0L / sqrtl(M_PI) + M_PI / 4.0);
	for (int xi = -R; xi <= R; xi++) {
		for (int yi = -R; yi <= R; yi++) {
			for (int zi = -R; zi <= R; zi++) {
				const int iii = xi * xi + yi * yi + zi * zi;
				if (iii > 12 || iii == 0) {
					continue;
				}
				const etype x = xi;
				const etype y = yi;
				const etype z = zi;
				const etype r2 = x * x + y * y + z * z;
				const etype r = sqrtl(r2);
				auto greal = accurate_greens(P, x, y, z);
				const etype xxx = 2.0L * r;
				etype gam1 = erfcl(xxx);
				const etype exp0 = expl(-xxx * xxx);
				gam1 *= sqrtl(pi);
				const etype xfac = (4.0L) * r2;
				etype xpow = (2.0L) * r;
				etype gam0inv = 1.0L / sqrtl(pi);
				for (int l = 0; l <= P; l++) {
					const etype gam = gam1 * gam0inv;
					for (int m = 0; m <= l; m++) {
						G[cindex(l, m)] += gam * greal[cindex(l, m)];
					}
					gam0inv /= (etype(l) + (0.5L));
					gam1 = fma((etype(l) + etype(0.5)), gam1, xpow * exp0);
					xpow *= xfac;
				}
			}
		}
	}
	for (int xi = -R; xi <= R; xi++) {
		for (int yi = -R; yi <= R; yi++) {
			for (int zi = -R; zi <= R; zi++) {
				const int iii = xi * xi + yi * yi + zi * zi;
				if (iii > 10 || iii == 0) {
					continue;
				}
				const etype hx = xi;
				const etype hy = yi;
				const etype hz = zi;
				const etype h2 = hx * hx + hy * hy + hz * hz;
				auto R = std::complex<etype>(1.0L, 0.0L) * expl(-0.25L * pi * pi * h2);
				auto gfour = accurate_greens(P, hx, hy, hz);
				const auto I = std::complex<etype>(0, -pi);
				etype gam0inv = 1.0L / sqrtl(pi);
				for (int l = 0; l <= P; l++) {
					const etype hpow = powl(h2, l - 0.5L);
					for (int m = 0; m <= l; m++) {
						G[cindex(l, m)] += R * hpow * gfour[cindex(l, m)] * gam0inv / sqrtl(pi);
					}
					R *= I;
					gam0inv /= (etype(l) + (0.5L));
				}
			}
		}
	}
}

constexpr int PEXTRA = 0;

std::string greens_ewald(int P, double alpha) {
	auto index = lindex;
	int R2, H2;
	if (is_float(type)) {
		ewald_limits<float>(R2, H2, alpha);
	} else {
		ewald_limits<double>(R2, H2, alpha);
	}

	int R = sqrt(R2);
	int H = sqrt(H2);
	if (!periodic) {
		return 0;
	}
	auto fname = func_header("greens_ewald", P, true, true, false, true, true, "", "G", EXP, "dx", VEC3);
	reset_running_flops();
	const auto c = tprint_on;
	tprint("expansion<%s, %i> Gr_st;\n", type.c_str(), P);
	tprint("T* Gr(Gr_st.data());\n", type.c_str(), P);
	int cnt = 0;
	for (int ix = -R; ix <= R; ix++) {
		for (int iy = -R; iy <= R; iy++) {
			for (int iz = -R; iz <= R; iz++) {
				const int i2 = ix * ix + iy * iy + iz * iz;
				if (i2 > R2 && i2 != 0) {
					continue;
				}
				cnt++;
			}
		}
	}
	bool first = true;
	tprint("expansion<%s,%i> G0_st;\n", type.c_str(), P);
	tprint("T* G0=G0_st.data();\n");
	int PY = P + PEXTRA;
	//	PY = 2 * (PY / 2);
	init_reals("Y", exp_sz(PY));
	init_real("sw");
	init_real("r");
	init_real("ax0");
	init_real("ay0");
	init_real("r2");
	init_real("xxx");
	init_real("tmp0");
	init_real("tmp1");
	init_real("tmp2");
	init_real("gam1");
	init_real("exp0");
	init_real("cgam");
	init_real("xfac");
	init_real("xpow");
	init_real("gam");
	init_real("x2");
	init_real("x2y2");
	init_real("hdotx");
	init_real("phi");
	init_real("rzero");
	init_real("rsmall");
	//	tprint( "dx = -dx;\n");
	const auto name = [](const char *base, int hx, int hy, int hz) {
		std::string s = base;
		const auto add_symbol = [&s](int h) {
			if (h < 0) {
				s += "n";
				s += std::to_string(-h);
			} else if (h > 0) {
				s += "p";
				s += std::to_string(h);
			} else {
				s += "x";
			}
		};
		add_symbol(hx);
		add_symbol(hy);
		add_symbol(hz);
		return s;
	};
	const auto cosname = [name](int hx, int hy, int hz) {
		return name("cos", hx, hy, hz);
	};
	const auto sinname = [name](int hx, int hy, int hz) {
		return name("sin", hx, hy, hz);
	};
	bool close = false;
	for (int hx = -H; hx <= H; hx++) {
		for (int hy = -H; hy <= H; hy++) {
			for (int hz = -H; hz <= H; hz++) {
				const int h2 = hx * hx + hy * hy + hz * hz;
				if (h2 <= H2 && h2 > 0) {
					init_real(cosname(hx, hy, hz));
					init_real(sinname(hx, hy, hz));
				}
			}
		}
	}
	auto flops = get_running_flops();
	reset_running_flops();
	tprint("const auto ewald_real = [&G,&Gr,&Gr_st](T x, T y, T z) {\n");
	indent();
	tprint("T gam1, exp0, gam;\n");
	tprint("const T r2 = fma(x, x, fma(y, y, z * z));\n");
	tprint("const T r = sqrt(r2);\n");
	tprint("greens(Gr_st, vec3<T>( x, y, z));\n");
	tprint("const T xxx = T(%.20e) * r;\n", alpha);
	tprint("gam1 = erfc(xxx);\n");
	tprint("exp0 = exp(-xxx * xxx);\n");
	tprint("gam1 *= T(%.20e);\n", sqrt(M_PI));
	tprint("const T xfac = T(%.20e) * r2;\n", alpha * alpha);
	tprint("T xpow = T(%.20e) * r;\n", alpha);
	double gam0inv = 1.0 / sqrt(M_PI);
	for (int l = 0; l <= P; l++) {

		tprint("gam = gam1 * T(%.20e);\n", gam0inv);
		for (int m = -l; m <= l; m++) {
			tprint("G[%i] = fma(gam, Gr[%i], G[%i]);\n", index(l, m), index(l, m), index(l, m));
		}
		gam0inv /= (double(l) + double(0.5));
		if (l != P) {
			tprint("gam1 = fma(T(%f), gam1, xpow * exp0);\n", l + 0.5);
			tprint("xpow *= xfac;\n");
		}
	}
	deindent();
	tprint("};\n");
	flops_t real_flops = get_running_flops();
	reset_running_flops();
	tprint("r2 = fma(x, x, fma(y, y, z * z));\n");
	tprint("rzero = TCONVERT(r2 < TCAST(%0.20e));\n", tiny());
	tprint("rsmall = TCONVERT(r2 < TCAST(.05*.05));\n");
	tprint("r = sqrt(r2) + rzero;\n");
	tprint("greens(Gr_st, vec3<T>(x + rzero, y, z));\n");
	tprint("xxx = TCAST(%.20e) * r;\n", alpha);
	tprint("gam1 = erfc(xxx);\n");
	tprint("exp0 = exp(-xxx * xxx);\n");
	tprint("gam1 *= TCAST(%.20e);\n", sqrt(M_PI));
	tprint("xfac = TCAST(%.20e) * r2;\n", alpha * alpha);
	tprint("xpow = TCAST(%.20e) * r;\n", alpha);
	gam0inv = 1.0 / sqrt(M_PI);
	tprint("sw = TCAST(1) - rzero;\n");
	for (int l = 0; l <= P; l++) {
		tprint("gam = gam1 * TCAST(%.20e);\n", gam0inv);
		tprint("cgam = (gam - TCAST(1)) * sw;\n");
		for (int m = -l; m <= l; m++) {
			tprint("G[%i] = cgam * Gr[%i];\n", lindex(l, m), lindex(l, m));
		}
		if (l == 0) {
			tprint("G[0] += rzero * TCAST(%.20e);\n", 2.0 * alpha / sqrt(M_PI));
		}
		gam0inv *= 1.0 / (l + 0.5);
		if (l != P) {
			tprint("gam1 = fma(TCAST(%.20e), gam1, xpow * exp0);\n", l + 0.5);
			if (l != P - 1) {
				tprint("xpow *= xfac;\n");
			}
		}
	}
	for (int ix = -R; ix <= R; ix++) {
		for (int iy = -R; iy <= R; iy++) {
			for (int iz = -R; iz <= R; iz++) {
				int ii = ix * ix + iy * iy + iz * iz;
				if (ii > R2 || ii == 0) {
					continue;
				}
				std::string xstr = "x";
				if (ix != 0) {
					xstr += std::string(" ") + (ix < 0 ? "+" : "-") + " TCAST(" + std::to_string(abs(ix)) + ")";
				}

				std::string ystr = "y";
				if (iy != 0) {
					ystr += std::string(" ") + (iy < 0 ? "+" : "-") + " TCAST(" + std::to_string(abs(iy)) + ")";
				}
				std::string zstr = "z";
				if (iz != 0) {
					zstr += std::string(" ") + (iz < 0 ? "+" : "-") + " TCAST(" + std::to_string(abs(iz)) + ")";
				}
				tprint("ewald_real((%s), (%s), (%s));\n", xstr.c_str(), ystr.c_str(), zstr.c_str());
				flops += real_flops;
			}
		}
	}

	for (int hx = -H; hx <= H; hx++) {
		if (hx) {
			if (abs(hx) == 1) {
				tprint("x2 = %cx;\n", hx < 0 ? '-' : ' ');
			} else {
				tprint("x2 = TCAST(%i) * x;\n", hx);
			}
		} else {
			tprint("x2 = TCAST(0);\n");
		}
		for (int hy = -H; hy <= H; hy++) {
			if (hx * hx + hy * hy > H2) {
				continue;
			}
			if (hy) {
				if (abs(hy) == 1) {
					tprint("x2y2 = x2 %c y;\n", hy > 0 ? '+' : '-');
				} else {
					tprint("x2y2 = fma(TCAST(%i), y, x2);\n", hy);
				}
			} else {
				tprint("x2y2 = x2;\n", hy);
			}
			for (int hz = -H; hz <= H; hz++) {
				const int h2 = hx * hx + hy * hy + hz * hz;
				if (h2 > H2 || h2 == 0) {
					continue;
				}
				if (hz) {
					if (abs(hz) == 1) {
						tprint("hdotx = x2y2 %c z;\n", hz > 0 ? '+' : '-');
					} else {
						tprint("hdotx = fma(TCAST(%i), z, x2y2);\n", hz);
					}
				} else {
					tprint("hdotx = x2y2;\n", hz);
				}
				tprint("phi = TCAST(%.20e) * hdotx;\n", 2.0 * M_PI);
				tprint("%s = sin(phi);\n", sinname(hx, hy, hz).c_str());
				tprint("%s = cos(phi);\n", cosname(hx, hy, hz).c_str());
			}
		}
	}
	using table_type = std::unordered_map<double, std::vector<std::pair<int, std::string>>>;
	table_type ops[exp_sz(P)];
	for (int hx = -H; hx <= H; hx++) {
		for (int hy = -H; hy <= H; hy++) {
			for (int hz = -H; hz <= H; hz++) {
				const int h2 = hx * hx + hy * hy + hz * hz;
				if (h2 <= H2 && h2 > 0) {
					const double h = sqrt(h2);
					bool init = false;
					const auto G0 = spherical_singular_harmonic(P, (double)hx, (double)hy, (double)hz);
					double gam0inv = 1.0 / sqrt(M_PI);
					double hpow = 1. / h;
					double pipow = 1. / sqrt(M_PI);
					for (int l = 0; l <= P; l++) {
						for (int m = 0; m <= l; m++) {
							double c0 = gam0inv * hpow * pipow * exp(-h * h * double(M_PI * M_PI) / (alpha * alpha));
							std::string ax = cosname(hx, hy, hz);
							std::string ay = sinname(hx, hy, hz);
							int xsgn = 1;
							int ysgn = 1;
							if (l % 4 == 3) {
								std::swap(ax, ay);
								ysgn = -1;
							} else if (l % 4 == 2) {
								ysgn = xsgn = -1;
							} else if (l % 4 == 1) {
								std::swap(ax, ay);
								xsgn = -1;
							}
							if (G0[cindex(l, m)].real() != (0)) {
								const double a = c0 * G0[cindex(l, m)].real();
								//						tprint("G[%i] += TCAST(%.20e) * %s;\n", index(l, m), -xsgn * c0 * G0[cindex(l,
								//m)].real(), ax.c_str());
								ops[index(l, m)][fabs(a)].push_back(std::make_pair(copysign(1.0, -xsgn * a), ax));
								if (m != 0) {
									//								tprint("G[%i] += TCAST(%.20e) * %s;\n", index(l, -m), -ysgn * c0 *
									//G0[cindex(l, m)].real(), ay.c_str());
									ops[index(l, -m)][fabs(a)].push_back(std::make_pair(copysign(1.0, -ysgn * a), ay));
								}
							}
							if (G0[cindex(l, m)].imag() != (0)) {
								const double a = c0 * G0[cindex(l, m)].imag();
								//								tprint("G[%i] += TCAST(%.20e) * %s;\n", index(l, m), ysgn * c0 *
								//G0[cindex(l, m)].imag(), ay.c_str());
								ops[index(l, -m)][fabs(a)].push_back(std::make_pair(copysign(1.0, ysgn * a), ay));
								if (m != 0) {
									//	tprint("G[%i] += TCAST(%.20e) * %s;\n", index(l, -m), -xsgn * c0 * G0[cindex(l, m)].imag(),
									//ax.c_str());
									ops[index(l, -m)][fabs(a)].push_back(std::make_pair(copysign(1.0, -xsgn * a), ax));
								}
							}
						}
						gam0inv /= l + 0.5;
						hpow *= h * h;
						pipow *= M_PI;
					}
				}
			}
		}
	}
	for (int ii = 0; ii < exp_sz(P); ii++) {
		table_type next_ops;
		for (auto j = ops[ii].begin(); j != ops[ii].end(); j++) {
			std::vector<int> remove;
			const std::vector<std::pair<int, std::string>> &these_ops = j->second;
			for (int dim = 0; dim < 3; dim++) {
				for (int hx = 1; hx <= H; hx++) {
					std::string sym1, sym2;
					if (dim == 0) {
						sym1 = sinname(hx, 0, 0);
						sym2 = sinname(-hx, 0, 0);
					} else if (dim == 1) {
						sym1 = sinname(0, hx, 0);
						sym2 = sinname(0, -hx, 0);
					} else {
						sym1 = sinname(0, 0, hx);
						sym2 = sinname(0, 0, -hx);
					}
					for (int k = 0; k < these_ops.size(); k++) {
						if (these_ops[k].second == sym1) {
							for (int l = 0; l < these_ops.size(); l++) {
								if (these_ops[l].second == sym2) {
									remove.push_back(l);
									remove.push_back(k);
									if (these_ops[l].first * these_ops[k].first < 0) {
										double a = 2 * j->first;
										next_ops[a].push_back(std::make_pair(these_ops[k].first > 0 ? 1 : -1, these_ops[k].second));
									}
								}
							}
						}
					}
					if (dim == 0) {
						sym1 = cosname(hx, 0, 0);
						sym2 = cosname(-hx, 0, 0);
					} else if (dim == 1) {
						sym1 = cosname(0, hx, 0);
						sym2 = cosname(0, -hx, 0);
					} else {
						sym1 = cosname(0, 0, hx);
						sym2 = cosname(0, 0, -hx);
					}
					for (int k = 0; k < these_ops.size(); k++) {
						if (these_ops[k].second == sym1) {
							for (int l = 0; l < these_ops.size(); l++) {
								if (these_ops[l].second == sym2) {
									remove.push_back(l);
									remove.push_back(k);
									if (these_ops[l].first * these_ops[k].first > 0) {
										double a = 2 * j->first;
										next_ops[a].push_back(std::make_pair(these_ops[k].first > 0 ? 1 : -1, these_ops[k].second));
									}
								}
							}
						}
					}
				}
			}
			std::vector<std::pair<int, std::string>> these_next_ops;
			for (int k = 0; k < these_ops.size(); k++) {
				bool use = true;
				for (int l = 0; l < remove.size(); l++) {
					if (k == remove[l]) {
						use = false;
						break;
					}
				}
				if (use) {
					these_next_ops.push_back(these_ops[k]);
				}
			}
			next_ops[j->first].insert(next_ops[j->first].end(), these_next_ops.begin(), these_next_ops.end());
		}
		ops[ii] = std::move(next_ops);
	}
	for (int ii = 0; ii < exp_sz(P); ii++) {
		std::vector<std::pair<double, std::vector<std::pair<int, std::string>>>> sorted_ops(ops[ii].begin(), ops[ii].end());
		std::sort(sorted_ops.begin(), sorted_ops.end(),
				  [](const std::pair<double, std::vector<std::pair<int, std::string>>> &a,
					 const std::pair<double, std::vector<std::pair<int, std::string>>> &b) {
					  return fabs(a.first) * sqrt(a.second.size()) < fabs(b.first) * sqrt(b.second.size());
				  });
		tprint_new_chain();
		for (auto j = sorted_ops.begin(); j != sorted_ops.end(); j++) {
			auto op = j->second;
			if (op.size()) {
				int sgn = op[0].first > 0 ? 1 : -1;
				for (int k = 0; k < op.size(); k++) {
					tprint_chain("tmp%i %c= %s;\n", current_chain, k == 0 ? ' ' : (sgn * op[k].first > 0 ? '+' : '-'),
								 op[k].second.c_str());
				}
				if (sgn > 0) {
					tprint_chain("G[%i] = fma(TCAST(+%.20e), tmp%i, G[%i]);\n", ii, sgn * j->first, current_chain, ii);
				} else {
					tprint_chain("G[%i] = fma(TCAST(%.20e), tmp%i, G[%i]);\n ", ii, sgn * j->first, current_chain, ii);
				}
			}
		}
	}
	tprint_flush_chains();

	std::complex<etype> G0[(P + PEXTRA + 1) * (P + PEXTRA + 1)];
	tprint("G[%i] += TCAST(%.20e);\n", index(0, 0), M_PI / (alpha * alpha));
	ewald_zero(G0, P + PEXTRA);

	struct entry_t {
		int l;
		double v;
		int o;
	};
	tprint("x = -x;\n");
	tprint("y = -y;\n");
	tprint("z = -z;\n");
	regular_harmonic_full(PY);
	std::vector<entry_t> pos;
	std::vector<bool> used((P + 1) * (P + 1), false);
	for (int n = 0; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			const auto add_work = [&pos, n](int sgn, int m, double mstr, int gstr) {
				if (std::abs(mstr) < 1) {
					return;
				}
				entry_t entry;
				entry.l = lindex(n, m);
				entry.v = sgn * mstr;
				entry.o = gstr;
				pos.push_back(entry);
			};
			for (int k = 0; k <= P + PEXTRA - n; k++) {
				for (int l = -k; l <= k; l++) {
					double mxstr = -99;
					int mystr = -99;
					int gxstr = -99;
					int gystr = -99;
					int mxsgn = 1;
					int mysgn = 1;
					int gxsgn = 1;
					int gysgn = 1;
					if (abs(m + l) > n + k) {
						continue;
					}
					if (-abs(m + l) < -(k + n)) {
						continue;
					}
					mxstr = G0[cindex(n + k, abs(m + l))].real();
					if (m + l > 0) {
					} else if (m + l < 0) {
						if (abs(m + l) % 2 == 0) {
						} else {
							mxsgn = -1;
						}
					}
					if (l > 0) {
						gxstr = lindex(k, abs(l));
						gystr = lindex(k, -abs(l));
						gysgn = -1;
					} else if (l < 0) {
						if (abs(l) % 2 == 0) {
							gxstr = lindex(k, abs(l));
							gystr = lindex(k, -abs(l));
						} else {
							gxstr = lindex(k, abs(l));
							gystr = lindex(k, -abs(l));
							gysgn = -1;
							gxsgn = -1;
						}
					} else {
						gxstr = lindex(k, 0);
					}
					add_work(mxsgn * gxsgn, m, mxstr, gxstr);
					if (m > 0) {
						if (gystr != -99) {
							add_work(mxsgn * gysgn, -m, mxstr, gystr);
						}
					}
				}
			}
		}

		const auto cmp = [](entry_t a, entry_t b) {
			if (a.l < b.l) {
				return true;
			} else if (a.l > b.l) {
				return false;
			} else {
				return std::abs(a.v) < std::abs(b.v);
			}
		};
		std::sort(pos.begin(), pos.end(), cmp);
		for (const auto &e : pos) {
			if (used[e.l]) {
				tprint("G0[%i] = fma(Y[%i], TCAST(%.20e), G0[%i]);\n", e.l, e.o, e.v, e.l);
			} else {
				tprint("G0[%i] = Y[%i] * TCAST(%.20e);\n", e.l, e.o, e.v);
				used[e.l] = true;
			}
		}
		pos.resize(0);
	}
	for (int i = 0; i < exp_sz(P); i++) {
		if (!used[i]) {
			tprint("G0[%i] = TCAST(0);\n", i);
		}
	}
	tprint("G_st.trace2() = TCAST(%.20e);\n", (4.0 * M_PI / 3.0));
	tprint("G0_st.trace2() = TCAST(%.20e);\n", (4.0 * M_PI / 3.0));
	tprint("G0[%i] = fma(x, G0_st.trace2(), G0[%i]);\n", index(1, 1), index(1, 1));
	tprint("G0[%i] = fma(y, G0_st.trace2(), G0[%i]);\n", index(1, -1), index(1, -1));
	tprint("G0[%i] = fma(z, G0_st.trace2(), G0[%i]);\n", index(1, 0), index(1, 0));
	tprint("G0[%i] = fma(-TCAST(0.5)*r2, G0_st.trace2(), G0[%i]);\n", index(0, 0), index(0, 0));
	tprint("sw = TCAST(1) - rsmall;\n");
	for (int n = 0; n <= P; n++) {
		for (int m = -n; m <= n; m++) {
			tprint("G[%i] = fma(sw, G[%i], rsmall * G0[%i]);\n", index(n, m), index(n, m), index(n, m));
		}
	}
	deindent();
	tprint("}\n");
	flops += get_running_flops();
	tprint("return %i;\n", flops.load());
	flops_map[P][type + "greens_ewald"] = flops;
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	TAB0();
	return fname;
}

const char *boolstr(bool b) {
	return b ? "true" : "false";
}

void greens_xz_body(int P) {
	tprint("O[0] = -sqrt(r2inv);\n");
	tprint("R *= -r2inv;\n");
	tprint("z *= -r2inv;\n");
	const auto index = [](int l, int m) {
		return l * (l + 1) / 2 + m;
	};
	tprint("O[%i] = R * O[0];\n", index(1, 1));
	for (int m = 2; m <= P; m++) {
		tprint("O[%i] = R * O[%i] * TCAST(%i);\n", index(m, m), index(m - 1, m - 1), 2 * m - 1);
	}
	if (2 <= P) {
		tprint("O[%i] = -r2inv * O[0];\n", index(2, 0), index(0, 0));
	}
	if (1 <= P) {
		tprint("O[%i] = z * O[0];\n", index(1, 0));
	}
	for (int n = 1; n < P; n++) {
		const double c0 = -(double((n + 1) * (n + 1)));
		const double c1 = double(2 * n + 1);
		int mmax = n - 1;
		tprint("ax0 = TCAST(%.20e) * z;\n", c1);
		for (int m = 0; m <= mmax; m++) {
			tprint("O[%i] = fma(ax0, O[%i], O[%i]);\n", index(n + 1, m), index(n, m), index(n + 1, m));
		}
		tprint("O[%i] = ax0 * O[%i];\n", index(n + 1, n), index(n, n));
		if (n != P - 1) {
			for (int m = 0; m <= n; m++) {
				const double c0 = -(double((n + 1) * (n + 1)) - double(m * m));
				tprint("O[%i] = TCAST(%.20e) * r2inv * O[%i];\n", index(n + 2, m), c0, index(n, m));
			}
		}
	}
}

int M2L_allrot(int P, int Q, int rot) {
	std::string mname = std::string(rot == 0 ? "M" : "Min") + ((scaled) ? "s" : "");
	std::string name = print2str("M2%cr%i", P == Q ? 'L' : 'P', rot);
	if (Q > 1) {
		func_header(name.c_str(), P, true, true, true, true, true, "", "Lout", EXP, mname.c_str(), CMUL, "dx", VEC3);
	} else {
		func_header(name.c_str(), P, true, true, true, true, true, "", "f", FORCE, mname.c_str(), CMUL, "dx", VEC3);
	}
	struct fentry_t {
		std::string name;
		std::string body;
		flops_t flops;
	};
	fentry_t functors[4][P + 1];
	open_timer(name);
	init_real("tmp0");
	init_real("tmp1");
	bool minit = false;
	if (rot != 0) {
		tprint("multipole<%s, %i> M_st;\n", type.c_str(), P);
		tprint("multipole<%s, %i> M0_st;\n", type.c_str(), P);
		tprint("T* M(M_st.data());\n");
		if (rot != 1) {
			tprint("T* M0(M0_st.data());\n");
		}
		if (Q == P) {
			tprint("expansion<%s, %i> L0_st;\n", type.c_str(), P);
			tprint("T* L0(L0_st.data());\n");
			if (rot == 1) {
				tprint("T* ptr;\n");
			}
		}
	}
	if (Q == 1) {
		init_reals("La", 4);
		if (rot != 0) {
			init_reals("Lb", 4);
		}
		tprint("T* L=La;\n");
		if (rot != 0) {
			tprint("T* L0=Lb;\n");
		}
	}
	if (Q > 1) {
		if (scaled) {
			tprint("expansion<T, %i> L_st(Lout_st.scale());\n", P);
		} else {
			tprint("expansion<T, %i> L_st;\n", P);
		}
		tprint("T* L=L_st.data();\n");
	}
	if (rot == 1) {
		tprint("detail::expansion_xz<%s, %i> O_st;\n", type.c_str(), P);
	} else if (rot == 0) {
		tprint("expansion<%s, %i> O_st;\n", type.c_str(), P);
	}
	if (rot != 2) {
		tprint("T* O(O_st.data());\n", type.c_str(), P);
	}
	init_real("r2");
	init_real("r2inv");
	init_real("ax0");
	init_real("ax1");
	init_real("ax2");
	init_real("ay0");
	init_real("ay1");
	init_real("ay2");
	init_real("R2");
	int N = std::max(P - 1, Q);
	if (rot > 0) {
		if (rot == 2) {
			init_reals("r0A", N);
			init_reals("ipA", N);
			init_reals("inA", N);
			init_reals("r0B", N);
			init_reals("ipB", N);
			init_reals("inB", N);
			tprint("T* r0=r0A;\n");
			tprint("T* ip=ipA;\n");
			tprint("T* in=inA;\n");
		} else {
			init_reals("r0", N);
			init_reals("ipA", N);
			init_reals("inA", N);
			tprint("T* ip=ipA;\n");
			tprint("T* in=inA;\n");
		}
		init_real("Rinv");
		init_real("R");
		init_real("Rzero");
		init_real("rzero");
		init_real("sinphi");
		init_real("cosphi");
		init_real("r2przero");
		if (rot == 2) {
			init_real("rinv");
			init_reals("A", P + 1);
		}
	}
	if (scaled) {
		const char *name = rot == 0 ? "M" : "Min";
		tprint("multipole<%s, %i> %s_st(%s_st);\n", type.c_str(), P, name, mname.c_str());
		tprint("T* %s=%s_st.data();\n", name, name);
		if (Q != 1) {
			tprint("tmp1 = TCAST(1) / Lout_st.scale();\n");
			tprint("x *= tmp1;\n");
			tprint("y *= tmp1;\n");
			tprint("z *= tmp1;\n");
			tprint("%s_st.rescale(L_st.scale());\n", name, name);
		} else {
			tprint("tmp1 = TCAST(1) / %s_st.scale();\n", mname.c_str());
			tprint("x *= tmp1;\n");
			tprint("y *= tmp1;\n");
			tprint("z *= tmp1;\n");
		}
	}

	if (rot > 0) {
		tprint("R2 = fma(x, x, y * y);\n");
		tprint("Rzero = TCONVERT( R2 < TCAST(%.20e) );\n", tiny());
		tprint("r2 = fma(z, z, R2);\n");
		tprint("rzero = TCONVERT( r2 < TCAST(%.20e) );\n", tiny());
		tprint("tmp1 = R2 + Rzero;\n");
		tprint("Rinv = rsqrt(tmp1);\n");
		tprint("R = (TCAST(1) - Rzero) / Rinv;\n");
		tprint("r2przero = (r2 + rzero);\n");
		if (rot == 1) {
			tprint("r2inv = TCAST(1) / (r2przero);\n");
			tprint("cosphi = fma(x, Rinv, Rzero);\n");
			tprint("sinphi = -y * Rinv;\n");
		} else {
			tprint("rinv = rsqrt(r2przero);\n");
			tprint("cosphi = y * Rinv;\n");
			tprint("sinphi = fma(x, Rinv, Rzero);\n");
		}
	}
	if (rot != 0) {
		tprint("r0[0] = cosphi;\n");
		tprint("ip[0] = sinphi;\n");
		tprint("in[0] = -sinphi;\n");
		for (int m = 1; m < N; m++) {
			tprint("r0[%i] = in[%i] * sinphi;\n", m, m - 1);
			tprint("ip[%i] = ip[%i] * cosphi;\n", m, m - 1);
			tprint("r0[%i] = fma(r0[%i], cosphi, r0[%i]);\n", m, m - 1, m);
			tprint("ip[%i] = fma(r0[%i], sinphi, ip[%i]);\n", m, m - 1, m);
			tprint("in[%i] = -ip[%i];\n", m, m);
		}
	}
	if (rot == 1) {
		z_rot2(P - 1, "M", "Min", FULL, P != Q ? "M2P" : "M2L", false, true);
	} else if (rot == 2) {
		z_rot2(P - 1, "M0", "Min", PRE1, P != Q ? "M2P" : "M2L", false, true);
		xz_swap2(P - 1, "M", "M0", false, PRE1, P != Q ? "M2P" : "M2L");
		tprint("cosphi = fma(z, rinv, rzero);\n");
		tprint("sinphi = -R * rinv;\n");
		tprint("r0=r0B;\n");
		tprint("in=inB;\n");
		tprint("ip=ipB;\n");
		tprint("r0[0] = cosphi;\n");
		tprint("ip[0] = sinphi;\n");
		tprint("in[0] = -sinphi;\n");
		for (int m = 1; m < N; m++) {
			tprint("r0[%i] = in[%i] * sinphi;\n", m, m - 1);
			tprint("ip[%i] = ip[%i] * cosphi;\n", m, m - 1);
			tprint("r0[%i] = fma(r0[%i], cosphi, r0[%i]);\n", m, m - 1, m);
			tprint("ip[%i] = fma(r0[%i], sinphi, ip[%i]);\n", m, m - 1, m);
			tprint("in[%i] = -ip[%i];\n", m, m);
		}
		z_rot2(P - 1, "M0", "M", PRE2, P != Q ? "M2P" : "M2L", false, true);
		tprint("in=ipB;\n");
		tprint("ip=inB;\n");
		xz_swap2(P - 1, "M", "M0", false, PRE2, P != Q ? "M2P" : "M2L");
	}

	if (rot == 0) {
		greens_body(P);
	} else if (rot == 1) {
		greens_xz_body(P);
	}
	if (rot == 1 && P == Q) {
		tprint("ptr=L0;\n");
		tprint("L0=L;\n");
		tprint("L=ptr;\n");
	}
	if (rot != 2) {
		mg2l_body(P, Q, rot == 1, rot == 0 ? lindex : cindex);
	} else {
		tprint("A[0] = -rinv;\n");
		tprint("A[1] = rinv * rinv;\n");
		for (int n = 2; n <= P; n++) {
			const int i = (n - 1) / 2;
			const int j = n - 1 - i;
			tprint("A[%i] = A[%i] * A[%i];\n", n, i, j);
		}
		for (int n = 2; n <= P; n++) {
			tprint("A[%i] *= TCAST(%.20e);\n", n, factorial(n));
		}
		bool first[(Q + 1) * (Q + 1)];
		for (int n = 0; n < (Q + 1) * (Q + 1); n++) {
			first[n] = true;
		}
		struct entry_t {
			int l;
			int m;
			int r;
		};
		std::vector<entry_t> cmds;
		for (int n = 0; n <= Q; n++) {
			for (int m = 0; m <= n; m++) {
				const int maxk = std::min(P - n, P - 1);
				for (int k = m; k <= maxk; k++) {
					if (nodip && k == 1) {
						continue;
					}
					entry_t cmd;
					cmd.l = lindex(n, m);
					cmd.m = mindex(k, m);
					cmd.r = n + k;
					cmds.push_back(cmd);
					if (m != 0) {
						cmd.l = lindex(n, -m);
						cmd.m = mindex(k, -m);
						cmds.push_back(cmd);
					}
				}
			}
		}
		std::sort(cmds.begin(), cmds.end(), [](entry_t a, entry_t b) {
			if (a.m < b.m) {
				return true;
			} else if (a.m > b.m) {
				return false;
			} else {
				return a.l < b.l;
			}
		});
		for (auto cmd : cmds) {
			if (first[cmd.l]) {
				first[cmd.l] = false;
				tprint("L[%i] = M[%i] * A[%i];\n", cmd.l, cmd.m, cmd.r);
			} else {
				tprint("L[%i] = fma(M[%i], A[%i], L[%i]);\n", cmd.l, cmd.m, cmd.r, cmd.l);
			}
		}
		for (int n = 0; n <= Q; n++) {
			for (int m = -n; m <= n; m++) {
				if (abs(m) % 2 != 0) {
					if (!first[lindex(n, m)]) {
						tprint("L[%i] = -L[%i];\n", lindex(n, m), lindex(n, m));
					}
				}
			}
		}
	}
	if (rot == 1) {
		tprint("ip=inA;\n");
		tprint("in=ipA;\n");
		if (nodip) {
			z_rot2(Q, "L0", "L", ((Q == P) || (Q == 1 && P == 2)) ? XZ2 : FULL, P != Q ? "M2P" : "M2L", false, false);
		} else {
			z_rot2(Q, "L0", "L", Q == P ? XZ1 : FULL, P != Q ? "M2P" : "M2L", false, false);
		}
	} else if (rot == 2) {
		xz_swap2(Q, "L0", "L", true, P == Q ? POST1 : FULL, P != Q ? "M2P" : "M2L");
		z_rot2(Q, "L", "L0", P == Q ? POST1 : FULL, P != Q ? "M2P" : "M2L", false, false);
		xz_swap2(Q, "L0", "L", true, P == Q ? POST2 : FULL, P != Q ? "M2P" : "M2L");
		tprint("r0=r0A;\n");
		tprint("in=ipA;\n");
		tprint("ip=inA;\n");
		z_rot2(Q, "L", "L0", P == Q ? POST2 : FULL, P != Q ? "M2P" : "M2L", false, false);
	}
	if (Q == 1) {
		if (rot == 1) {
			tprint("L=L0;\n");
		}
		if (scaled) {
			tprint("tmp0 = TCAST(1) / %s_st.scale();\n", mname.c_str());
			tprint("tmp1 = -tmp0 * tmp0;\n");
			tprint("f.potential = fma(tmp0, L[0], f.potential);\n");
			tprint("f.force[0] = fma(tmp1, L[3], f.force[0]);\n");
			tprint("f.force[1] = fma(tmp1, L[1], f.force[1]);\n");
			tprint("f.force[2] = fma(tmp1, L[2], f.force[2]);\n");
		} else {
			tprint("f.potential += L[0];\n");
			tprint("f.force[0] -= L[3];\n");
			tprint("f.force[1] -= L[1];\n");
			tprint("f.force[2] -= L[2];\n");
		}
	}
	if (Q > 1) {
		for (int n = 0; n < exp_sz(P); n++) {
			tprint("Lout[%i] += L[%i];\n", n, n);
		}
	}
	close_timer();
	deindent();
	tprint("}");

	auto flops = get_running_flops();
	flops += rescale_flops(P - 1);
	tprint("return %i;\n", flops.load());
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	timing_body += print2str("\"M2%c\", %i, %i, %i, 0.0, 0}", Q == P ? 'L' : 'P', P, rot, flops.load());
	TAB0();
	return get_running_flops().load();
}

std::string flags_header(const char *op, int P, int best) {
	std::string str;
	//	str += print2str("static const int best_rot = operator_best_rotation(%i, \"%s\", \"%s\");\n", P, type.c_str(), op);
	str += "\tint rot = -1;\n";
	str += "\tif( flags & sfmmWithBestOptimization ) {\n";
	str += "\t\tflags &= ~sfmmWithBestOptimization;\n";
	str += "\t\trot = " + std::to_string(best) + ";\n";
	str += "\t}\n";
	str += "\tif( rot >= 0 ) {\n";
	str += "\t\tflags |= (rot == 1 ? sfmmWithSingleRotationOptimization : (rot == 2 ? sfmmWithDoubleRotationOptimization : 0));\n";
	str += "\t}\n";
	return str;
}

std::string flags_choose3(std::string name, std::string name2) {
	std::string str;
	str += "\tif( flags & sfmmWithSingleRotationOptimization ) {\n";
	str += print2str("\t\treturn %sr1(%s, M0_st, dx, flags);\n", name.c_str(), name2.c_str());
	str += "\t} else if( flags & sfmmWithDoubleRotationOptimization ) {\n";
	str += print2str("\t\treturn %sr2(%s, M0_st, dx, flags);\n", name.c_str(), name2.c_str());
	str += "\t} else {\n";
	str += print2str("\t\treturn %sr0(%s, M0_st, dx, flags);\n", name.c_str(), name2.c_str());
	str += "\t}\n";
	return str;
}
std::string flags_choose2(std::string name, std::string name2) {
	std::string str;
	str += "\tif( flags & sfmmWithSingleRotationOptimization ) {\n";
	str += print2str("\t\treturn %sr1(%s, M0_st, dx, flags);\n", name.c_str(), name2.c_str());
	str += "\t} else {\n";
	str += print2str("\t\treturn %sr0(%s, M0_st, dx, flags);\n", name.c_str(), name2.c_str());
	str += "\t}\n";
	return str;
}

void M2L(int P, int Q) {
	std::string str;
	int flops[3];
	int bestrot;
	int bestflops = 1000000000;
	for (int rot = 0; rot < 3; rot++) {
		flops[rot] = M2L_allrot(P, Q, rot);
		if (flops[rot] < bestflops) {
			bestflops = flops[rot];
			bestrot = rot;
		}
	}
	if (Q > 1) {
		func_header("M2L", P, true, false, false, false, true, "", "L0", EXP, "M0", CMUL, "dx", VEC3);
		str = flags_header("M2L", P, bestrot);
		str += flags_choose3("M2L", "L0_st");
	} else {
		func_header("M2P", P, true, false, false, false, true, "", "f", FORCE, "M0", CMUL, "dx", VEC3);
		// str += print2str("static const int best_rot = operator_best_rotation(%i, \"%s\", \"%s\");\n", P, type.c_str(), "M2P");
		str += "\tint rot = -1;\n";
		str += "\tif( flags & sfmmWithBestOptimization ) {\n";
		str += "\t\tflags &= ~sfmmWithBestOptimization;\n";
		str += "\t\trot = " + std::to_string(bestrot) + ";\n";
		str += "\t}\n";
		str += "\tif( rot >= 0 ) {\n";
		str += "\t\tflags |= (rot == 1 ? sfmmWithSingleRotationOptimization : (rot == 2 ? sfmmWithDoubleRotationOptimization : 0));\n";
		str += "\t}\n";
		str += flags_choose3("M2P", "f");
	}
	tprint(str.c_str());
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	TAB0();
}
std::string M2L_ewald(int P) {
	auto fname = func_header("M2L_ewald", P, true, false, false, true, true, "", "L", EXP, (scaled) ? "Min" : "M", CMUL, "dx", VEC3);
	reset_running_flops();
	tprint("expansion<%s, %i> G_st;\n", type.c_str(), P);
	reset_running_flops();
	tprint("T* G=G_st.data();\n");
	tprint("T* L=L_st.data();\n");
	if (scaled) {
		tprint("multipole<%s, %i> M_st(Min_st);\n", type.c_str(), P);
		tprint("const T* M=M_st.data();\n");
		tprint("const %s scale = L_st.scale();\n", base_rtype[typenum].c_str());
		tprint("M_st.rescale(%s(1));\n", base_rtype[typenum].c_str());
		tprint("L_st.rescale(%s(1));\n", base_rtype[typenum].c_str());
	} else {
		tprint("const T* M=M_st.data();\n");
	}
	tprint("int flops=greens_ewald(G_st, dx);\n");
	tprint("flops+=MG2L(L_st, M_st, G_st);\n");
	if (scaled) {
		tprint("L_st.rescale(scale);\n");
	}
	auto flops = get_running_flops();
	flops += rescale_flops(P - 1);
	flops += rescale_flops(P);
	flops += rescale_flops(P);

	tprint("return flops+%i;\n", flops.load());
	deindent();
	tprint("} else {\n");
	indent();
	tprint("expansion<%s, %i> L_st;\n", type.c_str(), P);
	tprint("multipole<%s, %i> M_st;\n", type.c_str(), P);
	tprint("expansion<%s, %i> O_st;\n", type.c_str(), P);
	tprint("return greens_ewald(O_st, vec3<T>(T(0), T(0), T(0)), flags) + MG2L(L_st, M_st, O_st) + %i;\n", flops.load());
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	TAB0();
	return fname;
}

std::string M2P_ewald(int P) {
	auto fname = func_header("M2P_ewald", P, true, false, false, true, true, "", "f", FORCE, (scaled) ? "Min" : "M", CMUL, "dx", VEC3);
	reset_running_flops();
	tprint("expansion<%s, %i> O_st;\n", type.c_str(), P);
	tprint("T* O=O_st.data();\n");
	reset_running_flops();
	init_reals("L", 4);
	init_real("tmp0");
	init_real("tmp1");
	if (scaled) {
		tprint("multipole<%s, %i> M_st(Min_st);\n", type.c_str(), P);
		tprint("M_st.rescale(%s(1));\n", base_rtype[typenum].c_str());
	}
	tprint("const T* M=M_st.data();\n");
	tprint("int flops = greens_ewald(O_st, dx);\n");
	mg2l_body(P, 1);
	if (P > 2 && periodic) {
		tprint("L[%i] = fma(TCAST(-0.5) * O_st.trace2(), M_st.trace2(), L[%i]);\n", lindex(0, 0), lindex(0, 0));
	}
	if (P > 1 && periodic) {
		if (!nodip) {
			tprint("L[%i] = fma(TCAST(-2) * O_st.trace2(), M[%i], L[%i]);\n", lindex(1, -1), mindex(1, -1), lindex(1, -1));
			tprint("L[%i] -= O_st.trace2() * M[%i];\n", lindex(1, +0), lindex(1, +0), mindex(1, +0));
			tprint("L[%i] = fma(TCAST(-2) * O_st.trace2(), M[%i], L[%i]);\n", lindex(1, +1), mindex(1, +1), lindex(1, +1));
		}
	}
	tprint("f.potential += L[0];\n");
	tprint("f.force[0] -= L[3];\n");
	tprint("f.force[1] -= L[1];\n");
	tprint("f.force[2] -= L[2];\n");
	auto flops = get_running_flops();
	flops += rescale_flops(P - 1);
	tprint("return flops+%i;\n", flops.load());
	deindent();
	tprint("} else {\n");
	indent();
	tprint("expansion<%s, %i> O_st;\n", type.c_str(), P);
	tprint("return greens_ewald(O_st, vec3<T>(T(0), T(0), T(0)), flags) + %i;\n", flops.load());
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	TAB0();
	return fname;
}

std::string P2L_ewald(int P) {
	auto fname = func_header("P2L_ewald", P, true, false, false, true, true, "", "L0", EXP, "m", LIT, "dx", VEC3);
	reset_running_flops();
	init_real("tmp1");
	reset_running_flops();
	if (scaled) {
		tprint("expansion<T,%i> L_st(L0_st.scale());\n", P);
		tprint("L_st.rescale(%s(1));\n", base_rtype[typenum].c_str());
		tprint("const %s scale = L0_st.scale();\n", base_rtype[typenum].c_str());
	} else {
		tprint("expansion<T,%i> L_st;\n", P);
	}
	tprint("T* L=L_st.data();\n");
	tprint("T* L0=L0_st.data();\n");
	tprint("int flops = greens_ewald(L_st, dx);\n");
	if (scaled) {
		tprint("L_st.rescale(scale);\n");
	}
	for (int i = 0; i < exp_sz(P); i++) {
		tprint("L0[%i] = fma(m, L[%i], L0[%i]);\n", i, i, i);
	}
	tprint("L0_st.trace2() = fma(m, L_st.trace2(), L0_st.trace2());\n");
	auto flops = get_running_flops();
	flops += rescale_flops(P);
	flops += rescale_flops(P);
	tprint("return flops+%i;\n", flops.load());
	deindent();
	tprint("} else {\n");
	indent();
	tprint("expansion<%s, %i> O_st;\n", type.c_str(), P);
	tprint("return greens_ewald(O_st, vec3<T>(T(0), T(0), T(0)), flags) + %i;\n", flops.load());
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	TAB0();
	return fname;
}

void func_closer(int P, std::string name, bool pub) {
	deindent();
	tprint("}\n");
	tprint("return %i;\n", get_running_flops().load());
	flops_map[P][type + name] = get_running_flops();
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	if (!pub) {
		tprint("}\n");
	}
	tprint("\n");
}

void M2M_z(int P, int dir) {
	tprint("Y[0] = z;\n");
	tprint("Y[1] = z * z;\n");
	for (int i = 2; i < P - 1; i++) {
		const int n = (i - 1) / 2;
		const int m = (i - 1) - n;
		tprint("Y[%i] = Y[%i] * Y[%i];\n", i, n, m);
	}
	for (int i = 1; i < P - 1; i++) {
		tprint("Y[%i] *= TCAST(%0.20e);\n", i, 1.0 / factorial(i + 1));
	}
	for (int n = P - 1; n >= 0; n--) {
		std::vector<entry_t> cmds;
		for (int m = -n; m <= n; m++) {
			for (int k = 1; k <= n; k++) {
				entry_t e;
				if (abs(m) > n - k) {
					continue;
				}
				if (nodip && dir > 0) {
					if (n - k == 1) {
						continue;
					} else if (n == 1) {
						e.m = mindex(n - k, m);
						e.o = k - 1;
						e.l = -1;
						cmds.push_back(e);
						continue;
					}
				}
				if (nodip && dir < 0) {
					if (n - k == 1) {
						if (m == 1) {
							e.l = mindex(n, m);
							e.m = -1;
							e.o = k - 1;
							cmds.push_back(e);
							continue;
						} else {
							continue;
						}
					} else if (n == 1) {
						continue;
					}
				}
				e.l = mindex(n, m);
				e.m = mindex(n - k, m);
				e.o = k - 1;
				cmds.push_back(e);
			}
		}
		std::sort(cmds.begin(), cmds.end(), cmp);
		for (const auto &e : cmds) {
			if (e.l == -1) {
				tprint("Md = fma(Y[%i], M[%i], Md);\n", e.o, e.m);
			} else if (e.m == -1) {
				tprint("M[%i] = fma(Y[%i], Md, M[%i]);\n", e.l, e.o, e.l);
			} else {
				tprint("M[%i] = fma(Y[%i], M[%i], M[%i]);\n", e.l, e.o, e.m, e.l);
			}
		}
		cmds.resize(0);
	}
}

void regular_harmonic_full(int P) {
	tprint("Y[0] = TCAST(1);\n");
	for (int m = 1; m <= P; m++) {
		if (m - 1 > 0) {
			tprint("ax0 = Y[%i] * TCAST(%.20e);\n", lindex(m - 1, m - 1), 1.0 / (2.0 * m));
			tprint("ay0 = Y[%i] * TCAST(%.20e);\n", lindex(m - 1, -(m - 1)), 1.0 / (2.0 * m));
			tprint("Y[%i] = x * ax0 - y * ay0;\n", lindex(m, m));
			tprint("Y[%i] = fma(y, ax0, x * ay0);\n", lindex(m, -m));
		} else {
			tprint("Y[%i] = x * TCAST(%.20e);\n", lindex(m, m), 1.0 / (2.0 * m));
			tprint("Y[%i] = y * TCAST(%.20e);\n", lindex(m, -m), 1.0 / (2.0 * m));
		}
	}
	const double c0 = -0.25;
	const double c1 = double(1) / (double((1) * (1)));
	if (2 <= P) {
		tprint("Y[%i] = TCAST(-0.25) * r2;\n", lindex(2, 0));
	}
	if (1 <= P) {
		tprint("Y[%i] = z;\n", lindex(1, 0));
	}
	for (int n = 1; n < P; n++) {
		const double c0 = -double(1) / (double((n + 2) * (n + 2)));
		const double c1 = double(2 * n + 1) / (double((n + 1) * (n + 1)));
		if (n + 2 <= P) {
			tprint("Y[%i] = TCAST(%.20e) * r2 * Y[%i];\n", lindex(n + 2, 0), c0, lindex(n, 0));
		}
		tprint("Y[%i] = fma(TCAST(%.20e) * z, Y[%i], Y[%i]);\n", lindex(n + 1, 0), c1, lindex(n, 0), lindex(n + 1, 0));
		for (int m = 1; m <= n; m++) {
			const double c0 = -double(1) / (double((n + 2) * (n + 2)) - double(m * m));
			const double c1 = double(2 * n + 1) / (double((n + 1) * (n + 1)) - double(m * m));
			if (n + 2 <= P) {
				tprint("ax0 = TCAST(%.20e) * r2;\n", c0);
			}
			if (n + 2 <= P) {
				tprint("Y[%i] = ax0 * Y[%i];\n", lindex(n + 2, m), lindex(n, m));
				tprint("Y[%i] = ax0 * Y[%i];\n", lindex(n + 2, -m), lindex(n, -m));
			}
			if (n == m) {
				tprint("Y[%i] = z * Y[%i];\n", lindex(n + 1, m), lindex(n, m));
				tprint("Y[%i] = z * Y[%i];\n", lindex(n + 1, -m), lindex(n, -m));
			} else {
				tprint("ay0 = TCAST(%.20e) * z;\n", c1);
				tprint("Y[%i] = fma(ay0, Y[%i], Y[%i]);\n", lindex(n + 1, m), lindex(n, m), lindex(n + 1, m));
				tprint("Y[%i] = fma(ay0, Y[%i], Y[%i]);\n", lindex(n + 1, -m), lindex(n, -m), lindex(n + 1, -m));
			}
		}
	}
}

void regular_harmonic_xy(int P) {
	tprint("Y[0] = TCAST(1);\n");
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
			if (m - 1 > 0) {
				tprint("ax0 = Y[%i] * TCAST(%.20e);\n", xyindex(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("ay0 = Y[%i] * TCAST(%.20e);\n", xyindex(m - 1, -(m - 1)), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax0 - y * ay0;\n", xyindex(m, m));
				tprint("Y[%i] = fma(y, ax0, x * ay0);\n", xyindex(m, -m));
			} else {
				tprint("Y[%i] = x * TCAST(%.20e);\n", xyindex(m, m), 1.0 / (2.0 * m));
				tprint("Y[%i] = y * TCAST(%.20e);\n", xyindex(m, -m), 1.0 / (2.0 * m));
			}
		}
	}
	const double c0 = -0.25;
	const double c1 = double(1) / (double((1) * (1)));
	if (2 <= P) {
		tprint("Y[%i] = TCAST(-0.25) * R2;\n", xyindex(2, 0), xyindex(0, 0));
	}
	for (int n = 1; n < P; n++) {
		const double c0 = -double(1) / (double((n + 2) * (n + 2)));
		const double c1 = double(2 * n + 1) / (double((n + 1) * (n + 1)));
		if (n + 2 <= P && (n % 2 == 0)) {
			tprint("Y[%i] = TCAST(%.20e) * R2 * Y[%i];\n", xyindex(n + 2, 0), c0, xyindex(n, 0));
		}
		for (int m = 1; m <= n; m++) {
			if (!(n % 2 == m % 2)) {
				continue;
			}
			const double c0 = -double(1) / (double((n + 2) * (n + 2)) - double(m * m));
			const double c1 = double(2 * n + 1) / (double((n + 1) * (n + 1)) - double(m * m));
			if (n + 2 <= P) {
				tprint("ax0 = TCAST(%.20e) * R2;\n", c0);
			}
			if (n + 2 <= P) {
				tprint("Y[%i] = ax0 * Y[%i];\n", xyindex(n + 2, m), xyindex(n, m));
				tprint("Y[%i] = ax0 * Y[%i];\n", xyindex(n + 2, -m), xyindex(n, -m));
			}
		}
	}
}

int M2M_allrot(int P, int rot) {
	auto index = mindex;
	int flops = 0;
	const auto name = std::string("M2Mr") + std::to_string(rot);
	func_header(name.c_str(), P, true, true, true, true, true, "", "M", MUL, "dx", VEC3);
	open_timer(name);
	init_real("tmp1");
	init_real("ax0");
	init_real("ay0");
	init_real("ax1");
	init_real("ay1");
	init_real("ax2");
	init_real("ay2");
	init_real("R2");
	if (periodic || rot == 0) {
		init_real("r2");
	}
	if (nodip) {
		init_real("Md");
	}
	if (rot == 0) {
		init_reals("Y", exp_sz(P - 1));
	} else if (rot == 1) {
		init_reals("Y", std::max(P - 1, xyexp_sz(P - 1)));
	} else if (rot == 2) {
		init_real("tmp0");
		init_real("R");
		init_real("Rzero");
		init_real("Rinv");
		init_real("cosphi");
		init_real("sinphi");
		init_reals("A", 2 * (P - 1) + 1);
		tprint("T* const Y=A;\n");
		init_reals("r0A", P - 1);
		init_reals("ipA", P - 1);
		init_reals("inA", P - 1);
		tprint("T* r0=r0A;\n");
		tprint("T* ip=ipA;\n");
		tprint("T* in=inA;\n");
	}
	if (rot == 2) {
		init_reals("Ma", mul_sz(P));
		tprint("T* M0=Ma;\n");
	}
	if (scaled) {
		tprint("tmp1 = TCAST(1) / M_st.scale();\n");
		tprint("x *= tmp1;\n");
		tprint("y *= tmp1;\n");
		tprint("z *= tmp1;\n");
	}
	reset_running_flops();
	tprint("x = -x;\n");
	tprint("y = -y;\n");
	tprint("z = -z;\n");
	tprint("R2 = fma(x, x, y * y);\n");
	if (periodic || rot == 0) {
		tprint("r2 = fma(z, z, R2);\n");
	}
	std::function<int(int, int)> yindex;
	if (P > 2 && periodic && !nodip) {
		tprint("M_st.trace2() = fma(TCAST(4) * x, M[%i], M_st.trace2());\n", mindex(1, 1));
		tprint("M_st.trace2() = fma(TCAST(4) * y, M[%i], M_st.trace2());\n", mindex(1, -1));
		tprint("M_st.trace2() = fma(TCAST(2) * z, M[%i], M_st.trace2());\n", mindex(1, 0));
	}
	if (P > 2 && periodic) {
		tprint("M_st.trace2() = fma(r2, M[%i], M_st.trace2());\n", mindex(0, 0));
	}
	if (rot == 0) {
		regular_harmonic_full(P - 1);
		yindex = lindex;
	} else if (rot == 1) {
		M2M_z(P, +1);
		regular_harmonic_xy(P - 1);
		yindex = xyindex;
	}
	flops += get_running_flops().load();
	reset_running_flops();
	if (rot == 2) {
		tprint("Rzero = TCONVERT( R2 < TCAST(%.20e) );\n", tiny());
		tprint("tmp1 = R2 + Rzero;\n");
		tprint("Rinv = rsqrt(tmp1);\n");
		tprint("R = (TCAST(1) - Rzero) / Rinv;\n");
		tprint("cosphi = fma(x, Rinv, Rzero);\n");
		tprint("sinphi = -y * Rinv;\n");
		M2M_z(P, +1);
		z_rot2(P - 1, "M0", "M", FULL, "M2M", true, true);
		xz_swap2(P - 1, "M", "M0", false, FULL, "M2M");
		if (nodip) {
			tprint("Md *= TCAST(0.5);\n");
		}
		tprint("z = R;\n");
		M2M_z(P, -1);
		xz_swap2(P - 1, "M0", "M", false, FULL, "M2M");
		tprint("in=ipA;\n");
		tprint("ip=inA;\n");
		z_rot2(P - 1, "M", "M0", FULL, "M2M", false, true);
	} else {
		std::vector<entry_t> pos;
		std::vector<entry_t> neg;
		for (int n = P - 1; n >= 0; n--) {
			if (nodip && n == 1) {
				continue;
			}
			for (int m = 0; m <= n; m++) {
				const auto add_work = [&pos, &neg, n](int sgn, int m, int mstr, int gstr) {
					entry_t entry;
					entry.l = mindex(n, m);
					entry.m = mstr;
					entry.o = gstr;
					if (sgn == 1) {
						pos.push_back(entry);
					} else {
						neg.push_back(entry);
					}
				};
				for (int k = 1; k <= n; k++) {
					if (nodip && n - k == 1 && rot == 0) {
						continue;
					}
					const int lmin = std::max(-k, m - n + k);
					const int lmax = std::min(k, m + n - k);
					for (int l = -k; l <= k; l++) {
						if (rot == 1 && abs(l) % 2 != k % 2) {
							continue;
						}
						int mxstr = -99;
						int mystr = -99;
						int gxstr = -99;
						int gystr = -99;
						int mxsgn = 1;
						int mysgn = 1;
						int gxsgn = 1;
						int gysgn = 1;
						if (abs(m - l) > n - k) {
							continue;
						}
						if (-abs(m - l) < k - n) {
							continue;
						}
						if (n - k == 1 && nodip) {
							if (m - l > 0) {
								continue;
							} else if (m - l < 0) {
								continue;
							} else {
								mxstr = -1;
							}
						} else {
							if (m - l > 0) {
								mxstr = mindex(n - k, abs(m - l));
								mystr = mindex(n - k, -abs(m - l));
							} else if (m - l < 0) {
								if (abs(m - l) % 2 == 0) {
									mxstr = mindex(n - k, abs(m - l));
									mystr = mindex(n - k, -abs(m - l));
									mysgn = -1;
								} else {
									mxstr = mindex(n - k, abs(m - l));
									mystr = mindex(n - k, -abs(m - l));
									mxsgn = -1;
								}
							} else {
								mxstr = mindex(n - k, 0);
							}
						}
						if (l > 0) {
							gxstr = yindex(k, abs(l));
							gystr = yindex(k, -abs(l));
						} else if (l < 0) {
							if (abs(l) % 2 == 0) {
								gxstr = yindex(k, abs(l));
								gystr = yindex(k, -abs(l));
								gysgn = -1;
							} else {
								gxstr = yindex(k, abs(l));
								gystr = yindex(k, -abs(l));
								gxsgn = -1;
							}
						} else {
							gxstr = yindex(k, 0);
						}
						add_work(mxsgn * gxsgn, m, mxstr, gxstr);
						if (gystr != -99 && mystr != -99) {
							add_work(-mysgn * gysgn, m, mystr, gystr);
						}
						if (m > 0) {
							if (gystr != -99) {
								add_work(mxsgn * gysgn, -m, mxstr, gystr);
							}
							if (mystr != -99) {
								add_work(mysgn * gxsgn, -m, mystr, gxstr);
							}
						}
					}
				}
			}
			std::sort(pos.begin(), pos.end(), cmp);
			std::sort(neg.begin(), neg.end(), cmp);
			std::vector<bool> used(P * P, false);
			for (const auto &e : neg) {
				if (!used[e.l]) {
					tprint("M[%i] = -M[%i];\n", e.l, e.l);
					used[e.l] = true;
				}
			}
			for (const auto &e : neg) {
				if (e.m == -1) {
					tprint("M[%i] = fma(Y[%i], Md, M[%i]);\n", e.l, e.o, e.l);
				} else {
					tprint("M[%i] = fma(Y[%i], M[%i], M[%i]);\n", e.l, e.o, e.m, e.l);
				}
			}
			used = std::vector<bool>(P * P, false);
			for (const auto &e : neg) {
				if (!used[e.l]) {
					tprint("M[%i] = -M[%i];\n", e.l, e.l);
					used[e.l] = true;
				}
			}
			for (const auto &e : pos) {
				if (e.m == -1) {
					tprint("M[%i] = fma(Y[%i], Md, M[%i]);\n", e.l, e.o, e.l);
				} else {
					tprint("M[%i] = fma(Y[%i], M[%i], M[%i]);\n", e.l, e.o, e.m, e.l);
				}
			}
			neg.resize(0);
			pos.resize(0);
		}
	}
	close_timer();
	deindent();
	tprint("}\n");
	flops += get_running_flops().load();
	reset_running_flops();
	tprint("return %i;\n", flops);
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	timing_body += print2str("\"M2M\", %i, %i, %i, 0.0, 0}", P, rot, flops);
	TAB0();
	return flops;
}

void L2L_z(int P, int Q, const char *var = "z", const char *src = "") {
	const char *two = P == Q ? "" : "2";
	tprint("Y[0] = %s;\n", var);
	tprint("Y[1] = %s * %s;\n", var, var);
	for (int i = 2; i < P; i++) {
		const int n = (i - 1) / 2;
		const int m = (i - 1) - n;
		tprint("Y[%i] = Y[%i] * Y[%i];\n", i, n, m);
	}
	for (int i = 1; i < P; i++) {
		tprint("Y[%i] *= TCAST(%0.20e);\n", i, 1.0 / factorial(i + 1));
	}
	for (int n = 0; n <= Q; n++) {
		std::vector<entry_t> cmds;
		for (int m = -n; m <= n; m++) {
			tprint_new_chain();
			for (int k = 1; k <= P - n; k++) {
				if (abs(m) > n + k) {
					continue;
				}
				entry_t e;
				e.m = lindex(n + k, m);
				e.l = lindex(n, m);
				e.o = k - 1;
				cmds.push_back(e);
				//				tprint("L%s[%i] = fma(Y[%i], L[%i], L%s[%i]);\n", two, lindex(n, m), k - 1, lindex(n + k, m), two, lindex(n,
				//m));
			}
		}
		std::sort(cmds.begin(), cmds.end(), cmp);
		for (const auto &e : cmds) {
			tprint("L%s[%i] = fma(Y[%i], L%s[%i], L%s[%i]);\n", two, e.l, e.o, src, e.m, two, e.l);
		}
	}
}

int P2P_ewald() {
	func_header("P2P_ewald", 0, true, false, false, false, false, "", "f", FORCE, "m", LIT, "dx", VEC3);
	reset_running_flops();
	int R2, H2;
	init_real("x");
	init_real("y");
	init_real("z");
	init_real("r");
	init_real("r2");
	init_real("rinv");
	init_real("r2inv");
	init_real("r3inv");
	init_real("rzero");
	init_real("flag");
	init_real("exp0");
	init_real("erfc0");
	init_real("d0");
	init_real("d1");
	init_real("hdotx");
	init_real("phi");
	init_real("tmp");
	init_real("c");
	init_real("s");
	init_real("mflag");
	init_real("nmflag");
	constexpr double alpha = 2.0;
	if (is_float(type)) {
		ewald_limits<float>(R2, H2, alpha);
	} else {
		ewald_limits<double>(R2, H2, alpha);
	}

	const int R = sqrt(R2);
	const int H = sqrt(H2);
	tprint("rzero = fma(dx[0], dx[0], fma(dx[1], dx[1], dx[2] * dx[2])) < TCAST(%0.20e);\n", tiny());
	tprint("flag = TCAST(1) - rzero;\n");
	tprint("mflag = m * flag;\n");
	tprint("nmflag = -mflag;\n");
	for (int xi = -R; xi <= R; xi++) {
		for (int yi = -R; yi <= R; yi++) {
			for (int zi = -R; zi <= R; zi++) {
				const int ii = xi * xi + yi * yi + zi * zi;
				if (ii > R2) {
					continue;
				}
				if (xi == 0) {
					tprint("x = dx[0];\n");
				} else {
					tprint("x = dx[0] - TCAST(%i);\n", xi);
				}
				if (yi == 0) {
					tprint("y = dx[1];\n");
				} else {
					tprint("y = dx[1] - TCAST(%i);\n", yi);
				}
				if (zi == 0) {
					tprint("z = dx[2];\n");
				} else {
					tprint("z = dx[2] - TCAST(%i);\n", zi);
				}
				tprint("r2 = fma(x, x, fma(y, y, z * z));\n");
				tprint("r = sqrt(r2);\n");
				tprint("rinv = TCAST(1) / (r + rzero);\n");
				tprint("r2inv = rinv * rinv;\n");
				tprint("r3inv = r2inv * rinv;\n");
				tprint("erfc0 = erfc(TCAST(%.20e) * r);\n", alpha);
				tprint("exp0 = exp(-TCAST(%.20e) * r * r);\n", alpha * alpha);
				tprint("tmp = TCAST(%.20e) * r * exp0;\n", 2.0 * alpha / sqrt(M_PI));
				tprint("d0 = nmflag * erfc0 * rinv;\n");
				tprint("d1 = nmflag * (tmp + erfc0) * r3inv;\n");
				tprint("f.potential += d0;\n");
				tprint("f.force[0] = fma(x, d1, f.force[0]);\n");
				tprint("f.force[1] = fma(y, d1, f.force[1]);\n");
				tprint("f.force[2] = fma(z, d1, f.force[2]);\n");
			}
		}
	}
	for (int xi = -H; xi <= H; xi++) {
		for (int yi = -H; yi <= H; yi++) {
			for (int zi = -H; zi <= H; zi++) {
				const int h2 = xi * xi + yi * yi + zi * zi;
				if (h2 == 0 || h2 > H2) {
					continue;
				}
				if (xi == 0) {
					if (yi == 0) {
						tprint("hdotx = dx[2] * TCAST(%i);\n", zi);
					} else {
						if (zi == 0) {
							tprint("hdotx = dx[1] * TCAST(%i);\n", yi);
						} else {
							tprint("hdotx = dx[1] * TCAST(%i) + dx[2] * TCAST(%i);\n", yi, zi);
						}
					}
				} else {
					if (yi == 0) {
						if (zi == 0) {
							tprint("hdotx = dx[0] * TCAST(%i);\n", xi);
						} else {
							tprint("hdotx = dx[0] * TCAST(%i) + dx[2] * TCAST(%i);\n", xi, zi);
						}
					} else {
						if (zi == 0) {
							tprint("hdotx = dx[0] * TCAST(%i) + dx[1] * TCAST(%i);\n", xi, yi);
						} else {
							tprint("hdotx = dx[0] * TCAST(%i) + dx[1] * TCAST(%i) + dx[2] * TCAST(%i);\n", xi, yi, zi);
						}
					}
				}
				tprint("phi = TCAST(%.20e) * hdotx;\n", 2.0 * M_PI);
				tprint("s = sin(phi);\n");
				tprint("c = cos(phi);\n");
				const double c0 = -1.0 / h2 * exp((double)(-(M_PI * M_PI) / alpha / alpha) * h2) * (double)(1. / (M_PI));
				const float c1 = 2.0 * M_PI * c0;
				tprint("f.potential = fma(mflag, c * TCAST(%.20e), f.potential);\n", c0);
				tprint("tmp = mflag * s * TCAST(%.20e);\n", c1);
				if (xi) {
					tprint("f.force[0] = fma( TCAST(%i), tmp, f.force[0]);\n", xi);
				}
				if (yi) {
					tprint("f.force[1] = fma( TCAST(%i), tmp, f.force[1]);\n", yi);
				}
				if (zi) {
					tprint("f.force[2] = fma( TCAST(%i), tmp, f.force[2]);\n", zi);
				}
			}
		}
	}
	tprint("r2 = fma(dx[0], dx[0], fma(dx[1], dx[1], dx[2] * dx[2]));\n");
	tprint("rinv = rsqrt(r2 + rzero);\n");
	tprint("r3inv = rinv * rinv * rinv;\n");
	tprint("f.potential += mflag * rinv;\n");
	tprint("f.force[0] += mflag * dx[0] * r3inv;\n");
	tprint("f.force[1] += mflag * dx[1] * r3inv;\n");
	tprint("f.force[2] += mflag * dx[2] * r3inv;\n");
	tprint("f.potential = fma(mflag, TCAST(%.20e), f.potential);\n", M_PI / (alpha * alpha));
	tprint("f.potential = fma(rzero * m, TCAST(2.83729747948179022998), f.potential);\n");
	tprint("return %i;\n", get_running_flops().load() + 39);
	deindent();
	tprint("}\n");
	tprint("}\n");
	return get_running_flops().load();
}

int P2P() {
	func_header("P2P", 0, true, false, false, false, false, "", "f", FORCE, "m", LIT, "dx", VEC3);
	int R2, H2;
	tprint("const static double hsoft = 1e-6;\n");
	tprint("static const T h2(hsoft * hsoft);\n");
	tprint("static const T hinv(T(1) / hsoft);\n");
	tprint("static const T hinv3(T(1) / (hsoft * hsoft * hsoft));\n");
	reset_running_flops();
	tprint("const T r2 = fma(dx[0], dx[0], fma(dx[1], dx[1], dx[2] * dx[2]));\n");
	tprint("T wn(r2 < h2);\n");
	tprint("if( reduce_sum(wn) > 0 ) {\n");
	indent();
	auto flops0 = get_running_flops();
	reset_running_flops();
	tprint("T wf = TCAST(1) - wn;\n");
	tprint("vec3<T> fn, ff;\n");
	tprint("T rzero(r2 < T(%.20e));\n", tiny());
	tprint("const T rinv = rsqrt(r2 + rzero);\n");
	tprint("const T rinv3 = sqr(rinv) * rinv;\n");
	tprint("const T pf = rinv;\n");
	tprint("ff[0] = dx[0] * rinv3;\n");
	tprint("ff[1] = dx[1] * rinv3;\n");
	tprint("ff[2] = dx[2] * rinv3;\n");
	tprint("const T pn = (T(1.5) * hinv - T(0.5) * r2 * hinv3);\n");
	tprint("fn[0] = dx[0] * hinv3;\n");
	tprint("fn[1] = dx[1] * hinv3;\n");
	tprint("fn[2] = dx[2] * hinv3;\n");
	tprint("m = -m;\n");
	tprint("wn *= m;\n");
	tprint("wf *= m;\n");
	tprint("f.potential += fma(pn, wn, pf * wf);\n");
	tprint("f.force[0] += fma(fn[0], wn, ff[0] * wf);\n");
	tprint("f.force[1] += fma(fn[1], wn, ff[1] * wf);\n");
	tprint("f.force[2] += fma(fn[2], wn, ff[2] * wf);\n");
	flops_t flops = flops0;
	flops += get_running_flops();
	tprint("return %i;\n", flops.load());
	reset_running_flops();
	deindent();
	tprint("} else {\n");
	indent();
	tprint("const T rinv = rsqrt(r2);\n");
	tprint("m = -m;\n");
	tprint("const T mrinv3 = m * sqr(rinv) * rinv;\n");
	tprint("f.potential = fma(m, rinv, f.potential);\n");
	tprint("f.force[0] = fma(dx[0], mrinv3, f.force[0]);\n");
	tprint("f.force[1] = fma(dx[1], mrinv3, f.force[1]);\n");
	tprint("f.force[2] = fma(dx[2], mrinv3, f.force[2]);\n");
	flops = flops0;
	flops += get_running_flops();
	tprint("return %i;\n", flops.load());
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");
	tprint("}\n");
	return get_running_flops().load();
}

int L2L_allrot(int P, int Q, int rot) {
	auto index = lindex;
	int flops = 0;
	auto name = std::string("L2") + (Q != 1 ? "L" : "P") + "r" + std::to_string(rot);
	if (Q != 1) {
		if (P == Q) {
			func_header(name.c_str(), P, true, true, true, true, true, "", "L0", EXP, "dx", VEC3);
		} else {
			name += "_ewald";
			func_header(name.c_str(), P, true, true, false, true, true, "", "L0", EXP, "dx", VEC3);
		}
	} else {
		func_header(name.c_str(), P, true, true, true, true, true, "", "f", FORCE, "L0", CEXP, "dx", VEC3);
	}
	const char *two = Q != 1 ? "" : "2";
	open_timer(name);
	if (rot == 2 || scaled) {
		init_real("tmp1");
	}
	if (rot != 2) {
		init_real("ax0");
		init_real("ay0");
	}
	if (periodic) {
		init_real("x0");
		init_real("y0");
		init_real("z0");
	}
	init_real("R2");
	if (periodic || rot == 0) {
		init_real("r2");
	}
	if (Q == 1 && (rot != 2)) {
		init_reals("L2", 4);
	}
	if (rot == 0) {
		init_reals("Y", exp_sz(P));
	} else if (rot == 1) {
		init_reals("Y", std::max(P, xyexp_sz(P)));
	} else if (rot == 2) {
		init_real("R");
		init_real("Rzero");
		init_real("Rinv");
		init_real("cosphi");
		init_real("sinphi");
		init_reals("A", 2 * P + 1);
		tprint("T* const Y=A;\n");
		init_reals("r0A", P);
		init_reals("ipA", P);
		init_reals("inA", P);
		init_reals("La", exp_sz(P));
		tprint("T* r0=r0A;\n");
		tprint("T* ip=ipA;\n");
		tprint("T* in=inA;\n");
		tprint("T* L2=La;\n");
	}
	if (Q != 1) {
		tprint("auto* L=L0;\n");
	} else {
		tprint("expansion<%s,%i> L_st;\n", type.c_str(), P);
		tprint("T* L=L_st.data();\n");
		for (int i = 0; i < exp_sz(P); i++) {
			tprint("L[%i] = L0[%i];\n", i, i);
		}
	}
	reset_running_flops();
	const auto init_L2 = [Q]() {
		if (Q == 1) {
			tprint("L2[0] = L[0];\n");
			tprint("L2[3] = L[3];\n");
			tprint("L2[1] = L[1];\n");
			tprint("L2[2] = L[2];\n");
		}
	};
	if (scaled) {
		tprint("tmp1 = TCAST(1) / L0_st.scale();\n");
		tprint("x *= tmp1;\n");
		tprint("y *= tmp1;\n");
		tprint("z *= tmp1;\n");
	}
	if (periodic) {
		tprint("x0 = x;\n");
		tprint("y0 = y;\n");
		tprint("z0 = z;\n");
	}
	tprint("x = -x;\n");
	tprint("y = -y;\n");
	tprint("z = -z;\n");
	std::function<int(int, int)> yindex;
	tprint("R2 = fma(x, x, y * y);\n");
	if (periodic || rot == 0) {
		tprint("r2 = fma(z, z, R2);\n");
	}
	if (rot == 0) {
		init_L2();
		regular_harmonic_full(P);
		yindex = lindex;
	} else if (rot == 1) {
		L2L_z(P, P);
		init_L2();
		regular_harmonic_xy(P);
		yindex = xyindex;
	}
	flops += get_running_flops().load();
	reset_running_flops();
	if (rot == 2) {
		tprint("Rzero = TCONVERT( R2 < TCAST(%.20e) );\n", tiny());
		tprint("tmp1 = R2 + Rzero;\n");
		tprint("Rinv = rsqrt(tmp1);\n");
		tprint("R = (TCAST(1) - Rzero) / Rinv;\n");
		tprint("cosphi = fma(x, Rinv, Rzero);\n");
		tprint("sinphi = -y * Rinv;\n");
		if (Q != 1) {
			L2L_z(P, P);
			z_rot2(P, "L2", "L", FULL, "L2L", true, false);
			xz_swap2(P, "L", "L2", true, FULL, "L2L");
			tprint("z = R;\n");
			L2L_z(P, P);
			xz_swap2(P, "L2", "L", true, FULL, "L2L");
			tprint("in=ipA;\n");
			tprint("ip=inA;\n");
			z_rot2(P, "L", "L2", FULL, "L2L", false, false);
		} else {
			L2L_z(P, P, "z", "0");
			z_rot2(P, "L2", "L", PRE2, "L2P", true, false);
			xz_swap2(P, "L", "L2", true, PRE2, "L2P");
			init_L2();
			L2L_z(P, Q, "R");
			xz_swap2(Q, "L", "L2", true, FULL, "L2P");
			tprint("in=ipA;\n");
			tprint("ip=inA;\n");
			z_rot2(Q, "L2", "L", FULL, "L2P", false, false);
			flops += get_running_flops().load();
			reset_running_flops();
		}
	} else {
		std::vector<entry_t> pos;
		std::vector<entry_t> neg;
		for (int n = 0; n <= Q; n++) {
			for (int m = 0; m <= n; m++) {
				const auto add_work = [&pos, &neg, n](int sgn, int m, int mstr, int gstr) {
					entry_t entry;
					entry.l = lindex(n, m);
					entry.m = mstr;
					entry.o = gstr;
					if (sgn == 1) {
						pos.push_back(entry);
					} else {
						neg.push_back(entry);
					}
				};
				for (int k = 1; k <= P - n; k++) {
					for (int l = -k; l <= k; l++) {
						if (rot == 1 && abs(l) % 2 != k % 2) {
							continue;
						}
						int mxstr = -99;
						int mystr = -99;
						int gxstr = -99;
						int gystr = -99;
						int mxsgn = 1;
						int mysgn = 1;
						int gxsgn = 1;
						int gysgn = 1;
						if (abs(m + l) > n + k) {
							continue;
						}
						if (-abs(m + l) < -(k + n)) {
							continue;
						}
						if (m + l > 0) {
							mxstr = index(n + k, abs(m + l));
							mystr = index(n + k, -abs(m + l));
						} else if (m + l < 0) {
							if (abs(m + l) % 2 == 0) {
								mxstr = index(n + k, abs(m + l));
								mystr = index(n + k, -abs(m + l));
								mysgn = -1;
							} else {
								mxstr = index(n + k, abs(m + l));
								mystr = index(n + k, -abs(m + l));
								mxsgn = -1;
							}
						} else {
							mxstr = index(n + k, 0);
						}
						if (l > 0) {
							gxstr = yindex(k, abs(l));
							gystr = yindex(k, -abs(l));
							gysgn = -1;
						} else if (l < 0) {
							if (abs(l) % 2 == 0) {
								gxstr = yindex(k, abs(l));
								gystr = yindex(k, -abs(l));
							} else {
								gxstr = yindex(k, abs(l));
								gystr = yindex(k, -abs(l));
								gysgn = -1;
								gxsgn = -1;
							}
						} else {
							gxstr = yindex(k, 0);
						}
						add_work(mxsgn * gxsgn, m, mxstr, gxstr);
						if (gystr != -99 && mystr != -99) {
							add_work(-mysgn * gysgn, m, mystr, gystr);
						}
						if (m > 0) {
							if (gystr != -99) {
								add_work(mxsgn * gysgn, -m, mxstr, gystr);
							}
							if (mystr != -99) {
								add_work(mysgn * gxsgn, -m, mystr, gxstr);
							}
						}
					}
				}
			}
			std::sort(pos.begin(), pos.end(), cmp);
			std::sort(neg.begin(), neg.end(), cmp);
			std::vector<bool> used((P + 1) * (P + 1), false);
			for (const auto &e : neg) {
				if (!used[e.l]) {
					tprint("L%s[%i] = -L%s[%i];\n", two, e.l, two, e.l);
					used[e.l] = true;
				}
			}
			for (const auto &e : neg) {
				tprint("L%s[%i] = fma(Y[%i], L[%i], L%s[%i]);\n", two, e.l, e.o, e.m, two, e.l);
			}
			used = std::vector<bool>((P + 1) * (P + 1), false);
			for (const auto &e : neg) {
				if (!used[e.l]) {
					tprint("L%s[%i] = -L%s[%i];\n", two, e.l, two, e.l);
					used[e.l] = true;
				}
			}
			for (const auto &e : pos) {
				tprint("L%s[%i] = fma(Y[%i], L[%i], L%s[%i]);\n", two, e.l, e.o, e.m, two, e.l);
			}
			neg.resize(0);
			pos.resize(0);
		}
	}

	if (P > 1 && periodic) {
		if (Q != 1) {
			tprint("L[%i] = fma(x0, L0_st.trace2(), L[%i]);\n", index(1, 1), index(1, 1));
			tprint("L[%i] = fma(y0, L0_st.trace2(), L[%i]);\n", index(1, -1), index(1, -1));
			tprint("L[%i] = fma(z0, L0_st.trace2(), L[%i]);\n", index(1, 0), index(1, 0));
			tprint("L[%i] = fma(-TCAST(0.5)*r2, L0_st.trace2(), L[%i]);\n", index(0, 0), index(0, 0));
		} else {
			if (!simd[typenum]) {
				//	tprint( "printf( \"%e\\n\", L0_st.trace2());\n");
				//	tprint( "fflush(stdout);\n");
			}
			tprint("L2[%i] = fma(x0, L0_st.trace2(), L2[%i]);\n", index(1, 1), index(1, 1));
			tprint("L2[%i] = fma(y0, L0_st.trace2(), L2[%i]);\n", index(1, -1), index(1, -1));
			tprint("L2[%i] = fma(z0, L0_st.trace2(), L2[%i]);\n", index(1, 0), index(1, 0));
			tprint("L2[%i] = fma(-TCAST(0.5)*r2, L0_st.trace2(), L2[%i]);\n", index(0, 0), index(0, 0));
		}
	}
	if (Q == 1) {
		if (scaled) {
			tprint("tmp1 = TCAST(1) / L0_st.scale();\n");
			tprint("f.potential = fma(tmp1, L2[0], f.potential);\n");
			tprint("tmp1 = -tmp1 * tmp1;\n");
			tprint("f.force[0] = fma(tmp1, L2[3], f.force[0]);\n");
			tprint("f.force[1] = fma(tmp1, L2[1], f.force[1]);\n");
			tprint("f.force[2] = fma(tmp1, L2[2], f.force[2]);\n");
		} else {
			tprint("f.potential += L2[0];\n");
			tprint("f.force[0] -= L2[3];\n");
			tprint("f.force[1] -= L2[1];\n");
			tprint("f.force[2] -= L2[2];\n");
		}
	}
	close_timer();
	deindent();
	tprint("}\n");
	flops += get_running_flops().load();
	reset_running_flops();
	tprint("return %i;\n", flops);
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	if (Q != 1 && P != Q) {
		//	tprint("}\n");
	}
	if (Q == 1 || P == Q) {
		timing_body += print2str("\"L2%s\", %i, %i, %i, 0.0, 0}", (P == Q ? "L" : "P"), P, rot, flops);
	}
	TAB0();
	return flops;
}

void M2M(int P) {
	std::string str;
	int flops[3];
	int bestrot;
	int bestflops = 1000000000;
	for (int rot = 0; rot < 3; rot++) {
		flops[rot] = M2M_allrot(P, rot);
		if (flops[rot] < bestflops) {
			bestflops = flops[rot];
			bestrot = rot;
		}
	}
	func_header("M2M", P, true, false, false, false, true, "", "M", MUL, "dx", VEC3);
	str = flags_header("M2M", P, bestrot);
	str += "\tif( flags & sfmmWithSingleRotationOptimization ) {\n";
	str += print2str("\t\treturn M2Mr1(M_st, dx, flags);\n");
	str += "\t} else if( flags & sfmmWithDoubleRotationOptimization ) {\n";
	str += print2str("\t\treturn M2Mr2(M_st, dx, flags);\n");
	str += "\t} else {\n";
	str += print2str("\t\treturn M2Mr0(M_st, dx, flags);\n");
	str += "\t}\n";
	tprint(str.c_str());
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	TAB0();
}

std::string P2M(int P) {
	tprint("\n");
	auto fname = func_header("P2M", P, true, true, true, true, true, "", "M0", MUL, "m", LIT, "dx", VEC3);
	init_real("ax0");
	init_real("ay0");
	init_real("ax1");
	init_real("ay1");
	init_real("ax2");
	init_real("ay2");
	init_real("r2");
	init_real("tmp1");
	if (nodip) {
		init_real("Mdx");
		init_real("Mdy");
		init_real("Mdz");
		//		tprint("M[1] = TCAST(0);\n");
		//		tprint("M[2] = TCAST(0);\n");
		//		tprint("M[3] = TCAST(0);\n");
	}
	reset_running_flops();
	if (scaled) {
		tprint("multipole<T, %i> M_st(M0_st.scale());\n", P);
	} else {
		tprint("multipole<T, %i> M_st;\n", P);
	}
	tprint("T* M=M_st.data();\n");
	tprint("x = -x;\n");
	tprint("y = -y;\n");
	tprint("z = -z;\n");
	if (scaled) {
		tprint("tmp1 = TCAST(1) / M0_st.scale();\n");
		tprint("x *= tmp1;\n");
		tprint("y *= tmp1;\n");
		tprint("z *= tmp1;\n");
	}
	tprint("r2 = fma(x, x, fma(y, y, z * z));\n");
	tprint("M[0] = m;\n");
	if (periodic & P > 2) {
		tprint("M_st.trace2() = M[0] * r2;\n");
	}
	const auto mstr = [](int l, int m) {
		if (nodip && l == 1) {
			if (m == 0) {
				return std::string("Mdz");
			} else if (m == 1) {
				return std::string("Mdx");
			} else {
				return std::string("Mdy");
			}
		}
		return std::string("M[") + std::to_string(mindex(l, m)) + "]";
	};
	P--;
	for (int m = 1; m <= P; m++) {
		if (m - 1 > 0) {
			tprint("ax0 = %s * TCAST(%.20e);\n", mstr(m - 1, m - 1).c_str(), 1.0 / (2.0 * m));
			tprint("ay0 = %s * TCAST(%.20e);\n", mstr(m - 1, -(m - 1)).c_str(), 1.0 / (2.0 * m));
			tprint("%s = x * ax0 - y * ay0;\n", mstr(m, m).c_str());
			tprint("%s = fma(y, ax0, x * ay0);\n", mstr(m, -m).c_str());
		} else {
			tprint("%s = x * TCAST(%.20e) * M[0];\n", mstr(m, m).c_str(), 1.0 / (2.0 * m));
			tprint("%s = y * TCAST(%.20e) * M[0];\n", mstr(m, -m).c_str(), 1.0 / (2.0 * m));
		}
	}
	const double c0 = -0.25;
	const double c1 = double(1) / (double((1) * (1)));
	if (2 <= P) {
		tprint("%s = TCAST(-0.25) * r2 * M[0];\n", mstr(2, 0).c_str());
	}
	if (1 <= P) {
		tprint("%s = z * M[0];\n", mstr(1, 0).c_str());
	}
	for (int n = 1; n < P; n++) {
		const double c0 = -double(1) / (double((n + 2) * (n + 2)));
		const double c1 = double(2 * n + 1) / (double((n + 1) * (n + 1)));
		if (n + 2 <= P) {
			tprint("%s = TCAST(%.20e) * r2 * %s;\n", mstr(n + 2, 0).c_str(), c0, mstr(n, 0).c_str());
		}
		tprint("%s = fma(TCAST(%.20e) * z, %s, %s);\n", mstr(n + 1, 0).c_str(), c1, mstr(n, 0).c_str(), mstr(n + 1, 0).c_str());
		for (int m = 1; m <= n; m++) {
			const double c0 = -double(1) / (double((n + 2) * (n + 2)) - double(m * m));
			const double c1 = double(2 * n + 1) / (double((n + 1) * (n + 1)) - double(m * m));
			if (n + 2 <= P) {
				tprint("ax0 = TCAST(%.20e) * r2;\n", c0);
			}
			if (n + 2 <= P) {
				tprint("%s = ax0 * %s;\n", mstr(n + 2, m).c_str(), mstr(n, m).c_str());
				tprint("%s = ax0 * %s;\n", mstr(n + 2, -m).c_str(), mstr(n, -m).c_str());
			}
			if (n == m) {
				tprint("%s = z * %s;\n", mstr(n + 1, m).c_str(), mstr(n, m).c_str());
				tprint("%s = z * %s;\n", mstr(n + 1, -m).c_str(), mstr(n, -m).c_str());
			} else {
				tprint("ay0 = TCAST(%.20e) * z;\n", c1);
				tprint("%s = fma(ay0, %s, %s);\n", mstr(n + 1, m).c_str(), mstr(n, m).c_str(), mstr(n + 1, m).c_str());
				tprint("%s = fma(ay0, %s, %s);\n", mstr(n + 1, -m).c_str(), mstr(n, -m).c_str(), mstr(n + 1, -m).c_str());
			}
		}
	}
	P++;
	for (int n = 0; n < mul_sz(P); n++) {
		tprint("M0[%i] += M[%i];\n", n, n);
	}
	if (periodic) {
		tprint("M0_st.trace2() += M_st.trace2();\n");
	}
	deindent();
	tprint("}\n");
	tprint("return %i;\n", get_running_flops().load());
	timing_body += print2str("\"P2M\", %i, 0, %i, 0.0, 0}", P, get_running_flops(false).load());
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	TAB0();
	return fname;
}

void L2L(int P, int Q) {
	std::string str;
	int flops[3];
	int bestrot;
	int bestflops = 1000000000;
	for (int rot = 0; rot < 3; rot++) {
		flops[rot] = L2L_allrot(P, Q, rot);
		if (flops[rot] < bestflops) {
			bestflops = flops[rot];
			bestrot = rot;
		}
	}

	if (Q == P) {
		func_header("L2L", P, true, false, false, false, true, "", "L", EXP, "dx", VEC3);
		str = flags_header("L2L", P, bestrot);
		str += "\tif( flags & sfmmWithSingleRotationOptimization ) {\n";
		str += print2str("\t\treturn L2Lr1(L_st, dx, flags);\n");
		str += "\t} else if( flags & sfmmWithDoubleRotationOptimization ) {\n";
		str += print2str("\t\treturn L2Lr2(L_st, dx, flags);\n");
		str += "\t} else {\n";
		str += print2str("\t\treturn L2Lr0(L_st, dx, flags);\n");
		str += "\t}\n";
	} else {
		func_header("L2P", P, true, false, false, false, true, "", "f", FORCE, "L", CEXP, "dx", VEC3);
		str = flags_header("L2P", P, bestrot);
		str += "\tif( flags & sfmmWithSingleRotationOptimization ) {\n";
		str += print2str("\t\treturn L2Pr1(f, L_st, dx, flags);\n");
		str += "\t} else if( flags & sfmmWithDoubleRotationOptimization ) {\n";
		str += print2str("\t\treturn L2Pr2(f, L_st, dx, flags);\n");
		str += "\t} else {\n";
		str += print2str("\t\treturn L2Pr0(f, L_st, dx, flags);\n");
		str += "\t}\n";
	}
	tprint(str.c_str());
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	TAB0();
}

int flops_t::load() const {
	return r + 2 * fma + 4 * rdiv; // + con + rcmp;
	//	return r + i + fma + 4 * (rdiv + idiv) + con + asgn + icmp + rcmp;
}

void math_float(std::string _type) {
	if (is_vec(type) || precision[typenum] != 1) {
		return;
	}
	const char *type = _type.c_str();
	if (fp) {
		fclose(fp);
	}
	fp = fopen(full_header.c_str(), "at");
	tprint("\n");
	std::string pre;
	fprintf(fp, "#ifdef __CUDA_ARCH__\n");
	pre = "__device__ __host__";
	tprint("inline %s %s abs(%s);\n", pre.c_str(), type, type);
	tprint("inline %s %s copysign(%s, %s);\n", pre.c_str(), type, type, type);
	tprint("inline %s %s fmin(%s, %s);\n", pre.c_str(), type, type, type);
	tprint("inline %s %s fmax(%s, %s);\n", pre.c_str(), type, type, type);
	tprint("inline %s %s fma(%s, %s, %s);\n", pre.c_str(), type, type, type, type);
	tprint("inline %s %s rsqrt(%s);\n", pre.c_str(), type, type);
	tprint("inline %s %s sqrt(%s);\n", pre.c_str(), type, type);
	tprint("\n%s inline %s abs(%s a) {\n", pre.c_str(), type, type);
	indent();
	tprint("return fabsf(a);\n");
	deindent();
	tprint("}\n\n");
	tprint("\n%s inline %s sqrt(%s a) {\n", pre.c_str(), type, type);
	indent();
	tprint("return sqrtf(a);\n");
	deindent();
	tprint("}\n\n");

	tprint("\n%s inline %s rsqrt(%s a) {\n", pre.c_str(), type, type);
	indent();
	tprint("return rsqrtf(a);\n");
	deindent();
	tprint("}\n\n");

	tprint("\n%s inline %s copysign(%s x, %s y) {\n", pre.c_str(), type, type, type);
	indent();
	tprint("return copysignf(x, y);\n");
	deindent();
	tprint("}\n\n");

	tprint("\n%s inline %s fmin(%s x, %s y) {\n", pre.c_str(), type, type, type);
	indent();
	tprint("return fminf(x, y);\n");
	deindent();
	tprint("}\n\n");

	tprint("\n%s inline %s fmax(%s x, %s y) {\n", pre.c_str(), type, type, type);
	indent();
	tprint("return fmaxf(x, y);\n");
	deindent();
	tprint("}\n\n");

	tprint("\n%s inline %s fma(%s a, %s b, %s c) {\n", pre.c_str(), type, type, type, type);
	indent();
	tprint("return fmaf(a, b, c);\n");
	deindent();
	tprint("}\n\n");
	tprint("#endif\n");
	tprint("\n");

	fprintf(fp, "#ifndef __CUDA_ARCH__\n");
	pre = "";
	tprint("inline %s%s abs(%s);\n", pre.c_str(), type, type);
	tprint("inline %s%s copysign(%s, %s);\n", pre.c_str(), type, type, type);
	tprint("inline %s%s fmin(%s, %s);\n", pre.c_str(), type, type, type);
	tprint("inline %s%s fmax(%s, %s);\n", pre.c_str(), type, type, type);
	tprint("inline %s%s fma(%s, %s, %s);\n", pre.c_str(), type, type, type, type);
	tprint("inline %s%s rsqrt(%s);\n", pre.c_str(), type, type);
	tprint("inline %s %s sqrt(%s);\n", pre.c_str(), type, type);
	tprint("\n%s inline %s abs(%s a) {\n", pre.c_str(), type, type);
	indent();
	tprint("return %sabs(a);\n", is_vec(type) ? "simd::" : "std::");
	deindent();
	tprint("}\n\n");
	tprint("\n%sinline %s sqrt(%s a) {\n", pre.c_str(), type, type);
	indent();
	tprint("return %ssqrt(a);\n", is_vec(type) ? "simd::" : "std::");
	deindent();
	tprint("}\n\n");

	tprint("\n%sinline %s rsqrt(%s a) {\n", pre.c_str(), type, type);
	indent();
	tprint("return %s(1) / %ssqrt(a);\n", type, is_vec(type) ? "simd::" : "std::");
	deindent();
	tprint("}\n\n");

	tprint("\n%sinline %s copysign(%s x, %s y) {\n", pre.c_str(), type, type, type);
	indent();
	tprint("return %scopysign(x, y);\n", is_vec(type) ? "simd::" : "std::");
	deindent();
	tprint("}\n\n");

	tprint("\n%sinline %s fmin(%s x, %s y) {\n", pre.c_str(), type, type, type);
	indent();
	tprint("return %sfminf(x, y);\n", is_vec(type) ? "simd::" : "std::");
	deindent();
	tprint("}\n\n");

	tprint("\n%sinline %s fmax(%s x, %s y) {\n", pre.c_str(), type, type, type);
	indent();
	tprint("return %sfmaxf(x, y);\n", is_vec(type) ? "simd::" : "std::");
	deindent();
	tprint("}\n\n");

	tprint("\n%sinline %s fma(%s a, %s b, %s c) {\n", pre.c_str(), type, type, type, type);
	indent();
	tprint("return %sfmaf(a, b, c);\n", is_vec(type) ? "simd::" : "std::");
	deindent();
	tprint("}\n\n");
	tprint("#endif\n");
	tprint("\n");
	fclose(fp);
	fp = nullptr;
}

void typecast_functions() {
	printf("Doing type headers\n");
	if (fp) {
		fclose(fp);
	}
	for (int i = 0; i < base_rtype.size(); i++) {
		fp = fopen(print2str("./generated_code/include/typecast_%s.hpp", rtype[i].c_str()).c_str(), "at");
		tprint("#pragma once\n\n");
		tprint("#include \"sfmm.hpp\"\n\n");
		tprint("\n");
		tprint("namespace sfmm {\n\n");
		tprint("\n");
		tprint("typedef %s T;\n", rtype[i].c_str());
		tprint("typedef %s V;\n", itype[i].c_str());
		tprint("\n");
		tprint("typedef T TCONVERT;\n");
		tprint("typedef V VCONVERT;\n");
		tprint("\n");
		tprint("#define TCAST(a) (%s(%s(a)))\n", rtype[i].c_str(), base_rtype[i].c_str());
		tprint("#define VCAST(a) (%s(%s(a)))\n", itype[i].c_str(), base_itype[i].c_str());
		tprint("\n");
		tprint("}\n");
		tprint("\n");
		fclose(fp);
	}
	fp = nullptr;
}

int main() {
	printf("generation attributes: ");
#ifdef USE_FLOAT
	printf("float, ");
#endif
#ifdef USE_DOUBLE
	printf("double, ");
#endif
#ifdef USE_SIMD
	printf("simd, ");
#endif
	printf("simd double width = %i, ", SIMD_DOUBLE_WIDTH);
	printf("simd float width = %i, ", SIMD_FLOAT_WIDTH);

#ifdef USE_FLOAT
	base_rtype.push_back("float");
	rtype.push_back("float");
	base_itype.push_back("int32_t");
	itype.push_back("int32_t");
	base_uitype.push_back("uint32_t");
	uitype.push_back("uint32_t");
	precision.push_back(1);
	simd.push_back(0);
	simd_size.push_back(1);
	fixed_type.push_back("fixed32");
#ifdef USE_SIMD
	base_rtype.push_back("float");
	rtype.push_back("simd_f32");
	base_itype.push_back("int32_t");
	itype.push_back("simd_i32");
	base_uitype.push_back("uint32_t");
	uitype.push_back("simd_ui32");
	precision.push_back(1);
	simd.push_back(1);
	simd_size.push_back(SIMD_FLOAT_WIDTH);
	fixed_type.push_back("simd_fixed32");
#endif
#endif
#ifdef USE_DOUBLE
	base_rtype.push_back("double");
	rtype.push_back("double");
	base_itype.push_back("int64_t");
	itype.push_back("int64_t");
	base_uitype.push_back("uint64_t");
	uitype.push_back("uint64_t");
	precision.push_back(2);
	simd.push_back(0);
	simd_size.push_back(1);
	fixed_type.push_back("fixed64");
#ifdef USE_SIMD
	base_rtype.push_back("double");
	rtype.push_back("simd_f64");
	base_itype.push_back("int64_t");
	itype.push_back("simd_i64");
	base_uitype.push_back("uint64_t");
	uitype.push_back("simd_ui64");
	precision.push_back(2);
	simd.push_back(1);
	simd_size.push_back(SIMD_DOUBLE_WIDTH);
	fixed_type.push_back("simd_fixed64");
#endif
#endif
	if (rtype.size() == 0) {
		printf("WARNING: Code generator not given any types - enable at least one of SFMM_USE_FLOAT or SFMM_USE_DOUBLE.\n");
	}
	SYSTEM("mkdir -p generated_code\n");
	SYSTEM("mkdir -p ./generated_code/include\n");
	SYSTEM("mkdir -p ./generated_code/include/detail\n");
	SYSTEM("mkdir -p ./generated_code/src\n");
	SYSTEM("mkdir -p ./generated_code/src/math\n");
	tprint("\n");
	set_file(full_header.c_str());
	tprint("#pragma once\n");
	tprint("\n");
	tprint("#ifdef __CUDACC__\n");
	tprint("#define SFMM_PREFIX __device__ __host__\n");
	tprint("#endif\n");
	tprint("#ifndef __CUDACC__\n");
	tprint("#define SFMM_PREFIX\n");
	tprint("#endif\n");
	tprint("\n");
	tprint("#include <atomic>\n");
	tprint("#include <array>\n");
	tprint("#include <cmath>\n");
	tprint("#include <cstdint>\n");
	tprint("#include <limits>\n");
	tprint("#include <string>\n");
	tprint("#include <utility>\n");
	tprint("#include <time.h>\n");
	tprint("#ifndef __CUDACC__\n");
	tprint("#include <simd.hpp>\n");
	tprint("namespace sfmm {\n");
	tprint("using namespace simd;\n");
	tprint("}\n");
	tprint("#endif /*__CUDACC__*/\n");
	tprint("\n");
	tprint("namespace sfmm {\n");
	tprint("#define sfmmProfilingOff 0\n");
	tprint("#define sfmmWithoutOptimization 1\n");
	tprint("#define sfmmWithSingleRotationOptimization 2\n");
	tprint("#define sfmmWithDoubleRotationOptimization 4\n");
	tprint("#define sfmmWithBestOptimization 16\n");
	tprint("#define sfmmProfilingOn 32\n");
	tprint("#define sfmmFLOPsOnly 64\n");
	tprint("#define sfmmDefaultFlags (sfmmWithBestOptimization)\n");
	include("complex.hpp");
	tprint("\n");
	int ntypenames = rtype.size();
	std::string str;
	str += "template<class T>\n";
	str += "struct type_traits {\n";
	str += "\tstatic constexpr int precision = 0;\n";
	str += "\tstatic constexpr bool is_simd = false;\n";
	str += "\tusing type = void;\n";
	str += "\tusing base_type = void;\n";
	str += "};\n\n";
#ifdef USE_FLOAT
	str += "template<>\n";
	str += "struct type_traits<float> {\n";
	str += "\tstatic constexpr int precision = 1;\n";
	str += "\tstatic constexpr bool is_simd = false;\n";
	str += "\tusing type = float;\n";
	str += "\tusing base_type = float;\n";
	str += "};\n\n";
	str += "class fixed32;\n\n";
	str += "template<>\n";
	str += "struct type_traits<fixed32> {\n";
	str += "\tstatic constexpr int precision = 1;\n";
	str += "\tstatic constexpr bool is_simd = false;\n";
	str += "\tusing type = fixed32;\n";
	str += "\tusing base_type = float;\n";
	str += "};\n\n";
#ifdef USE_SIMD
	str += "#ifndef __CUDACC__\n";
	str += "template<>\n";
	str += "struct type_traits<simd_f32> {\n";
	str += "\tstatic constexpr int precision = 1;\n";
	str += "\tstatic constexpr bool is_simd = true;\n";
	str += "\tusing type = float;\n";
	str += "\tusing base_type = simd::simd_f32;\n";
	str += "};\n\n";
	str += "class simd_fixed32;\n\n";
	str += "template<>\n";
	str += "struct type_traits<simd_fixed32> {\n";
	str += "\tstatic constexpr int precision = 1;\n";
	str += "\tstatic constexpr bool is_simd = true;\n";
	str += "\tusing type = fixed32;\n";
	str += "\tusing base_type = simd::simd_f32;\n";
	str += "};\n\n";
	str += "#endif\n";
#endif
#endif
#ifdef USE_DOUBLE
	str += "template<>\n";
	str += "struct type_traits<double> {\n";
	str += "\tstatic constexpr int precision = 2;\n";
	str += "\tstatic constexpr bool is_simd = false;\n";
	str += "\tusing type = double;\n";
	str += "\tusing base_type = double;\n";
	str += "};\n\n";
	str += "class fixed64;\n\n";
	str += "template<>\n";
	str += "struct type_traits<fixed64> {\n";
	str += "\tstatic constexpr int precision = 2;\n";
	str += "\tstatic constexpr bool is_simd = false;\n";
	str += "\tusing type = fixed64;\n";
	str += "\tusing base_type = double;\n";
	str += "};\n\n";
#ifdef USE_SIMD
	str += "#ifndef __CUDACC__\n";
	str += "template<>\n";
	str += "struct type_traits<simd_f64> {\n";
	str += "\tstatic constexpr int precision = 2;\n";
	str += "\tstatic constexpr bool is_simd = true;\n";
	str += "\tusing type = double;\n";
	str += "\tusing base_type = simd::simd_f64;\n";
	str += "};\n\n";
	str += "class simd_fixed64;\n\n";
	str += "template<>\n";
	str += "struct type_traits<simd_fixed64> {\n";
	str += "\tstatic constexpr int precision = 2;\n";
	str += "\tstatic constexpr bool is_simd = true;\n";
	str += "\tusing type = fixed64;\n";
	str += "\tusing base_type = simd::simd_f64;\n";
	str += "};\n\n";
	str += "#endif\n";
#endif
#endif
	tprint(str.c_str());
	include("vec3.hpp");

	std::string str1 = "\n#define SFMM_EXPANSION_MEMBERS(classname, type, ppp) \\\n"
					   "\tclass reference { \\\n"
					   "\t\tT* ax; \\\n"
					   "\t\tT* ay; \\\n"
					   "\t\tT rsgn; \\\n"
					   "\t\tT isgn; \\\n"
					   "\tpublic: \\\n"
					   "\t\toperator complex<T>() const { \\\n"
					   "\t\t\tif(ax != ay) { \\\n"
					   "\t\t\t\treturn complex<T>(rsgn * *ax, isgn * *ay); \\\n"
					   "\t\t\t} else { \\\n"
					   "\t\t\t\treturn complex<T>(rsgn * *ax, T(0)); \\\n"
					   "\t\t\t} \\\n"
					   "\t\t} \\\n"
					   "\t\treference& operator=(complex<T> other) { \\\n"
					   "\t\t\t*ax = other.real() * rsgn; \\\n"
					   "\t\t\tif(ax != ay) { \\\n"
					   "\t\t\t\t*ay = other.imag() * isgn; \\\n"
					   "\t\t\t} else { \\\n"
					   "\t\t\t\t*ay = T(0); \\\n"
					   "\t\t\t} \\\n"
					   "\t\t\treturn *this; \\\n"
					   "\t\t} \\\n"
					   "\t\tfriend classname<type,ppp>; \\\n"
					   "\t}; \\\n"
					   "\tSFMM_PREFIX complex<T> operator()(int n, int m) const { \\\n"
					   "\t\tcomplex<T> c; \\\n"
					   "\t\tconst int n2n = n * n + n; \\\n"
					   "\t\tconst int m0 = std::abs(m); \\\n"
#ifdef NO_DIPOLE
					   "\tconst int ip = n == 0 ? 0 : n2n + m0 - 3; \\\n"
					   "\tconst int im = n == 0 ? 0 : n2n - m0 - 3; \\\n"
#else
					   "\t\tconst int ip = n2n + m0; \\\n"
					   "\t\tconst int im = n2n - m0; \\\n"
#endif
					   "\t\tc.real() = o[ip]; \\\n"
					   "\t\tc.imag() = o[im]; \\\n"
					   "\t\tif( m < 0 ) { \\\n"
					   "\t\t\tif( m % 2 == 0 ) { \\\n"
					   "\t\t\t\tc.imag() = -c.imag(); \\\n"
					   "\t\t\t} else { \\\n"
					   "\t\t\t\tc.real() = -c.real(); \\\n"
					   "\t\t\t} \\\n"
					   "\t\t} else if( m == 0 ) { \\\n"
					   "\t\t\tc.imag() = T(0); \\\n"
					   "\t\t} \\\n"
					   "\t\treturn c; \\\n"
					   "\t} \\\n"
					   "\tSFMM_PREFIX reference operator()(int n, int m) { \\\n"
					   "\t\treference ref; \\\n"
					   "\t\tconst int n2n = n * n + n; \\\n"
					   "\t\tconst int m0 = std::abs(m); \\\n"
#ifdef NO_DIPOLE
					   "\tconst int ip = n == 0 ? 0 : n2n + m0 - 3; \\\n"
					   "\tconst int im = n == 0 ? 0 : n2n - m0 - 3; \\\n"
#else
					   "\t\tconst int ip = n2n + m0; \\\n"
					   "\t\tconst int im = n2n - m0; \\\n"
#endif
					   "\t\tref.ax = o + ip; \\\n"
					   "\t\tref.ay = o + im; \\\n"
					   "\t\tref.rsgn = ref.isgn = T(1); \\\n"
					   "\t\tif( m < 0 ) { \\\n"
					   "\t\t\tif( m % 2 == 0 ) { \\\n"
					   "\t\t\t\tref.isgn = -ref.isgn; \\\n"
					   "\t\t\t} else { \\\n"
					   "\t\t\t\tref.rsgn = -ref.rsgn; \\\n"
					   "\t\t\t} \\\n"
					   "\t\t} \\\n"
					   "\t\treturn ref; \\\n"
					   "\t} \\\n"
					   "\tSFMM_PREFIX T* data() { \\\n"
					   "\t\treturn o; \\\n"
					   "\t} \\\n"
					   "\tSFMM_PREFIX const T* data() const { \\\n"
					   "\t\treturn o; \\\n"
					   "\t} \\\n"
					   "\n"
					   "";

	fprintf(fp, "%s", str1.c_str());

	set_file(full_header.c_str());
	tprint("\n");
	tprint("namespace detail {\n");
	tprint("template<class T, int P>\n");
	tprint("class expansion_xz {\n");
	tprint("};\n");
	tprint("\ntemplate<class T, int P>\n");
	tprint("class expansion_xy {\n");
	tprint("};\n");
	tprint("}\n");
	tprint("template<class T, int P>\n");
	tprint("class expansion {\n");
	tprint("};\n");
	tprint("\n");
	tprint("template<class T, int P>\n");
	tprint("class multipole {\n");
	tprint("};\n");
	tprint("\n");

	typecast_functions();

	set_file(full_header.c_str());

	for (int ti = 0; ti < ntypenames; ti++) {
		if (simd[ti]) {
			tprint("#ifndef __CUDACC__\n");
		}
		typenum = ti;
		printf("%s\n", rtype[ti].c_str());
		math_float(rtype[ti]);
		prefix = "SFMM_PREFIX";
		type = rtype[ti];
		if (is_vec(type)) {
			set_file(full_header.c_str());
		}
		for (int P = pmin; P <= pmax; P++) {
			std::string fname;
			greens_flops = greens(P);
		}
		for (int P = pmin; P <= pmax; P++) {
			flops_t flops0, flops1, flops2;
			std::string fname;
			flops_t fps;
			const double alpha = is_float(type) ? 2.4 : 2.25;
			MG2L(P);
			if (periodic) {
				greens_ewald(P, alpha);
			}
		}
		fflush(stdout);
		if (simd[ti]) {
			tprint("#endif\n");
		}
		if (is_vec(type)) {
			set_file(full_header.c_str());
		}
	}
	for (int ti = 0; ti < ntypenames; ti++) {
		if (simd[ti]) {
			tprint("#ifndef __CUDACC__\n");
		}
		typenum = ti;
		printf("%s\n", rtype[ti].c_str());
		prefix = "SFMM_PREFIX";
		type = rtype[ti];
		if (is_vec(type)) {
			set_file(full_header.c_str());
		}
		P2P();
		if (periodic) {
			P2P_ewald();
		}
		for (int P = pmin; P <= pmax; P++) {
			L2L(P, P);
			L2L(P, 1);
			P2M(P);
			M2L(P, P);
			M2L(P, 1);
			P2L(P);
			if (periodic) {
				M2L_ewald(P);
				M2P_ewald(P);
				P2L_ewald(P);
			}
			M2M(P);
		}
		fflush(stdout);
		if (simd[ti]) {
			tprint("#endif\n");
		}
		if (is_vec(type)) {
			set_file(full_header.c_str());
		}
	}
	//	printf("./generated_code/include/sfmm.h");
	fflush(stdout);
	set_file(full_header.c_str());
	tprint("\n");
	set_file(full_header.c_str());
	tprint("namespace detail {\n");
	tprint("\n");
	//	tprint("#ifndef __CUDACC__\n");
	//	tprint("%s", detail_header_vec.c_str());
	//	tprint("#endif /* __CUDACC__ */\n\n");
	tprint("\n");
	tprint("template<class T, int P, int ALPHA100>\n");
	tprint("SFMM_PREFIX void greens_ewald_real(expansion<T, P>& G_st, T x, T y, T z) {\n");
	indent();
	tprint("constexpr double ALPHA = ALPHA100 / 100.0;\n");
	tprint("expansion<T, P> Gr_st;\n");
	tprint("const T r2 = fma(x, x, fma(y, y, z * z));\n");
	tprint("const T r = sqrt(r2);\n");
	tprint("greens(Gr_st, vec3<T>( x, y, z));\n");
	tprint("const T* Gr(Gr_st.data());\n");
	tprint("T* G (G_st.data());\n");
	tprint("const T xxx = T(ALPHA) * r;\n");
	tprint("T gam1, exp0;\n");
	tprint("gam1 = erfc(xxx);\n");
	tprint("exp0 = exp(-xxx * xxx);\n");
	tprint("gam1 *= T(%.20e);\n", sqrt(M_PI));
	tprint("const T xfac = T(ALPHA * ALPHA) * r2;\n");
	tprint("T xpow = T(ALPHA) * r;\n");
	tprint("T gam0inv = T(%.20e);\n", 1.0 / sqrt(M_PI));
	tprint("for (int l = 0; l <= P; l++) {\n");
	indent();
	tprint("const int l2 = l * (l + 1);\n");
	tprint("const T gam = gam1 * gam0inv;\n");
	tprint("for (int m = -l; m <= l; m++) {\n");
	indent();
	tprint("const int i = l2 + m;\n");
	tprint("G[i] = fma(gam, Gr[i], G[i]);\n");
	deindent();
	tprint("}\n");
	tprint("gam0inv /= (T(l) + T(0.5));\n");
	tprint("gam1 = fma(T(l + 0.5), gam1, xpow * exp0);\n");
	tprint("xpow *= xfac;\n");
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");

	set_file(full_header.c_str());
#ifdef USE_FLOAT
	include("fixed32.hpp");
#ifdef USE_SIMD
	include("simd_fixed32.hpp");
#endif
#endif
#ifdef USE_DOUBLE
	include("fixed64.hpp");
#ifdef USE_SIMD
	include("simd_fixed64.hpp");
#endif
#endif

	include("p2p.hpp");
	include("periodic.hpp");
	for (int ti = 0; ti < ntypenames; ti++) {
		typenum = ti;
		type = rtype[ti];
		if (simd[ti]) {
			tprint("#ifndef __CUDACC__\n");
		}
		for (int P = pmin - 1; P <= pmax; P++) {
			tprint("\n");
			tprint("template<>\n");
			tprint("class expansion<%s,%i> {\n", type.c_str(), P);
			indent();
			tprint("typedef %s T;\n", type.c_str());
			tprint("typedef typename type_traits<T>::type base_type;\n");
			tprint("T o[%i];\n", exp_sz(P));
			if (periodic && P > 1) {
				tprint("T t;\n");
			}
			if (scaled) {
				tprint("base_type r;\n");
			}
			deindent();
			tprint("public:\n");
			indent();
			tprint("typedef T type;\n");

			if (scaled) {
				tprint("expansion& load( expansion<base_type,%i> other, int index = -1) {\n", P);
				indent();
				tprint("other.rescale(r);\n");
			} else {
				tprint("expansion& load( const expansion<base_type,%i>& other, int index = -1) {\n", P);
				indent();
			}
			if (simd[typenum]) {
				tprint("if( index == -1 ) {\n");
				indent();
				tprint("for( int i = 0; i < %i; i++ ) {\n", mul_sz(P));
				indent();
				tprint("o[i] = other[i];\n");
				deindent();
				tprint("}\n");
				if (periodic && P > 2) {
					tprint("t = other.trace2();\n");
				}
				deindent();
				tprint("} else {\n");
				indent();
				tprint("for( int i = 0; i < %i; i++ ) {\n", mul_sz(P));
				indent();
				tprint("o[i][index] = other[i];\n");
				deindent();
				tprint("}\n");
				if (periodic && P > 2) {
					tprint("t[index] = other.trace2();\n");
				}
				deindent();
				tprint("}\n");
			} else {
				tprint("for( int i = 0; i < %i; i++ ) {\n", mul_sz(P));
				indent();
				tprint("o[i] = other[i];\n");
				deindent();
				tprint("}\n");
				if (periodic && P > 2) {
					tprint("t = other.trace2();\n");
				}
			}
			tprint("return *this;\n");
			deindent();
			tprint("}\n");

			tprint("SFMM_EXPANSION_MEMBERS(expansion, %s, %i);\n", type.c_str(), P);
			tprint("template<class V>\n");
			tprint("SFMM_PREFIX expansion(const expansion<V, %i>& other) {\n", P);
			indent();
			tprint("for( int n = 0; n < %i; n++ ) {\n", exp_sz(P));
			indent();
			tprint("o[n] = T(other.o[n]);\n");
			deindent();
			tprint("}\n");
			if (periodic && P > 1) {
				tprint("t = T(other.t);\n");
			}
			if (scaled) {
				tprint("r = T(other.r);\n");
			}
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX expansion& operator=(const expansion& other) {\n");
			indent();
			tprint("for( int n = 0; n < %i; n++ ) {\n", exp_sz(P));
			indent();
			tprint("o[n] = other.o[n];\n");
			deindent();
			tprint("}\n");
			if (periodic && P > 1) {
				tprint("t = other.t;\n");
			}
			if (scaled) {
				tprint("r = other.r;\n");
			}
			tprint("return *this;\n");
			deindent();
			tprint("}\n");

			tprint("SFMM_PREFIX expansion& operator+=(expansion other) {\n");
			indent();
			if (scaled) {
				tprint("other.rescale(r);\n");
			}
			tprint("for( int n = 0; n < %i; n++ ) {\n", exp_sz(P));
			indent();
			tprint("o[n] += other.o[n];\n");
			deindent();
			tprint("}\n");
			if (periodic && P > 1) {
				tprint("t += other.t;\n");
			}
			tprint("return *this;\n");
			deindent();
			tprint("}\n");

			tprint("SFMM_PREFIX expansion(base_type r0 = base_type(1)) {\n");
			indent();
			fprintf(fp, "#if !defined(NDEBUG) && !defined(__CUDA_ARCH__)\n");
			tprint("for( int n = 0; n < %i; n++ ) {\n", exp_sz(P));
			indent();
			tprint("o[n] = std::numeric_limits<T>::signaling_NaN();\n");
			deindent();
			tprint("}\n");
			if (periodic && P > 1) {
				tprint("t = std::numeric_limits<T>::signaling_NaN();\n");
			}
			fprintf(fp, "#endif\n");
			if (scaled) {
				tprint("r = r0;\n");
			}
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX void init(base_type r0 = base_type(1)) {\n");
			indent();
			tprint("for( int n = 0; n < %i; n++ ) {\n", exp_sz(P));
			indent();
			tprint("o[n] = T(0);\n");
			deindent();
			tprint("}\n");
			if (periodic && P > 1) {
				tprint("t = T(0);\n");
			}
			if (scaled) {
				tprint("r = r0;\n");
			}
			deindent();
			tprint("}\n");

			tprint("SFMM_PREFIX int rescale(base_type r0) {\n");
			indent();
			reset_running_flops();
			if (scaled) {
				tprint("const base_type a = r0 / r;\n");
				tprint("base_type b = a;\n");
				tprint("r = r0;\n");
				for (int n = 0; n <= P; n++) {
					for (int m = -n; m <= n; m++) {
						tprint("o[%i] *= T(b);\n", lindex(n, m));
					}
					if (periodic && P > 1 && n == 2) {
						tprint("t *= T(b);\n", exp_sz(P));
					}
					if (n != P) {
						tprint("b *= base_type(a);\n");
					}
				}
			}
			tprint("return %i;\n", get_running_flops().load());
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX base_type scale() const {\n");
			indent();
			if (scaled) {
				tprint("return r;\n");
			} else {
				tprint("return base_type(1);\n");
			}
			deindent();
			tprint("}\n");
			if (periodic && P > 1) {
				tprint("SFMM_PREFIX T& trace2() {\n");
				indent();
				tprint("return t;\n");
				deindent();
				tprint("}\n");
				tprint("SFMM_PREFIX T trace2() const {\n");
				indent();
				tprint("return t;\n");
				deindent();
				tprint("}\n");
			}
			tprint("SFMM_PREFIX static constexpr size_t size() {\n");
			indent();
			tprint("return %i;\n", exp_sz(P));
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX const T& operator[](int i) const {\n");
			indent();
			tprint("return o[i];\n");
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX T& operator[](int i) {\n");
			indent();
			tprint("return o[i];\n");
			deindent();
			tprint("}\n");
			tprint("template<class V, int P>\n");
			tprint("friend class expansion;\n");
			if (simd[typenum]) {
				if (scaled && periodic && P >= pmin) {
					tprint("friend int M2L_ewald(expansion<T, %i>&, const multipole<T, %i>&, vec3<T>, int);\n", P, P);
				}
				if (scaled && P >= pmin) {
					for (int rot = 0; rot <= 2; rot++) {
						tprint("friend int M2Lr%i(expansion<T, %i>&, const multipole<T, %i>&, vec3<T>, int);\n", rot, P, P);
					}
				}
			}
			deindent();
			tprint("};\n");
		}
		for (int P = pmin; P <= pmax; P++) {
			tprint("\n");
			tprint("template<>\n");
			tprint("class multipole<%s,%i> {\n", type.c_str(), P);
			indent();
			tprint("typedef %s T;\n", type.c_str());
			tprint("typedef typename type_traits<T>::type base_type;\n");
			tprint("T o[%i];\n", mul_sz(P));
			if (periodic && P > 2) {
				tprint("T t;\n");
			}
			if (scaled) {
				tprint("base_type r;\n");
			}
			deindent();
			tprint("public:\n");
			indent();
			tprint("typedef T type;\n");
			if (scaled) {
				tprint("multipole& load( multipole<base_type,%i> other, int index ) {\n", P);
				indent();
				tprint("other.rescale(r);\n");
			} else {
				tprint("multipole& load( const multipole<base_type,%i>& other, int index ) {\n", P);
				indent();
			}
			if (simd[typenum]) {
				tprint("if( index == -1 ) {\n");
				indent();
				tprint("for( int i = 0; i < %i; i++ ) {\n", mul_sz(P));
				indent();
				tprint("o[i] = other[i];\n");
				deindent();
				tprint("}\n");
				if (periodic && P > 2) {
					tprint("t = other.trace2();\n");
				}
				deindent();
				tprint("} else {\n");
				indent();
				tprint("for( int i = 0; i < %i; i++ ) {\n", mul_sz(P));
				indent();
				tprint("o[i][index] = other[i];\n");
				if (periodic && P > 2) {
					tprint("t[index] = other.trace2();\n");
				}
				deindent();
				tprint("}\n");
				deindent();
				tprint("}\n");
			} else {
				tprint("for( int i = 0; i < %i; i++ ) {\n", mul_sz(P));
				indent();
				tprint("o[i] = other[i];\n");
				deindent();
				tprint("}\n");
				if (periodic && P > 2) {
					tprint("t = other.trace2();\n");
				}
			}
			tprint("return *this;\n");
			deindent();
			tprint("}\n");

			tprint("SFMM_EXPANSION_MEMBERS(multipole, %s, %i);\n", type.c_str(), P);
			tprint("template<class V>\n");
			tprint("SFMM_PREFIX multipole(const multipole<V, %i>& other) {\n", P);
			indent();
			tprint("for( int n = 0; n < %i; n++ ) {\n", mul_sz(P));
			indent();
			tprint("o[n] = T(other.o[n]);\n");
			deindent();
			tprint("}\n");
			if (periodic && P > 1) {
				tprint("t = T(other.t);\n");
			}
			if (scaled) {
				tprint("r = T(other.r);\n");
			}
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX multipole& operator=(const multipole& other) {\n");
			indent();
			tprint("for( int n = 0; n < %i; n++ ) {\n", mul_sz(P));
			indent();
			tprint("o[n] = other.o[n];\n");
			deindent();
			tprint("}\n");
			if (periodic && P > 2) {
				tprint("t = other.t;\n");
			}
			if (scaled) {
				tprint("r = other.r;\n");
			}
			tprint("return *this;\n");
			deindent();
			tprint("}\n");

			tprint("SFMM_PREFIX multipole& operator+=(multipole other) {\n");
			indent();
			if (scaled) {
				tprint("other.rescale(r);\n");
			}
			tprint("for( int n = 0; n < %i; n++ ) {\n", mul_sz(P));
			indent();
			tprint("o[n] += other.o[n];\n");
			deindent();
			tprint("}\n");
			if (periodic && P > 2) {
				tprint("t += other.t;\n");
			}
			tprint("return *this;\n");
			deindent();
			tprint("}\n");

			tprint("SFMM_PREFIX multipole(base_type r0 = base_type(1)) {\n");
			indent();
			fprintf(fp, "#if !defined(NDEBUG) && !defined(__CUDA_ARCH__)\n");
			tprint("for( int n = 0; n < %i; n++ ) {\n", mul_sz(P));
			indent();
			tprint("o[n] = std::numeric_limits<T>::signaling_NaN();\n");
			deindent();
			tprint("}\n");
			if (periodic && P > 2) {
				tprint("t = std::numeric_limits<T>::signaling_NaN();\n");
			}
			fprintf(fp, "#endif\n");
			if (scaled) {
				tprint("r = r0;\n");
			}
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX void init(base_type r0 = base_type(1)) {\n");
			indent();
			tprint("for( int n = 0; n < %i; n++ ) {\n", mul_sz(P));
			indent();
			tprint("o[n] = T(0);\n");
			deindent();
			tprint("}\n");
			if (periodic && P > 2) {
				tprint("t = T(0);\n");
			}
			if (scaled) {
				tprint("r = r0;\n");
			}
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX int rescale(base_type r0) {\n");
			indent();
			reset_running_flops();
			if (scaled) {
				tprint("const base_type a = r / r0;\n");
				tprint("base_type b = a;\n");
				tprint("r = r0;\n");
				for (int n = 1; n < P; n++) {
					if (!(nodip && n == 1)) {
						for (int m = -n; m <= n; m++) {
							tprint("o[%i] *= T(b);\n", mindex(n, m));
						}
						if (periodic && P > 2 && n == 2) {
							tprint("t *= T(b);\n");
						}
					}
					if (n != P - 1) {
						tprint("b *= base_type(a);\n");
					}
				}
			}
			tprint("return %i;\n", get_running_flops().load());
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX base_type scale() const {\n");
			indent();
			if (scaled) {
				tprint("return r;\n");
			} else {
				tprint("return base_type(1);\n");
			}
			deindent();
			tprint("}\n");
			if (periodic && P > 2 && P > 1) {
				tprint("SFMM_PREFIX T& trace2() {\n");
				indent();
				tprint("return t;\n");
				deindent();
				tprint("}\n");
				tprint("SFMM_PREFIX T trace2() const {\n");
				indent();

				tprint("return t;\n");
				deindent();
				tprint("}\n");
			}
			tprint("SFMM_PREFIX static constexpr size_t size() {\n");
			indent();
			tprint("return %i;\n", mul_sz(P));
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX const T& operator[](int i) const {\n");
			indent();
			tprint("return o[i];\n");
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX T& operator[](int i) {\n");
			indent();
			tprint("return o[i];\n");
			deindent();
			tprint("}\n");
			tprint("template<class V, int P>\n");
			tprint("friend class multipole;\n");
			if (simd[typenum]) {
				if (periodic) {
					tprint("friend int M2L_ewald(expansion<T, %i>&, const multipole<T, %i>&, vec3<T>, int);\n", P, P);
				}
				for (int rot = 0; rot <= 2; rot++) {
					tprint("friend int M2Lr%i(expansion<T, %i>&, const multipole<T, %i>&, vec3<T>, int);\n", rot, P, P);
					//	if (rot < 2) {
					tprint("friend int M2Pr%i(force_type<T>&, const multipole<T, %i>&, vec3<T>, int);\n", rot, P);
					//		}
				}
			}
			deindent();
			tprint("};\n");
			tprint("\n");
			tprint("\n");
			tprint("namespace detail {\n");
			tprint("template<>\n");
			tprint("class expansion_xz<%s,%i> {\n", type.c_str(), P);
			indent();
			tprint("typedef %s T;\n", type.c_str());
			tprint("T o[%i];\n", half_exp_sz(P));
			deindent();
			tprint("public:\n");
			indent();
			tprint("SFMM_PREFIX expansion_xz() {\n");
			indent();
			fprintf(fp, "#if !defined(NDEBUG) && !defined(__CUDA_ARCH__)\n");
			tprint("for( int n = 0; n < %i; n++ ) {\n", half_exp_sz(P));
			indent();
			tprint("o[n] = std::numeric_limits<T>::signaling_NaN();\n");
			deindent();
			tprint("}\n");
			fprintf(fp, "#endif\n");
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX T* data() {\n");
			indent();
			tprint("return o;\n");
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX const T* data() const {\n");
			indent();
			tprint("return o;\n");
			deindent();
			tprint("}\n");
			deindent();
			tprint("};\n");
			tprint("\ntemplate<>\n");
			tprint("class expansion_xy<%s,%i> {\n", type.c_str(), P);
			indent();
			tprint("typedef %s T;\n", type.c_str());
			tprint("T o[%i];\n", xyexp_sz(P));
			deindent();
			tprint("public:\n");
			indent();
			tprint("SFMM_PREFIX expansion_xy() {\n");
			indent();
			fprintf(fp, "#if !defined(NDEBUG) && !defined(__CUDA_ARCH__)\n");
			tprint("for( int n = 0; n < %i; n++ ) {\n", half_exp_sz(P));
			indent();
			tprint("o[n] = std::numeric_limits<T>::signaling_NaN();\n");
			deindent();
			tprint("}\n");
			fprintf(fp, "#endif\n");
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX T* data() {\n");
			indent();
			tprint("return o;\n");
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX const T* data() const {\n");
			indent();
			tprint("return o;\n");
			deindent();
			tprint("}\n");
			deindent();
			tprint("};\n");
			tprint("}\n");
		}
		fixed_point_covers();

		if (simd[ti]) {
			tprint("#endif\n");
		}
	}
	str = "";
	tprint("%s\n", str.c_str());
	include("complex_impl.hpp");
	include("expansion.hpp");
	tprint("\n");

	tprint("\n");
	str = "template<class V, typename std::enable_if<is_compound_type<V>::value>::type* = nullptr>\n"
		  "inline void apply_padding(V& A, int n) {\n"
		  "\tfor (int i = 0; i < V::size(); i++) {\n"
		  "\t\tapply_padding(A[i], n);\n"
		  "\t}\n"
#ifdef USE_PERIODIC
		  "\tapply_padding(A.trace2(), n);\n"
#endif
		  "}\n"
		  "\n"
		  "template<class V, typename std::enable_if<is_compound_type<V>::value>::type* = nullptr>\n"
		  "inline void apply_mask(V& A, int n) {\n"
		  "\tconst auto mask = create_mask<typename V::type>(n);\n"
		  "\tfor (int i = 0; i < V::size(); i++) {\n"
		  "\t\tA[i] *= mask;\n"
		  "\t}\n"
#ifdef USE_PERIODIC
		  "\tA.trace2() *= mask;\n"
#endif
		  "}\n"
		  "\n"
		  "";
	str += "template<class T, int P, typename std::enable_if<type_traits<T>::is_simd, T>* = nullptr>\n"
		   "inline expansion<typename type_traits<T>::type, P> reduce_sum(const expansion<T, P>& A) {\n"
		   "\tconstexpr int end = expansion<T, P>::size();\n"
#ifdef USE_SCALED
		   "\texpansion<typename type_traits<T>::type, P> B(A.scale());\n"
#else
		   "\texpansion<typename type_traits<T>::type, P> B;\n"
#endif
		   "\tfor (int i = 0; i < end; i++) {\n"
		   "\t\tB[i] = reduce_sum(A[i]);\n"
		   "\t}\n"
#ifdef USE_PERIODIC
		   "\tB.trace2() = reduce_sum(A.trace2());\n"
#endif
		   "\treturn B;\n"
		   "}\n"
		   "\n"
		   "";

	str += "template<class T, int P, typename std::enable_if<type_traits<T>::is_simd, T>* = nullptr>\n"
		   "inline multipole<typename type_traits<T>::type, P> reduce_sum(const multipole<T, P>& A) {\n"
		   "\tconstexpr int end = multipole<T, P>::size();\n"
#ifdef USE_SCALED
		   "\tmultipole<typename type_traits<T>::type, P> B(A.scale());\n"
#else
		   "\tmultipole<typename type_traits<T>::type, P> B;\n"
#endif
		   "\tfor (int i = 0; i < end; i++) {\n"
		   "\t\tB[i] = reduce_sum(A[i]);\n"
		   "\t}\n"
#ifdef USE_PERIODIC
		   "\tB.trace2() = reduce_sum(A.trace2());\n"
#endif
		   "\treturn B;\n"
		   "}\n"
		   "";
	fprintf(fp, "%s", str.c_str());
	fprintf(fp, "}\n");
	fprintf(fp, "#ifndef SFMM_INCLUDE_DONE\n");
	fprintf(fp, "#define SFMM_INCLUDE_DONE\n");
	fprintf(fp, "#include \"./detail/sfmm_detail.hpp\"\n");
	fprintf(fp, "#endif\n");
	/*	include("timer.hpp");
	 tprint("}\n");
	 tprint("extern \"C\" {\n");
	 tprint("void sfmm_detail_atomic_inc_dbl(double* num, double val);\n");
	 tprint("void sfmm_detail_atomic_inc_int(unsigned long long* num, unsigned long long val);\n");
	 tprint("}\n\n");
	 set_file("./generated_code/src/database.cpp");
	 timing_body = std::string(
	 "#include \"sfmm.hpp\"\n\nnamespace sfmm {\nnamespace detail {\nstatic func_data_t function_data[] = {")
	 + timing_body;
	 timing_body += "\n};\n";
	 timing_body += "\nint operator_count() {\n";
	 timing_body += print2str("\treturn %i;\n", timing_cnt);
	 timing_body += print2str("}\n\n", timing_cnt);
	 timing_body += "\nfunc_data_t* operator_data(int index) {\n";
	 timing_body += print2str("\treturn function_data + index;\n");
	 timing_body += print2str("}\n\n", timing_cnt);
	 timing_body += "\n}\n}\n";
	 tprint("%s\n", timing_body.c_str());
	 set_file("./generated_code/src/timing.cpp");
	 include("timing.cpp");
	 set_file("./generated_code/src/atomic.c");
	 include("atomic.c");

	 set_file("./generated_code/src/constants.cpp");
	 tprint("#include \"sfmm.hpp\"\n");
	 tprint("\n");
	 */
	str = "#ifndef __CUDACC__\n"
		  "namespace sfmm {\ninline const simd_f32 simd_fixed32::c0s = simd_f32(std::numeric_limits<std::uint32_t>::max()) + simd_f32(1);\n"
		  "inline const simd_f32 simd_fixed32::c0si = simd_f32(1) / c0s;\n"
		  "inline const simd_f64 simd_fixed64::c0d = simd_f64(std::numeric_limits<std::uint64_t>::max()) + simd_f64(1);\n"
		  "inline const simd_f64 simd_fixed64::c0di = simd_f64(1) / c0d;\n}\n"
		  "#endif\n";
	tprint("%s\n", str.c_str());

	return 0;
}
