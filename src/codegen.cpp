#include <stdio.h>
#include <array>
#include <cmath>
#include <utility>
#include <algorithm>
#include <vector>
#include <climits>
#include <unordered_map>
#include <complex>
#include <set>
#include <algorithm>

using complex = std::complex<double>;

static int ntab = 0;
static int tprint_on = true;

#define FLOAT
#define DOUBLE
#define VEC_DOUBLE
#define VEC_FLOAT
#define VEC_DOUBLE_SIZE 2
#define VEC_FLOAT_SIZE 8

#define ASPRINTF(...) if( asprintf(__VA_ARGS__) == 0 ) {printf( "ASPRINTF error %s %i\n", __FILE__, __LINE__); abort(); }
#define SYSTEM(...) if( system(__VA_ARGS__) != 0 ) {printf( "SYSTEM error %s %i\n", __FILE__, __LINE__); abort(); }

struct flops_t {
	int r;
	int i;
	int d;
	int a;
	int f;
	flops_t() {
		reset();
	}
	flops_t& operator+=(const flops_t& other) {
		r += other.r;
		i += other.i;
		d += other.d;
		a += other.a;
		f += other.f;
		return *this;
	}
	void reset() {
		r = 0;
		i = 0;
		d = 0;
		a = 0;
		f = 0;
	}
	int load() const {
		return r + i + 4 * d + f + a;
	}
};

std::unordered_map<std::string, flops_t> flops_map;

static bool nophi = false;
static bool fmaops = true;
static bool periodic = true;
static int pmin = FMM_PMIN;
static int pmax = FMM_PMAX;
static std::string type = "float";
static std::string sitype = "int";
static std::string uitype = "unsigned";
static int rsqrt_flops;
static int sqrt_flops;
static int sincos_flops;
static int erfcexp_flops;
static const int divops = 4;
static const char* prefix = "";
static std::string detail_header;
static std::string detail_header_vec;
static std::vector<std::string> lines[2];
static std::string vec_header = ""
		"\n"
		"#define create_binary_op(type, op) \\\n"
		"		inline type operator op (const type& u ) const { \\\n"
		"			type w; \\\n"
		"			w.v = v op u.v; \\\n"
		"			return w; \\\n"
		"		} \\\n"
		"		inline type& operator op##= (const type& u ) { \\\n"
		"			*this = *this op u; \\\n"
		"			return *this; \\\n"
		"		}\n"
		"\n"
		"#define create_unary_op(type, op) \\\n"
		"		inline type operator op () const { \\\n"
		"			type w; \\\n"
		"			w.v = op v; \\\n"
		"			return w; \\\n"
		"		}\n"
		"\n"
		"#define create_convert_op_prot(type,otype) \\\n"
		"		inline vec_##type(const vec_##otype&); \\\n"
		"		inline vec_##type& operator=(const vec_##otype&); \\\n"
		"		inline vec_##type& operator=(const otype&)\n"
		"\n"
		"#define create_convert_op_def(type,otype) \\\n"
		"	inline vec_##type::vec_##type(const vec_##otype& other) { \\\n"
		"		v = __builtin_convertvector(other.v, vtype); \\\n"
		"	} \\\n"
		"	inline vec_##type& vec_##type::operator=(const vec_##otype& other) { \\\n"
		"		v = __builtin_convertvector(other.v, vtype); \\\n"
		"		return *this; \\\n"
		"	}\n"
		"\n"
		"#define create_broadcast_op(type) \\\n"
		"	inline vec_##type(const type& other) { \\\n"
		"		v = other - vtype{}; \\\n"
		"	} \\\n"
		"	inline vec_##type& operator=(const type& other) { \\\n"
		"		v = other - vtype{}; \\\n"
		"		return *this; \\\n"
		"	}\n"
		"\n"
		"#define create_compare_op_prot(type,sitype,  op) \\\n"
		"	inline vec_##sitype operator op (const vec_##type&) const\n"
		"\n"
		"#define create_compare_op_def(type,sitype,  op) \\\n"
		"	inline vec_##sitype vec_##type::operator op (const vec_##type& other) const { \\\n"
		"		vec_##sitype w; \\\n"
		"		w.v = (-(v op other.v)); \\\n"
		"		return w; \\\n"
		"	}\n"
		"\n"
		"#define create_vec_types_fwd(type)              \\\n"
		"	class vec_##type\n"
		"\n"
		"#define create_rvec_types(type, sitype, uitype, size)              \\\n"
		"	class vec_##type {                                           \\\n"
		"		typedef type vtype __attribute__ ((vector_size(size*sizeof(type))));  \\\n"
		"		vtype v;  \\\n"
		"	public: \\\n"
		"	inline constexpr vec_##type() : v() {} \\\n"
		"	inline type operator[](int i) const {  \\\n"
		"		return v[i]; \\\n"
		"	}\\\n"
		"	inline type& operator[](int i) {  \\\n"
		"		return v[i]; \\\n"
		"	}\\\n"
		"	create_binary_op(vec_##type, +); \\\n"
		"	create_binary_op(vec_##type, -); \\\n"
		"	create_binary_op(vec_##type, *); \\\n"
		"	create_binary_op(vec_##type, /); \\\n"
		"	create_unary_op(vec_##type, +); \\\n"
		"	create_unary_op(vec_##type, -); \\\n"
		"	create_convert_op_prot(type, sitype); \\\n"
		"	create_convert_op_prot(type, uitype); \\\n"
		"	create_broadcast_op(type); \\\n"
		"	create_compare_op_prot(type, sitype, <); \\\n"
		"	create_compare_op_prot(type, sitype, >); \\\n"
		"	create_compare_op_prot(type, sitype, <=); \\\n"
		"	create_compare_op_prot(type, sitype, >=); \\\n"
		"	create_compare_op_prot(type, sitype, ==); \\\n"
		"	create_compare_op_prot(type, sitype, !=); \\\n"
		"	friend class vec_##sitype; \\\n"
		"	friend class vec_##uitype; \\\n"
		"}\n"
		"\n"
		"#define create_ivec_types(type, otype, rtype, sitype, size)              \\\n"
		"	class vec_##type {                                           \\\n"
		"		typedef type vtype __attribute__ ((vector_size(size*sizeof(type))));  \\\n"
		"		vtype v;  \\\n"
		"	public: \\\n"
		"	inline constexpr vec_##type() : v() {} \\\n"
		"	inline type operator[](int i) const {  \\\n"
		"		return v[i]; \\\n"
		"	}\\\n"
		"	inline type& operator[](int i) {  \\\n"
		"		return v[i]; \\\n"
		"	}\\\n"
		"	create_binary_op(vec_##type, +); \\\n"
		"	create_binary_op(vec_##type, -); \\\n"
		"	create_binary_op(vec_##type, *); \\\n"
		"	create_binary_op(vec_##type, /); \\\n"
		"	create_binary_op(vec_##type, &); \\\n"
		"	create_binary_op(vec_##type, ^); \\\n"
		"	create_binary_op(vec_##type, |); \\\n"
		"	create_binary_op(vec_##type, >>); \\\n"
		"	create_binary_op(vec_##type, <<); \\\n"
		"	create_unary_op(vec_##type, +); \\\n"
		"	create_unary_op(vec_##type, -); \\\n"
		"	create_unary_op(vec_##type, ~); \\\n"
		"	create_broadcast_op(type); \\\n"
		"	create_convert_op_prot(type, rtype); \\\n"
		"	create_convert_op_prot(type, otype); \\\n"
		"	create_compare_op_prot(type,sitype,  <); \\\n"
		"	create_compare_op_prot(type,sitype,  >); \\\n"
		"	create_compare_op_prot(type, sitype, <=); \\\n"
		"	create_compare_op_prot(type,sitype,  >=); \\\n"
		"	create_compare_op_prot(type, sitype,  ==); \\\n"
		"	create_compare_op_prot(type, sitype,  !=); \\\n"
		"	friend class vec_##rtype; \\\n"
		"	friend class vec_##otype; \\\n"
		"}\n"
		"\n"
		"#define create_rvec_types_def(type, sitype, uitype, size)\\\n"
		"	create_convert_op_def(type, sitype); \\\n"
		"	create_convert_op_def(type, uitype)\n"
		"\n"
		"#define create_ivec_types_def(type, otype, rtype, size)              \\\n"
		"	create_convert_op_def(type, rtype); \\\n"
		"	create_convert_op_def(type, otype)\n"
		"\n"
		"#define create_vec_types(rtype, sitype, uitype, size) \\\n"
		"	create_vec_types_fwd(rtype); \\\n"
		"	create_vec_types_fwd(uitype); \\\n"
		"	create_vec_types_fwd(sitype); \\\n"
		"	create_rvec_types(rtype, sitype, uitype, size); \\\n"
		"	create_ivec_types(uitype, sitype, rtype, sitype, size); \\\n"
		"	create_ivec_types(sitype, uitype, rtype, sitype, size); \\\n"
		"	create_rvec_types_def(rtype, sitype, uitype, size); \\\n"
		"	create_ivec_types_def(uitype, sitype, rtype, size); \\\n"
		"	create_ivec_types_def(sitype, uitype, rtype, size); \\\n"
		"	create_compare_op_def(rtype, sitype, <); \\\n"
		"	create_compare_op_def(rtype,sitype,  >); \\\n"
		"	create_compare_op_def(rtype, sitype, <=); \\\n"
		"	create_compare_op_def(rtype, sitype, >=); \\\n"
		"	create_compare_op_def(rtype,sitype,  ==); \\\n"
		"	create_compare_op_def(rtype, sitype, !=); \\\n"
		"	create_compare_op_def(uitype,sitype,  <); \\\n"
		"	create_compare_op_def(uitype, sitype, >); \\\n"
		"	create_compare_op_def(uitype, sitype, <=); \\\n"
		"	create_compare_op_def(uitype, sitype, >=); \\\n"
		"	create_compare_op_def(uitype, sitype, ==); \\\n"
		"	create_compare_op_def(uitype, sitype, !=); \\\n"
		"	create_compare_op_def(sitype, sitype, <); \\\n"
		"	create_compare_op_def(sitype,sitype,  >); \\\n"
		"	create_compare_op_def(sitype,sitype,  <=); \\\n"
		"	create_compare_op_def(sitype, sitype, >=); \\\n"
		"	create_compare_op_def(sitype, sitype, ==); \\\n"
		"	create_compare_op_def(sitype, sitype, !=) \n";

static int cuda = 0;

static bool is_float(std::string str) {
	return str == "float" || str == "vec_float";
}

static bool is_double(std::string str) {
	return str == "double" || str == "vec_double";
}

static bool is_vec(std::string str) {
	return str == "vec_double" || str == "vec_float";
}

const double ewald_r2 = (2.6 + 0.5 * sqrt(3));
const int ewald_h2 = 8;

static FILE* fp = nullptr;
int exp_sz(int P) {
	return (P + 1) * (P + 1);
}

int half_exp_sz(int P) {
	return (P + 2) * (P + 1) / 2;
}

int mul_sz(int P) {
	return P * P;
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

template<class ...Args>
void tprint(const char* str, Args&&...args) {
	if (fp == nullptr) {
		return;
	}
	if (tprint_on) {
		for (int i = 0; i < ntab; i++) {
			fprintf(fp, "\t");
		}
		fprintf(fp, str, std::forward<Args>(args)...);
	}
}

void tprint(const char* str) {
	if (fp == nullptr) {
		return;
	}
	if (tprint_on) {
		for (int i = 0; i < ntab; i++) {
			fprintf(fp, "\t");
		}
		fprintf(fp, "%s", str);
	}
}

template<class ...Args>
void tprint(int index, const char* fstr, Args&&...args) {
	if (fp == nullptr) {
		return;
	}
	std::string str;
	if (tprint_on) {
		for (int i = 0; i < ntab; i++) {
			str += "\t";
		}
		char* buf;
		ASPRINTF(&buf, fstr, std::forward<Args>(args)...)
		str += buf;
		free(buf);
		lines[index].push_back(str);
	}
}

void tprint(int index, const char* fstr) {
	if (fp == nullptr) {
		return;
	}
	std::string str;
	if (tprint_on) {
		for (int i = 0; i < ntab; i++) {
			str += "\t";
		}
		str += fstr;
		lines[index].push_back(str);
	}
}

void tprint_flush() {
	int n0 = 0;
	int n1 = 0;
	while (n0 < lines[0].size() || n1 < lines[1].size()) {
		if (n0 < lines[0].size() && n1 < lines[1].size()) {
			if ((double) n0 / lines[0].size() < (double) n1 / lines[1].size()) {
				fprintf(fp, "%s", lines[0][n0].c_str());
				n0++;
			} else {
				fprintf(fp, "%s", lines[1][n1].c_str());
				n1++;
			}
		} else if (n0 < lines[0].size()) {
			fprintf(fp, "%s", lines[0][n0].c_str());
			n0++;
		} else {
			fprintf(fp, "%s", lines[1][n1].c_str());
			n1++;
		}
	}
	lines[0].resize(0);
	lines[1].resize(0);
}

double factorial(int n) {
	return n == 0 ? 1.0 : n * factorial(n - 1);
}

double nonepow(int i) {
	return i % 2 == 0 ? 1.0 : -1.0;
}

template<class T>
void ewald_limits(int& r2, int& h2, double alpha) {
	r2 = 0;
	h2 = 0;
	const auto threesqr = [](int i) {
		for( int x = 0; x <= i; x++ ) {
			for( int y = 0; y <= i; y++) {
				for( int z = 0; z <= i; z++) {
					if( x*x + y*y + z*z == i ) {
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

enum arg_type {
	LIT, PTR, CPTR, EXP, MUL, CEXP, CMUL, FORCE
};

void init_real(std::string var) {
	tprint("T %s;\n", var.c_str());
}

void init_reals(std::string var, int cnt) {
	std::string str;
	str = "T " + var + "[" + std::to_string(cnt) + "];";
//			" = {";
	/*	for (int n = 0; n < cnt - 1; n++) {
	 str += "std::numeric_limits<T>::signaling_NaN(), ";
	 }
	 str += "std::numeric_limits<T>::signaling_NaN()};";*/
	tprint("%s\n", str.c_str());
}

template<class ...Args>
std::string func_args(int P, const char* arg, arg_type atype, int term) {
	std::string str;
	if (atype == EXP) {
		str += std::string("expansion_type<") + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == MUL) {
		str += std::string("multipole_type<") + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == CEXP) {
		str += std::string("const expansion_type<") + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == CMUL) {
		str += std::string("const multipole_type<") + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == FORCE) {
		str += std::string("force_type<") + type + ">& " + arg;
	} else {
		if (atype == CPTR) {
			str += std::string("const ");
		}
		str += std::string(type) + " ";
		if (atype == PTR || atype == CPTR) {
			str += "*";
		}
		str += std::string(arg);
	}
	return str;
}

template<class ...Args>
std::string func_args(int P, const char* arg, arg_type atype, Args&& ...args) {
	auto str = func_args(P, arg, atype, 1);
	str += std::string(", ");
	str += func_args(P, std::forward<Args>(args)...);
	return str;
}

template<class ...Args>
void func_args_cover(int P, const char* arg, arg_type atype, int term) {
	std::string str;
	if (atype == EXP) {
		tprint("T* %s = %s_st.data();\n", arg, arg);
		tprint("T %s_r = %s_st.scale();\n", arg, arg);
	} else if (atype == CEXP) {
		tprint("const T* %s = %s_st.data();\n", arg, arg);
		tprint("T %s_r = %s_st.scale();\n", arg, arg);
	} else if (atype == MUL) {
		tprint("T* %s = %s_st.data();\n", arg, arg);
		tprint("T %s_r = %s_st.scale();\n", arg, arg);
	} else if (atype == CMUL) {
		tprint("const T* %s = %s_st.data();\n", arg, arg);
		tprint("T %s_r = %s_st.scale();\n", arg, arg);
	}
}

template<class ...Args>
void func_args_cover(int P, const char* arg, arg_type atype, Args&& ...args) {
	func_args_cover(P, arg, atype, 1);
	func_args_cover(P, std::forward<Args>(args)...);
}

template<class ... Args>
void func_header(const char* func, int P, bool pub, bool calcpot, bool flags, std::string head, Args&& ...args) {
	if (tprint_on) {
		static std::set<std::string> igen;
		std::string func_name = std::string(func);
		std::string file_name = func_name + "_" + type + "_P" + std::to_string(P) + (cuda ? ".cu" : ".cpp");
		func_name = "void " + func_name;
		func_name += "(" + func_args(P, std::forward<Args>(args)..., 0);
		auto func_name2 = func_name;
		if (flags) {
			func_name += ", int flags = 0)";
			func_name2 += ", int flags)";
			if (!pub) {
				func_name = func_name2;
			}
		} else {
			func_name += ")";
			func_name2 += ")";
		}
		set_file("./generated_code/include/spherical_fmm.hpp");
		static std::set<std::string> already_printed;
		if (already_printed.find(func_name) == already_printed.end()) {
			already_printed.insert(func_name);
			if (prefix[0]) {
				func_name = std::string(prefix) + " " + func_name;
				func_name2 = std::string(prefix) + " " + func_name2;
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
		}
		std::string func1 = std::string(func) + std::string("_") + type;
		auto dir = std::string("./generated_code/src/") + type;
		if (cuda) {
			dir += "_cuda";
		}
		std::string cmd = "mkdir -p " + dir;
		SYSTEM(cmd.c_str());
		dir += "/P" + std::to_string(P);
		cmd = "mkdir -p " + dir;
		SYSTEM(cmd.c_str());
		file_name = dir + "/" + file_name;
//		printf("%s ", file_name.c_str());
		set_file(file_name);
		tprint("#include <stdio.h>\n");
		tprint("#include \"spherical_fmm.hpp\"\n");
		tprint("#include \"typecast_%s.hpp\"\n", type.c_str());
		tprint("\n");
		tprint("namespace fmm {\n");
		if (!pub) {
			tprint("namespace detail {\n");
		}
		if (head != "") {
			tprint("\n");
			tprint("%s\n", head.c_str());
		}
		tprint("\n");
		tprint("%s {\n", func_name2.c_str());
		indent();
		if (flags && calcpot) {
			tprint("const bool calcpot = !(flags & FMM_NOCALC_POT);\n");
		}
		func_args_cover(P, std::forward<Args>(args)..., 0);
	} else {
		indent();
	}
}

void set_tprint(bool c) {
	tprint_on = c;
}

int index(int l, int m) {
	return l * (l + 1) + m;
}

constexpr int cindex(int l, int m) {
	return l * (l + 1) / 2 + m;
}

template<class T>
T nonepow(int m) {
	return m % 2 == 0 ? double(1) : double(-1);
}

template<int P>
struct spherical_expansion: public std::array<complex, (P + 1) * (P + 2) / 2> {
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

int z_rot(int P, const char* name, bool noevenhi, bool exclude, bool noimaghi, flops_t& fps) {
//noevenhi = false;
	int flops = 0;
	tprint("rx[0] = cosphi;\n");
	fps.a++;
	tprint("ry[0] = sinphi;\n");
	fps.a++;
	for (int m = 1; m < P; m++) {
		tprint("rx[%i] = rx[%i] * cosphi - ry[%i] * sinphi;\n", m, m - 1, m - 1);
		fps.r += 3;
		tprint("ry[%i] = detail::fma(rx[%i], sinphi, ry[%i] * cosphi);\n", m, m - 1, m - 1);
		fps.r++;
		fps.f++;
		flops += 6 - fmaops;
	}
	int mmin = 1;
	bool initR = true;
	int sw = 1;
	for (int m = 1; m <= P; m++) {
		sw = sw == 0 ? 1 : 0;
		for (int l = m; l <= P; l++) {
			if (noevenhi && l == P) {
				if ((((P + l) / 2) % 2 == 1) ? m % 2 == 0 : m % 2 == 1) {
					continue;
				}
			}
			if (exclude && l == P && m % 2 == 1) {
				tprint(sw, "%s[%i] = -%s[%i] * ry[%i];\n", name, index(l, m), name, index(l, -m), m - 1);
				fps.r += 2;
				tprint(sw, "%s[%i] *= rx[%i];\n", name, index(l, -m), m);
				fps.r++;
				flops += 3;
			} else if ((exclude && l == P && m % 2 == 0) || (noimaghi && l == P)) {
				tprint(sw, "%s[%i] = %s[%i] * ry[%i];\n", name, index(l, -m), name, index(l, m), m - 1);
				fps.r++;
				tprint(sw, "%s[%i] *= rx[%i];\n", name, index(l, m), m - 1);
				fps.r++;
				flops += 2;
			} else {
				if (noevenhi && ((l >= P - 1 && m % 2 == P % 2))) {
					tprint(sw, "%s[%i] = %s[%i] * ry[%i];\n", name, index(l, -m), name, index(l, m), m - 1);
					fps.r++;
					tprint(sw, "%s[%i] *= rx[%i];\n", name, index(l, m), m - 1);
					fps.r++;
					flops += 2;
				} else {
					tprint(sw, "tmp%i = %s[%i];\n", sw, name, index(l, m));
					fps.a++;
					tprint(sw, "%s[%i] = %s[%i] * rx[%i] - %s[%i] * ry[%i];\n", name, index(l, m), name, index(l, m), m - 1, name, index(l, -m), m - 1);
					fps.r += 3;
					tprint(sw, "%s[%i] = detail::fma(tmp%i, ry[%i], %s[%i] * rx[%i]);\n", name, index(l, -m), sw, m - 1, name, index(l, -m), m - 1);
					fps.r++;
					fps.f++;
					flops += 6 - fmaops;
				}
			}

		}
	}
	tprint_flush();
	return flops;
}

int m2l(int P, int Q, const char* mname, const char* lname, flops_t& fps) {
	int flops = 0;
	tprint("A[0] = rinv;\n");
	fps.a++;
	for (int n = 1; n <= P; n++) {
		tprint("A[%i] = rinv * A[%i];\n", n, n - 1);
		fps.r++;
		flops++;
	}
	for (int n = 2; n <= P; n++) {
		tprint("A[%i] *= TCAST(%.20e);\n", n, factorial(n));
		fps.r++;
		flops++;
	}
	int sw = 0;
	tprint(sw, "if( calcpot ) {\n");
	indent();
	for (int n = 0; n <= Q; n++) {
		for (int m = 0; m <= n; m++) {
			if (n > 0) {
				sw = sw == 0 ? 1 : 0;
			}
			bool pfirst = true;
			bool nfirst = true;
			const int maxk = std::min(P - n, P - 1);
			bool looped = true;
			for (int k = m; k <= maxk; k++) {
				looped = true;
				if (pfirst) {
					pfirst = false;
					tprint(sw, "%s[%i] = %s[%i] * A[%i];\n", lname, index(n, m), mname, index(k, m), n + k);
					fps.r++;
					flops += 1;
				} else {
					tprint(sw, "%s[%i] = detail::fma(%s[%i], A[%i], %s[%i]);\n", lname, index(n, m), mname, index(k, m), n + k, lname, index(n, m));
					fps.f++;
					flops += 2 - fmaops;
				}
				if (m != 0) {
					if (nfirst) {
						nfirst = false;
						tprint(sw, "%s[%i] = %s[%i] * A[%i];\n", lname, index(n, -m), mname, index(k, -m), n + k);
						fps.r++;
						flops += 1;
					} else {
						tprint(sw, "%s[%i] = detail::fma(%s[%i], A[%i], %s[%i]);\n", lname, index(n, -m), mname, index(k, -m), n + k, lname, index(n, -m));
						fps.f++;
						flops += 2 - fmaops;
					}
				}
			}
			if (m % 2 != 0) {
				if (!pfirst) {
					tprint(sw, "%s[%i] = -%s[%i];\n", lname, index(n, m), lname, index(n, m));
					fps.r++;
					flops++;
				}
				if (!nfirst) {
					tprint(sw, "%s[%i] = -%s[%i];\n", lname, index(n, -m), lname, index(n, -m));
					fps.r++;
					flops++;
				}
			}
		}
		if (n == 0) {
			deindent();
			tprint(sw, "}\n");
			tprint_flush();
		}
	}
	tprint_flush();
	return flops;

}

int xz_swap(int P, const char* name, bool inv, bool m_restrict, bool l_restrict, bool noevenhi, flops_t& fps) {
//noevenhi = false;
//	tprint("template<class T>\n");
//	tprint(" inline void %s( array<T,%i>& %s ) {\n", fname, (P + 1) * (P + 1), name);
	int flops = 0;
	auto brot = [inv](int n, int m, int l) {
		if( inv ) {
			return Brot(n,m,l);
		} else {
			return Brot(n,l,m);
		}
	};
	for (int n = 1; n <= P; n++) {
		int lmax = n;
		if (l_restrict && lmax > (P) - n) {
			lmax = (P) - n;
		}
		for (int m = -lmax; m <= lmax; m++) {
			if (noevenhi && P == n && P % 2 != abs(m) % 2) {
			} else {
				tprint("A[%i] = %s[%i];\n", m + P, name, index(n, m));
				fps.a++;
			}
		}
		std::vector<std::vector<std::pair<double, int>>>ops(2 * n + 1);
		int mmax = n;
		if (m_restrict && mmax > (P) - n) {
			mmax = (P + 1) - n;
		}
		int mmin = 0;
		int stride = 1;
		for (int m = 0; m <= mmax; m += stride) {
			for (int l = 0; l <= lmax; l++) {
				if (noevenhi && P == n && P % 2 != abs(l) % 2) {
					continue;
				}
				double r = l == 0 ? brot(n, m, 0) : brot(n, m, l) + nonepow<double>(l) * brot(n, m, -l);
				double i = l == 0 ? 0.0 : brot(n, m, l) - nonepow<double>(l) * brot(n, m, -l);
				if (r != 0.0) {
					ops[n + m].push_back(std::make_pair(r, P + l));
				}
				if (i != 0.0 && m != 0) {
					ops[n - m].push_back(std::make_pair(i, P - l));
				}
			}
		}
		for (int m = 0; m < 2 * n + 1; m++) {
			std::sort(ops[m].begin(), ops[m].end(), [](std::pair<double,int> a, std::pair<double,int> b) {
				return a.first < b.first;
			});
		}
		int sw = 0;
		for (int m = 0; m < 2 * n + 1; m++) {
			sw = sw == 0 ? 1 : 0;
			for (int l = 0; l < ops[m].size(); l++) {
				int len = 1;
				while (len + l < ops[m].size()) {
					if (ops[m][len + l].first == ops[m][l].first && !close21(ops[m][l].first)) {
						len++;
					} else {
						break;
					}
				}
				if (len == 1) {
					if (close21(ops[m][l].first)) {
						tprint(sw, "%s[%i] %s= A[%i];\n", name, index(n, m - n), l == 0 ? "" : "+", ops[m][l].second);
						if (l == 0) {
							fps.a++;
						} else {
							fps.r++;
						}
						flops += 1 - (l == 0);
					} else {
						if (l == 0) {
							tprint(sw, "%s[%i] = TCAST(%.20e) * A[%i];\n", name, index(n, m - n), ops[m][l].first, ops[m][l].second);
							fps.r++;
							flops += 1;
						} else {
							tprint(sw, "%s[%i] = detail::fma(TCAST(%.20e), A[%i], %s[%i]);\n", name, index(n, m - n), ops[m][l].first, ops[m][l].second, name,
									index(n, m - n));
							fps.f++;
							flops += 2 - fmaops;
						}
					}
				} else {
					tprint(sw, "tmp%i = A[%i];\n", sw, ops[m][l].second);
					fps.a++;
					for (int p = 1; p < len; p++) {
						tprint(sw, "tmp%i += A[%i];\n", sw, ops[m][l + p].second);
						fps.r++;
						flops++;
					}
					if (l == 0) {
						tprint(sw, "%s[%i] = TCAST(%.20e) * tmp%i;\n", name, index(n, m - n), ops[m][l].first, sw);
						fps.r++;
						flops += 1;
					} else {
						tprint(sw, "%s[%i] = detail::fma(TCAST(%.20e), tmp%i, %s[%i]);\n", name, index(n, m - n), ops[m][l].first, sw, name, index(n, m - n));
						fps.f++;
						flops += 2 - fmaops;
					}
				}
				l += len - 1;
			}
		}
		tprint_flush();
	}
	return flops;
}

int greens_body(int P, flops_t& fps, const char* M = nullptr) {
	int flops = 0;
	init_real("r2");
	init_real("r2inv");
	init_real("ax");
	init_real("ay");
	tprint("r2 = detail::fma(x, x, detail::fma(y, y, z * z));\n");
	fps.r++;
	fps.f += 2;
	flops += 5 - 2 * fmaops;
	if (M) {
		tprint("r2inv = %s / r2;\n", M);
	} else {
		tprint("r2inv = TCAST(1) / r2;\n");
	}
	fps.d++;
	tprint("O[0] = detail::rsqrt(r2);\n");
	fps += flops_map[std::string("rqrt_") + type];
	flops += rsqrt_flops;
	if (M) {
		tprint("O[0] *= %s;\n", M);
		fps.r++;
		flops++;
	}
	tprint("O_st.trace2() = TCAST(0);\n");
	fps.a++;
	tprint("x *= r2inv;\n");
	fps.r++;
	tprint("y *= r2inv;\n");
	fps.r++;
	tprint("z *= r2inv;\n");
	fps.r++;
	flops += 3;
	tprint("r2inv = -r2inv;\n");
	fps.r++;
	for (int m = 0; m <= P; m++) {
		if (m == 1) {
			tprint("O[%i] = x * O[0];\n", index(m, m));
			fps.r++;
			tprint("O[%i] = y * O[0];\n", index(m, -m));
			fps.r++;
			flops += 2;
		} else if (m > 0) {
			tprint("ax = O[%i] * TCAST(%i);\n", index(m - 1, m - 1), 2 * m - 1);
			fps.r++;
			tprint("ay = O[%i] * TCAST(%i);\n", index(m - 1, -(m - 1)), 2 * m - 1);
			fps.r++;
			tprint("O[%i] = x * ax - y * ay;\n", index(m, m));
			fps.r += 3;
			tprint("O[%i] = detail::fma(y, ax, x * ay);\n", index(m, -m));
			fps.r++;
			fps.f++;
			flops += 8 - fmaops;
		}
		if (m + 1 <= P) {
			tprint("O[%i] = TCAST(%i) * z * O[%i];\n", index(m + 1, m), 2 * m + 1, index(m, m));
			fps.r += 2;
			flops += 2;
			if (m != 0) {
				tprint("O[%i] = TCAST(%i) * z * O[%i];\n", index(m + 1, -m), 2 * m + 1, index(m, -m));
				fps.r += 2;
				flops += 2;
			}
		}
		for (int n = m + 2; n <= P; n++) {
			if (m != 0) {
				tprint("ax = TCAST(%i) * z;\n", 2 * n - 1);
				fps.r++;
				tprint("ay = TCAST(%i) * r2inv;\n", (n - 1) * (n - 1) - m * m);
				fps.r++;
				tprint("O[%i] = detail::fma(ax, O[%i], ay * O[%i]);\n", index(n, m), index(n - 1, m), index(n - 2, m));
				fps.f++;
				fps.r++;
				tprint("O[%i] = detail::fma(ax, O[%i], ay * O[%i]);\n", index(n, -m), index(n - 1, -m), index(n - 2, -m));
				fps.f++;
				fps.r++;
				flops += 8 - fmaops;
			} else {
				if ((n - 1) * (n - 1) - m * m == 1) {
					tprint("O[%i] = detail::fma(TCAST(%i) * z, O[%i], r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), index(n - 2, m));
					fps.f++;
					fps.r += 2;
				} else {
					tprint("O[%i] = detail::fma(TCAST(%i) * z, O[%i], TCAST(%i) * r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m),
							(n - 1) * (n - 1) - m * m, index(n - 2, m));
					fps.f++;
					fps.r += 3;
				}

			}
		}
	}
	return flops;
}

int m2lg_body(int P, int Q, flops_t& fps) {
	int flops = 0;
	int sw = 0;
	tprint(sw, "if( calcpot ) {\n");
	indent();
	for (int n = 0; n <= Q; n++) {
		for (int m = 0; m <= n; m++) {
			if (n > 0) {
				sw = sw == 0 ? 1 : 0;
			}
			const int kmax = std::min(P - n, P - 1);
			std::vector<std::pair<std::string, std::string>> pos_real;
			std::vector<std::pair<std::string, std::string>> neg_real;
			std::vector<std::pair<std::string, std::string>> pos_imag;
			std::vector<std::pair<std::string, std::string>> neg_imag;
			for (int k = 0; k <= kmax; k++) {
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					bool greal = false;
					bool mreal = false;
					int gxsgn = 1;
					int gysgn = 1;
					int mxsgn = 1;
					int mysgn = 1;
					char* gxstr = nullptr;
					char* gystr = nullptr;
					char* mxstr = nullptr;
					char* mystr = nullptr;
					if (m + l > 0) {
						ASPRINTF(&gxstr, "O[%i]", index(n + k, m + l));
						ASPRINTF(&gystr, "O[%i]", index(n + k, -m - l));
					} else if (m + l < 0) {
						if (abs(m + l) % 2 == 0) {
							ASPRINTF(&gxstr, "O[%i]", index(n + k, -m - l));
							ASPRINTF(&gystr, "O[%i]", index(n + k, m + l));
							gysgn = -1;
						} else {
							ASPRINTF(&gxstr, "O[%i]", index(n + k, -m - l));
							ASPRINTF(&gystr, "O[%i]", index(n + k, m + l));
							gxsgn = -1;
						}
					} else {
						greal = true;
						ASPRINTF(&gxstr, "O[%i]", index(n + k, 0));
					}
					if (l > 0) {
						ASPRINTF(&mxstr, "M[%i]", index(k, l));
						ASPRINTF(&mystr, "M[%i]", index(k, -l));
						mysgn = -1;
					} else if (l < 0) {
						if (l % 2 == 0) {
							ASPRINTF(&mxstr, "M[%i]", index(k, -l));
							ASPRINTF(&mystr, "M[%i]", index(k, l));
						} else {
							ASPRINTF(&mxstr, "M[%i]", index(k, -l));
							ASPRINTF(&mystr, "M[%i]", index(k, l));
							mxsgn = -1;
							mysgn = -1;
						}
					} else {
						mreal = true;
						ASPRINTF(&mxstr, "M[%i]", index(k, 0));
					}
					const auto csgn = [](int i) {
						return i > 0 ? '+' : '-';
					};
					const auto add_work = [&pos_real,&pos_imag,&neg_real,&neg_imag](int sgn, int m, char* mstr, char* gstr) {
						if( sgn == 1) {
							if( m >= 0 ) {
								pos_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
							} else {
								pos_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
							}
						} else {
							if( m >= 0 ) {
								neg_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
							} else {
								neg_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
							}
						}
					};
					if (!mreal) {
						if (!greal) {
							add_work(mxsgn * gxsgn, 1, mxstr, gxstr);
							add_work(-mysgn * gysgn, 1, mystr, gystr);
							flops += 4;
							if (m > 0) {
								add_work(mysgn * gxsgn, -1, mystr, gxstr);
								add_work(mxsgn * gysgn, -1, mxstr, gystr);
							}
						} else {
							add_work(mxsgn * gxsgn, 1, mxstr, gxstr);
							if (m > 0) {
								add_work(mysgn * gxsgn, -1, mystr, gxstr);
							}
						}
					} else {
						if (!greal) {
							add_work(mxsgn * gxsgn, 1, mxstr, gxstr);
							if (m > 0) {
								add_work(mxsgn * gysgn, -1, mxstr, gystr);
							}
						} else {
							add_work(mxsgn * gxsgn, 1, mxstr, gxstr);
						}
					}
					if (gxstr) {
						free(gxstr);
					}
					if (gystr) {
						free(gystr);
					}
					if (mxstr) {
						free(mxstr);
					}
					if (mystr) {
						free(mystr);
					}
				}
				//		if(!fmaops) {
				//	}

			}
			if (fmaops && neg_real.size() >= 2) {
				tprint(sw, "L[%i] = -L[%i];\n", index(n, m), index(n, m));
				fps.r++;
				for (int i = 0; i < neg_real.size(); i++) {
					tprint(sw, "L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), index(n, m));
					fps.f++;
					flops++;
				}
				tprint(sw, "L[%i] = -L[%i];\n", index(n, m), index(n, m));
				fps.r++;
				flops += 2;
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint(sw, "L[%i] -= %s * %s;\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
					fps.r += 2;
					flops += 2;
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint(sw, "L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), index(n, m));
				fps.f++;
				flops += 2 - fmaops;
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint(sw, "L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
				fps.r++;
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint(sw, "L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), index(n, -m));
					fps.f++;
					flops++;
				}
				tprint(sw, "L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
				fps.r++;
				flops += 2;
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint(sw, "L[%i] -= %s * %s;\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
					fps.r += 2;
					flops += 2;
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint(sw, "L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), index(n, -m));
				fps.f++;
				flops += 2 - fmaops;
			}

		}
		if (n == 0) {
			deindent();
			tprint(sw, "}\n");
			tprint_flush();
		}
	}
	tprint_flush();
	return flops;
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
			O[cindex(n, m)] = (double(2 * n - 1) * z * O[cindex(n - 1, m)] - double((n - 1) * (n - 1) - m * m) * r2inv * O[cindex(n - 2, m)]);
		}
	}
	return O;
}

bool close2zero(double a) {
	return abs(a) < 1e-10;
}

int p2l(int P, flops_t& fps) {
	int flops = 0;
	func_header("P2L", P, true, false, true, "", "L", EXP, "M", LIT, "x", LIT, "y", LIT, "z", LIT);
	init_real("tmp1");
	tprint("tmp1 = TCAST(1) / L_st.scale();\n");
	fps.d++;
	tprint("x *= tmp1;\n");
	fps.r++;
	tprint("y *= tmp1;\n");
	fps.r++;
	tprint("z *= tmp1;\n");
	fps.r++;
	tprint("expansion_type<%s,%i> O_st(L_st.scale());\n", type.c_str(), P);
	tprint("T* O = O_st.data();\n");
	flops += greens_body(P, fps, "M");
	tprint("L_st += O_st;\n");
	fps.r += exp(P);
	flops += exp_sz(P);
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return flops;
}

int greens(int P, flops_t& fps) {
	int flops = 0;
	func_header("greens", P, true, false, true, "", "O", EXP, "x", LIT, "y", LIT, "z", LIT);
	flops += greens_body(P, fps);
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}");
	tprint("\n");
	return flops;
}

int greens_xz(int P, flops_t& fps) {
	int flops = 0;
	func_header("greens_xz", P, false, false, true, "", "O", EXP, "x", LIT, "z", LIT, "r2", LIT);
	init_real("ax");
	init_real("ay");
	init_real("r2inv");
	tprint("O[0] = detail::rsqrt(r2);\n");
	fps += flops_map[std::string("rsqrt_") + type];
	tprint("r2inv = O[0] * O[0];\n");
	fps.r++;
	flops += sqrt_flops;
	tprint("O_st.trace2() = TCAST(0);\n");
	fps.a++;
	tprint("x *= r2inv;\n");
	fps.r++;
	flops += 1;
	tprint("z *= r2inv;\n");
	fps.r++;
	flops += 1;
	const auto index = [](int l, int m) {
		return l*(l+1)/2+m;
	};
	tprint("r2inv = -r2inv;\n");
	fps.r++;
	for (int m = 0; m <= P; m++) {
		if (m == 1) {
			tprint("O[%i] = x * O[0];\n", index(m, m));
			fps.r++;
			flops += 1;
		} else if (m > 0) {
			tprint("ax = O[%i] * TCAST(%i);\n", index(m - 1, m - 1), 2 * m - 1);
			fps.r++;
			tprint("O[%i] = x * ax;\n", index(m, m));
			fps.r++;
			flops += 2;
		}
		if (m + 1 <= P) {
			tprint("O[%i] = TCAST(%i) * z * O[%i];\n", index(m + 1, m), 2 * m + 1, index(m, m));
			fps.r += 2;
			flops += 2;
		}
		for (int n = m + 2; n <= P; n++) {
			if (m != 0) {
				tprint("ax = TCAST(%i) * z;\n", 2 * n - 1);
				fps.r++;
				tprint("ay = TCAST(%i) * r2inv;\n", (n - 1) * (n - 1) - m * m);
				fps.r++;
				tprint("O[%i] = detail::fma(ax, O[%i], ay * O[%i]);\n", index(n, m), index(n - 1, m), index(n - 2, m));
				fps.r++;
				fps.f++;
				flops += 5 - fmaops;
			} else {
				if ((n - 1) * (n - 1) - m * m == 1) {
					tprint("O[%i] = detail::fma(TCAST(%i) * z, O[%i], r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), index(n - 2, m));
					fps.f++;
					fps.r += 2;

				} else {
					tprint("O[%i] = detail::fma(TCAST(%i) * z, O[%i], TCAST(%i) * r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m),
							(n - 1) * (n - 1) - m * m, index(n - 2, m));
					fps.f++;
					fps.r += 3;
				}
			}
		}
	}
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("}\n");
	tprint("\n");
	return flops;
}

flops_t rescale_flops(int P) {
	flops_t fps;
	fps.a++;
	fps.r += (P + 1) * (P + 2) + 1;
	fps.d++;
	return fps;
}

flops_t init_flops(int P) {
	flops_t fps;
	fps.r += (P + 1) * (P + 1) + 1;
	return fps;
}

flops_t copy_flops(int P) {
	flops_t fps;
	fps.r += (P + 1) * (P + 1) + 2;
	return fps;
}

flops_t accumulate_flops(int P) {
	flops_t fps;
	fps = rescale_flops(P);
	fps.r += (P + 1) * (P + 1) + 1;
	fps.a++;
	return fps;
}

int M2L_ewald(int P, flops_t& fps) {
	int flops = 0;
	if (periodic) {
		func_header("M2L_ewald", P, true, false, true, "", "L0", EXP, "M0", CMUL, "x", LIT, "y", LIT, "z", LIT);
		tprint("expansion_type<%s, %i> G_st;\n", type.c_str(), P);
		tprint("T* G = G_st.data();\n", type.c_str(), P);
		tprint("auto M_st = M0_st;\n");
		fps += copy_flops(P - 1);
		tprint("expansion_type<%s,%i> L_st;\n", type.c_str(), P);
		tprint("auto* M = M_st.data();\n");
		tprint("auto* L = M_st.data();\n");
		tprint("L_st.init();\n");
		fps += init_flops(P);
		tprint("M_st.rescale(TCAST(1));\n");
		fps += rescale_flops(P - 1);
		tprint("greens_ewald(G_st, x, y, z, flags);\n");
		fps += flops_map[std::string("greens_ewald_") + type];
		tprint("M2LG(L_st, M_st, G_st, flags);\n");
		fps += flops_map[std::string("M2LG_") + type];
		tprint("L0_st += L_st;\n");
		fps += accumulate_flops(P);
		deindent();
		tprint("}");
		tprint("\n");
		tprint("}\n");
		tprint("\n");
	}
	return flops;
}

int m2lg(int P, int Q, flops_t fps) {
	int flops = 0;
	func_header("M2LG", P, true, true, true, "", "L", EXP, "M", CMUL, "O", EXP);
	flops += m2lg_body(P, Q, fps);
	if (!nophi && P > 2 && periodic) {
		tprint("if( calcpot ) {\n");
		indent();
		tprint("L[%i] = detail::fma(TCAST(-0.5) * O_st.trace2(), M_st.trace2(), L[%i]);\n", index(0, 0), index(0, 0));
		fps.f++;
		fps.r++;
		deindent();
		tprint("}\n");
	}
	if (P > 1 && periodic) {
		tprint("L[%i] = detail::fma(TCAST(-2) * O_st.trace2(), M[%i], L[%i]);\n", index(1, -1), index(1, -1), index(1, -1));
		fps.f++;
		fps.r++;
		tprint("L[%i] -= O_st.trace2() * M[%i];\n", index(1, +0), index(1, +0), index(1, +0));
		fps.r += 2;
		tprint("L[%i] = detail::fma(TCAST(-2) * O_st.trace2(), M[%i], L[%i]);\n", index(1, +1), index(1, +1), index(1, +1));
		fps.f++;
		fps.r++;
		tprint("L_st.trace2() = detail::fma(TCAST(-0.5) * O_st.trace2(), M[%i], L_st.trace2());\n", index(0, 0));
		fps.f++;
		fps.r++;
		flops += 10 - 3 * fmaops;
	}
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	fps += flops_map[std::string("M2LG_") + type] = fps;
	return flops;
}

int greens_ewald(int P, double alpha, flops_t& fps) {
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
	int flops = 0;
	func_header("greens_ewald", P, true, true, true, "", "G", EXP, "x0", LIT, "y0", LIT, "z0", LIT);
	const auto c = tprint_on;

//set_tprint(false);
//flops += greens(P);
//set_tprint(c);
	tprint("expansion_type<%s, %i> Gr_st;\n", type.c_str(), P);
	tprint("T* Gr = Gr_st.data();\n", type.c_str(), P);
	set_tprint(false);
	flops += greens(P, fps);
	set_tprint(c);
	int cnt = 0;
	auto fps0 = fps;
	for (int ix = -R; ix <= R; ix++) {
		for (int iy = -R; iy <= R; iy++) {
			for (int iz = -R; iz <= R; iz++) {
				const int i2 = ix * ix + iy * iy + iz * iz;
				if (i2 > R2 && i2 != 0) {
					continue;
				}
				fps += fps0;
				cnt++;
			}
		}
	}
	flops *= cnt + 1;
	bool first = true;
	int eflops;
	init_real("sw");
	init_real("r");
	init_real("r2");
	init_real("xxx");
	init_real("tmp0");
	init_real("tmp1");
	init_real("gam1");
	init_real("exp0");
	init_real("xfac");
	init_real("xpow");
	init_real("gam0inv");
	init_real("gam");
	init_real("x");
	init_real("y");
	init_real("z");
	init_real("x2");
	init_real("x2y2");
	init_real("hdotx");
	init_real("phi");
	const auto name = [](const char* base, int hx, int hy, int hz) {
		std::string s = base;
		const auto add_symbol = [&s](int h) {
			if( h < 0 ) {
				s += "n";
				s += std::to_string(-h);
			} else if( h > 0 ) {
				s += "p";
				s += std::to_string(h);
			} else {
				s += "x0";
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
	tprint("int ix, iy, iz, ii;\n");
	tprint("r2 = detail::fma(x0, x0, detail::fma(y0, y0, z0 * z0));\n");
	fps.f += 2;
	fps.r++;
	flops += 5 - 2 * fmaops;
	tprint("r = detail::sqrt(r2);\n");
	fps += flops_map[std::string("sqrt_") + type];
	tprint("greens(Gr_st, x0, y0, z0, flags);\n");
	tprint("xxx = TCAST(%.20e) * r;\n", alpha);
	fps.r++;
	flops++;
	tprint("detail::erfcexp(xxx, &gam1, &exp0);\n");
	fps += flops_map[std::string("erfcexp_") + type];
	flops += erfcexp_flops;
	tprint("gam1 *= TCAST(%.20e);\n", sqrt(M_PI));
	fps.r++;
	flops += 1;
	tprint("xfac = TCAST(%.20e) * r2;\n", alpha * alpha);
	fps.r++;
	flops += 1;
	tprint("xpow = TCAST(%.20e) * r;\n", alpha);
	fps.r++;
	flops++;
	double gam0inv = 1.0 / sqrt(M_PI);
	tprint("sw = r2 > TCAST(0);\n");
	fps.r++;
	fps.i += is_vec(type);
	flops += 1;
	for (int l = 0; l <= P; l++) {
		tprint("gam = gam1 * TCAST(%.20e);\n", gam0inv);
		fps.r++;
		flops++;
		for (int m = -l; m <= l; m++) {
			tprint("G[%i] = sw*(TCAST(%.1e) - gam) * Gr[%i];\n", index(l, m), nonepow<double>(l), index(l, m));
			fps.r += 3;
		}
		if (l == 0) {
			tprint("G[%i] += (TCAST(1) - sw)*TCAST(%.20e);\n", index(0, 0), (2) * alpha / sqrt(M_PI));
			fps.r += 3;
			flops += 3;
		}
		gam0inv *= 1.0 / -(l + 0.5);
		if (l != P) {
			tprint("gam1 = detail::fma(TCAST(%.20e), gam1, xpow * exp0);\n", l + 0.5);
			fps.r++;
			fps.f++;
			flops += 3;
			if (l != P - 1) {
				tprint("xpow *= xfac;\n");
				fps.r++;
				flops++;
			}
		}
	}
	fps0.reset();
	tprint("for (ix = -%i; ix <= %i; ix++) {\n", R, R);
	indent();
	tprint("for (iy = -%i; iy <= %i; iy++) {\n", R, R);
	indent();
	tprint("for (iz = -%i; iz <= %i; iz++) {\n", R, R);
	indent();
	tprint("ii = ix * ix + iy * iy + iz * iz;\n");
	tprint("if (ii > %i || ii == 0) {\n", R2);
	indent();
	tprint("continue;\n");
	deindent();
	tprint("}\n");
	tprint("x = x0 - T(ix);\n");
	fps0.r += 2;
	flops++;
	tprint("y = y0 - T(iy);\n");
	fps0.r += 2;
	flops++;
	tprint("z = z0 - T(iz);\n");
	fps0.r += 2;
	tprint("r2 = detail::fma(x, x, detail::fma(y, y, z * z));\n");
	fps0.r++;
	fps0.f++;
	flops += 3 * cnt;
	tprint("r = detail::sqrt(r2);\n");
	fps0 += flops_map[std::string("sqrt_") + type];

	flops += sqrt_flops * cnt;
	tprint("greens(Gr_st, x, y, z, flags);\n");
	tprint("xxx = TCAST(%.20e) * r;\n", alpha);
	fps0.r++;
	flops += cnt;
	tprint("detail::erfcexp(xxx, &gam1, &exp0);\n");
	flops += erfcexp_flops * cnt;
	tprint("gam1 *= TCAST(%.20e);\n", -sqrt(M_PI));
	fps0.r++;
	flops += cnt;
	tprint("xfac = TCAST(%.20e) * r2;\n", alpha * alpha);
	fps0.r++;
	flops += cnt;
	tprint("xpow = TCAST(%.20e) * r;\n", alpha);
	fps0.r++;
	flops++;
	tprint("gam0inv = TCAST(%.20e);\n", 1.0 / sqrt(M_PI));
	fps0.a++;
	for (int l = 0; l <= P; l++) {
		tprint("gam = gam1 * gam0inv;\n");
		fps0.r++;
		flops += cnt;
		for (int m = -l; m <= l; m++) {
			tprint("G[%i] = detail::fma(gam, Gr[%i], G[%i]);\n", index(l, m), index(l, m), index(l, m));
			fps0.f++;
			flops += (2 - fmaops) * cnt;
		}
		if (l != P) {
			tprint("gam0inv *= TCAST(%.20e);\n", 1.0 / -(l + 0.5));
			fps0.r++;
			tprint("gam1 = TCAST(%.20e) * gam1 - xpow * exp0;\n", l + 0.5);
			fps0.r += 3;
			flops += (4 - fmaops) * cnt;
			if (l != P - 1) {
				tprint("xpow *= xfac;\n");
				fps0.r++;
				flops += cnt;
			}
		}
	}
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");

	int rflops = flops;
	for (int hx = -H; hx <= H; hx++) {
		if (hx) {
			if (abs(hx) == 1) {
				tprint("x2 = %cx0;\n", hx < 0 ? '-' : ' ');
				fps.a += hx >= 0;
				fps.r+=hx<0;
				flops += hx < 0;
			} else {
				tprint("x2 = TCAST(%i) * x0;\n", hx);
				fps.r++;
				flops++;
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
					tprint("x2y2 = x2 %c y0;\n", hy > 0 ? '+' : '-');
					fps.r++;
					flops++;
				} else {
					tprint("x2y2 = detail::fma(TCAST(%i), y0, x2);\n", hy);
					fps.f++;
					flops += 2 - fmaops;
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
						tprint("hdotx = x2y2 %c z0;\n", hz > 0 ? '+' : '-');
						fps.r++;
						flops++;
					} else {
						tprint("hdotx = detail::fma(TCAST(%i), z0, x2y2);\n", hz);
						fps.f++;
						flops += 2 - fmaops;
					}
				} else {
					tprint("hdotx = x2y2;\n", hz);
				}
				tprint("phi = TCAST(%.20e) * hdotx;\n", 2.0 * M_PI);
				fps.r++;
				flops++;
				tprint("detail::sincos(phi, &%s, &%s);\n", sinname(hx, hy, hz).c_str(), cosname(hx, hy, hz).c_str());
				fps += flops_map[std::string("sincos_") + type];
				flops += sincos_flops;
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
					const auto G0 = spherical_singular_harmonic(P, (double) hx, (double) hy, (double) hz);
					flops++;
					flops += 8;
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
								//						tprint("G[%i] += TCAST(%.20e) * %s;\n", index(l, m), -xsgn * c0 * G0[cindex(l, m)].real(), ax.c_str());
								ops[index(l, m)][fabs(a)].push_back(std::make_pair(copysign(1.0, -xsgn * a), ax));
								if (m != 0) {
									//								tprint("G[%i] += TCAST(%.20e) * %s;\n", index(l, -m), -ysgn * c0 * G0[cindex(l, m)].real(), ay.c_str());
									ops[index(l, -m)][fabs(a)].push_back(std::make_pair(copysign(1.0, -ysgn * a), ay));
								}
							}
							if (G0[cindex(l, m)].imag() != (0)) {
								const double a = c0 * G0[cindex(l, m)].imag();
//								tprint("G[%i] += TCAST(%.20e) * %s;\n", index(l, m), ysgn * c0 * G0[cindex(l, m)].imag(), ay.c_str());
								ops[index(l, -m)][fabs(a)].push_back(std::make_pair(copysign(1.0, ysgn * a), ay));
								if (m != 0) {
									//	tprint("G[%i] += TCAST(%.20e) * %s;\n", index(l, -m), -xsgn * c0 * G0[cindex(l, m)].imag(), ax.c_str());
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
			const std::vector<std::pair<int, std::string>>& these_ops = j->second;
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
	int sw = 0;
	tprint(sw, "if( calcpot ) {\n");
	indent();
	for (int ii = 0; ii < exp_sz(P); ii++) {
		std::vector<std::pair<double, std::vector<std::pair<int, std::string> > > > sorted_ops(ops[ii].begin(), ops[ii].end());
		std::sort(sorted_ops.begin(), sorted_ops.end(),
				[](const std::pair<double,std::vector<std::pair<int, std::string> > >& a, const std::pair<double,std::vector<std::pair<int, std::string> > > & b ) {
					return fabs(a.first) * sqrt(a.second.size()) < fabs(b.first) * sqrt(b.second.size());
				});
		if (ii > 0) {
			sw = sw == 0 ? 1 : 0;
		}
		for (auto j = sorted_ops.begin(); j != sorted_ops.end(); j++) {
			auto op = j->second;
			if (op.size()) {
				int sgn = op[0].first > 0 ? 1 : -1;
				for (int k = 0; k < op.size(); k++) {
					tprint(sw, "tmp%i %c= %s;\n", sw, k == 0 ? ' ' : (sgn * op[k].first > 0 ? '+' : '-'), op[k].second.c_str());
					fps.r ++;
					flops++;
				}
				if (sgn > 0) {
					tprint(sw, "G[%i] = detail::fma(TCAST(+%.20e), tmp%i, G[%i]);\n", ii, sgn * j->first, sw, ii);
					fps.f++;
				} else {
					tprint(sw, "G[%i] = detail::fma(TCAST(%.20e), tmp%i, G[%i]);\n ", ii, sgn * j->first, sw, ii);
					fps.f++;
				}
				flops += 2 - fmaops;
			}
		}
		if (ii == 0) {
			deindent();
			tprint(sw, "}\n");
			tprint_flush();
		}
	}
	tprint_flush();
	tprint("G_st.trace2() = TCAST(%.20e);\n", (4.0 * M_PI / 3.0));
	if (!nophi) {
		tprint("G[%i] += TCAST(%.20e);\n", index(0, 0), M_PI / (alpha * alpha));
		fps.r++;
		flops++;
	}
	int fflops = flops - rflops;
//	fprintf(stderr, "%e %i %i %i %i %i\n", alpha, R2, H2, rflops, fflops, rflops + fflops);
	deindent();
	tprint("/* flops = %i */\n", flops);
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
//	printf( "%i\n", flops);
	flops_map[std::string("greens_ewald_") + type] = fps;
	return flops;

}

int M2L_norot(int P, int Q, flops_t& fps) {
	int flops = 0;
	if (Q > 1) {
		func_header("M2L", P, true, true, true, "", "L", EXP, "M0", CMUL, "x", LIT, "y", LIT, "z", LIT);
	} else {
		func_header("M2P", P, true, true, true, "", "f", FORCE, "M0", CMUL, "x", LIT, "y", LIT, "z", LIT);
	}
	const auto c = tprint_on;
	set_tprint(false);
	flops += greens(P, fps);
	set_tprint(c);
	init_real("tmp1");
	init_real("rinv");
	init_real("r2inv");
	tprint("expansion_type<%s, %i> O_st;\n", type.c_str(), P);
	tprint("T* O = O_st.data();\n", type.c_str(), P);
	tprint("int n;\n");
	tprint("auto M_st = M0_st;\n");
	tprint("auto* M = M_st.data();\n");
	if (Q > 1) {
		tprint("M_st.rescale(L_st.scale());\n");
	}
	tprint("tmp1 = TCAST(1) / M_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");
	if (Q == 1) {
		init_reals("L", exp_sz(Q));
		tprint("for( n = 0; n < %i; n++) {\n", exp_sz(Q));
		indent();
		tprint("L[n] = TCAST(0);\n");
		deindent();
		tprint("}\n");
	}
	tprint("greens(O_st, x, y, z, flags);\n");
	flops += m2lg_body(P, Q, fps);
	if (Q == 1) {
		tprint("rinv = TCAST(1) / M_st.scale();\n");
		tprint("r2inv = rinv * rinv;\n");
		tprint("f.potential += L[0] * rinv;\n");
		tprint("f.force[0] -= L[3] * r2inv;\n");
		tprint("f.force[1] -= L[1] * r2inv;\n");
		tprint("f.force[2] -= L[2] * r2inv;\n");
	}
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return flops;
}

int M2L_rot1(int P, int Q) {
	flops_t fps;
	int flops = 0;
	if (Q > 1) {
		func_header("M2L", P, true, true, true, "", "L0", EXP, "M0", CMUL, "x", LIT, "y", LIT, "z", LIT);
	} else {
		func_header("M2P", P, true, true, true, "", "f", FORCE, "M0", CMUL, "x", LIT, "y", LIT, "z", LIT);
	}
	if (Q > 1) {
		tprint("int n;\n");
	}
	init_real("R2");
	init_real("Rzero");
	init_real("r2");
	init_real("tmp1");
	init_real("Rinv");
	if (Q == 1) {
		init_real("r2inv");
	}
	init_real("R");
	init_real("cosphi");
	init_real("sinphi");
	tprint("T rx[%i];\n", P);
	tprint("T ry[%i];\n", P);
	init_real("tmp0");
	init_reals("L", exp_sz(Q));
	tprint("expansion_type<%s, %i> O_st;\n", type.c_str(), P);
	tprint("T* O = O_st.data();\n", type.c_str(), P);
	tprint("auto M_st = M0_st;\n");
	tprint("auto* M = M_st.data();\n");
	if (Q == 1) {
		init_real("rinv");
	}
	if (Q > 1) {
		tprint("M_st.rescale(L0_st.scale());\n");
	}
	tprint("tmp1 = TCAST(1) / M_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");

	const auto tpo = tprint_on;
	set_tprint(false);
	flops += greens_xz(P, fps);
	set_tprint(tpo);
	tprint("R2 = detail::fma(x, x, y * y);\n");
	flops += 3 - fmaops;
	tprint("Rzero = T(R2<TCAST(1e-37));\n");
	flops++;
	tprint("r2 = detail::fma(z, z, R2);\n");
	flops++;
	tprint("tmp1 = R2 + Rzero;\n");
	flops++;
	tprint("Rinv = detail::rsqrt(tmp1);\n");
	flops + rsqrt_flops;
	tprint("R = TCAST(1) / Rinv;\n");
	flops += divops;
	tprint("cosphi = detail::fma(x, Rinv, Rzero);\n");
	flops += 2 - fmaops;
	tprint("sinphi = -y * Rinv;\n");
	flops += 2;
	flops += z_rot(P - 1, "M", false, false, false, fps);
	tprint("detail::greens_xz(O_st, R, z, r2, flags);\n");
	const auto oindex = [](int l, int m) {
		return l*(l+1)/2+m;
	};
	int sw = 0;
	tprint(sw, "if( calcpot ) {\n");
	indent();
	for (int n = 0; n <= Q; n++) {
		for (int m = 0; m <= n; m++) {
			if (n > 0) {
				sw = sw == 0 ? 1 : 0;
			}
			bool nfirst = true;
			bool pfirst = true;
			const int kmax = std::min(P - n, P - 1);
			for (int sgn = -1; sgn <= 1; sgn += 2) {
				for (int k = 0; k <= kmax; k++) {
					const int lmin = std::max(-k, -n - k - m);
					const int lmax = std::min(k, n + k - m);
					for (int l = lmin; l <= lmax; l++) {
						bool mreal = false;
						int gxsgn = 1;
						int mxsgn = 1;
						int mysgn = 1;
						char* gxstr = nullptr;
						char* mxstr = nullptr;
						char* mystr = nullptr;
						if (m + l > 0) {
							ASPRINTF(&gxstr, "O[%i]", oindex(n + k, m + l));
						} else if (m + l < 0) {
							if (abs(m + l) % 2 == 0) {
								ASPRINTF(&gxstr, "O[%i]", oindex(n + k, -m - l));
							} else {
								ASPRINTF(&gxstr, "O[%i]", oindex(n + k, -m - l));
								gxsgn = -1;
							}
						} else {
							ASPRINTF(&gxstr, "O[%i]", oindex(n + k, 0));
						}
						if (l > 0) {
							ASPRINTF(&mxstr, "M[%i]", index(k, l));
							ASPRINTF(&mystr, "M[%i]", index(k, -l));
							mysgn = -1;
						} else if (l < 0) {
							if (l % 2 == 0) {
								ASPRINTF(&mxstr, "M[%i]", index(k, -l));
								ASPRINTF(&mystr, "M[%i]", index(k, l));
							} else {
								ASPRINTF(&mxstr, "M[%i]", index(k, -l));
								ASPRINTF(&mystr, "M[%i]", index(k, l));
								mxsgn = -1;
								mysgn = -1;
							}
						} else {
							mreal = true;
							ASPRINTF(&mxstr, "M[%i]", index(k, 0));
						}
						if (!mreal) {
							if (mxsgn * gxsgn == sgn) {
								if (pfirst) {
									tprint(sw, "L[%i] = %s * %s;\n", index(n, m), mxstr, gxstr);
									pfirst = false;
									flops += 1;
								} else {
									tprint(sw, "L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, m), mxstr, gxstr, index(n, m));
									flops += 2 - fmaops;
								}
							}
							if (mysgn * gxsgn == sgn) {
								if (m > 0) {
									if (nfirst) {
										tprint(sw, "L[%i] = %s * %s;\n", index(n, -m), mystr, gxstr);
										nfirst = false;
										flops += 1;
									} else {
										tprint(sw, "L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, -m), mystr, gxstr, index(n, -m));
										flops += 2 - fmaops;
									}
								}
							}
						} else {
							if (mxsgn * gxsgn == sgn) {
								if (pfirst) {
									tprint(sw, "L[%i] = %s * %s;\n", index(n, m), mxstr, gxstr);
									pfirst = false;
									flops += 1;
								} else {
									tprint(sw, "L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, m), mxstr, gxstr, index(n, m));
									flops += 2 - fmaops;
								}
							}
						}
						if (gxstr) {
							free(gxstr);
						}
						if (mxstr) {
							free(mxstr);
						}
						if (mystr) {
							free(mystr);
						}
					}
				}
				if (!pfirst && sgn == -1) {
					tprint(sw, "L[%i] = -L[%i];\n", index(n, m), index(n, m));
					flops++;
				}
				if (!nfirst && sgn == -1) {
					tprint(sw, "L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
					flops++;
				}

			}
		}
		if (n == 0) {
			deindent();
			tprint(sw, "}\n");
			tprint_flush();
		}
	}
	tprint_flush();
	tprint("sinphi = -sinphi;\n");
	flops++;
	flops += z_rot(Q, "L", false, false, Q == P, fps);
	flops++;
	if (Q > 1) {
		tprint("for(n = 0; n < %i; n++) {\n", std::max(4, exp_sz(Q) - periodic));
		indent();
		tprint("L0[n] += L[n];\n");
		deindent();
		tprint("}\n");
	} else {
		tprint("rinv = TCAST(1) / M_st.scale();\n");
		tprint("r2inv = rinv * rinv;\n");
		tprint("f.potential += L[0] * rinv;\n");
		tprint("f.force[0] -= L[3] * r2inv;\n");
		tprint("f.force[1] -= L[1] * r2inv;\n");
		tprint("f.force[2] -= L[2] * r2inv;\n");
	}
	flops += (Q + 1) * (Q + 1);

	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return flops;
}

int M2L_rot2(int P, int Q) {
	int flops = 0;
	flops_t fps;
	if (Q > 1) {
		func_header("M2L", P, true, true, true, "", "L0", EXP, "M0", CMUL, "x", LIT, "y", LIT, "z", LIT);
	} else {
		func_header("M2P", P, true, true, true, "", "f", FORCE, "M0", CMUL, "x", LIT, "y", LIT, "z", LIT);
	}
	init_reals("L", exp_sz(Q));
	init_reals("A", 2 * P + 1);
	init_real("R2");
	init_real("Rzero");
	init_real("r2");
	init_real("rzero");
	init_real("tmp1");
	init_real("Rinv");
	if (Q == 1) {
		init_real("r2inv");
	}
	init_real("R");
	init_real("cosphi");
	init_real("sinphi");
	init_real("cosphi0");
	init_real("sinphi0");
	tprint("T rx[%i];\n", P);
	tprint("T ry[%i];\n", P);
	init_real("tmp0");
	init_real("r2przero");
	init_real("rinv");
	if (Q > 1) {
		tprint("int n;\n");
	}
	tprint("bool sw;\n");
	tprint("auto M_st = M0_st;\n");
	tprint("auto* M = M_st.data();\n");
	if (Q > 1) {
		tprint("M_st.rescale(L0_st.scale());\n");
	}
	tprint("tmp1 = TCAST(1) / M_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");
	tprint("R2 = detail::fma(x, x, y * y);\n");
	flops += 3 - fmaops;
	tprint("Rzero = T(R2<TCAST(1e-37));\n");
	flops++;
	tprint("tmp1 = R2 + Rzero;\n");
	flops++;
	tprint("Rinv = detail::rsqrt(tmp1);\n");
	flops += rsqrt_flops;
	tprint("R = TCAST(1) / Rinv;\n");
	flops += divops;
	tprint("r2 = (detail::fma(z, z, R2));\n");
	tprint("rzero = T(r2<TCAST(1e-37));\n");
//	tprint("r2inv = TCAST(1) / r2;\n");
	flops += 3 + divops - fmaops;
	tprint("r2przero = (r2 + rzero);\n");
	flops += 1;
	tprint("rinv = detail::rsqrt(r2przero);\n");
	flops += rsqrt_flops;

	tprint("cosphi = y * Rinv;\n");
	flops++;
	tprint("sinphi = detail::fma(x, Rinv, Rzero);\n");
	flops += 2 - fmaops;
	tprint("sw = false;\n");
	if (tprint_on) {
		fprintf(fp, "MULTIPOLE_ROTATION:\n");
	}
	flops_t fps1;
	flops += 2 * z_rot(P - 1, "M", false, false, false, fps1);
	fps += fps1;
	fps += fps1;
	tprint("if( sw ) {\n");
	indent();
	tprint("goto CONTINUE_AFTER_ROTATION;\n");
	deindent();
	tprint("}\n");
	flops += xz_swap(P - 1, "M", false, false, false, false, fps);

	tprint("cosphi0 = cosphi;\n");
	tprint("sinphi0 = sinphi;\n");
	tprint("cosphi = detail::fma(z, rinv, rzero);\n");
	flops += 2 - fmaops;
	tprint("sinphi = -R * rinv;\n");
	flops += 2;
	tprint("sw = true;\n");
	tprint("goto MULTIPOLE_ROTATION;\n");
	if (tprint_on) {
		fprintf(fp, "CONTINUE_AFTER_ROTATION:\n");
	}
	flops += xz_swap(P - 1, "M", false, true, false, false, fps);
	flops += m2l(P, Q, "M", "L", fps);
	flops += xz_swap(Q, "L", true, false, true, false, fps);

	tprint("sinphi = -sinphi;\n");
	flops += 1;
	flops += z_rot(Q, "L", true, false, false, fps);
	flops += xz_swap(Q, "L", true, false, false, true, fps);
	tprint("cosphi = cosphi0;\n");
	tprint("sinphi = -sinphi0;\n");
	flops += 1;
	flops += z_rot(Q, "L", false, true, false, fps);
	if (Q > 1) {
		tprint("for(n = 0; n < %i; n++) {\n", std::max(4, exp_sz(Q) - periodic));
		indent();
		tprint("L0[n] += L[n];\n");
		deindent();
		tprint("}\n");
	} else {
		tprint("rinv = TCAST(1) / M_st.scale();\n");
		tprint("r2inv = rinv * rinv;\n");
		tprint("f.potential += L[0] * rinv;\n");
		tprint("f.force[0] -= L[3] * r2inv;\n");
		tprint("f.force[1] -= L[1] * r2inv;\n");
		tprint("f.force[2] -= L[2] * r2inv;\n");
	}
	flops += (Q + 1) * (Q + 1);
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return flops;
}

int regular_harmonic(int P) {
	int flops = 0;
	func_header("regular_harmonic", P, false, false, true, "", "Y", EXP, "x", LIT, "y", LIT, "z", LIT);
	init_real("ax");
	init_real("ay");
	init_real("r2");
	tprint("r2 = detail::fma(x, x, detail::fma(y, y, z * z));\n");
	flops += 5 - 2 * fmaops;
	tprint("Y[0] = TCAST(1);\n");
	tprint("Y_st.trace2() = r2;\n");
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
//	Y[index(m, m)] = Y[index(m - 1, m - 1)] * R / TCAST(2 * m);
			if (m - 1 > 0) {
				tprint("ax = Y[%i] * TCAST(%.20e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("ay = Y[%i] * TCAST(%.20e);\n", index(m - 1, -(m - 1)), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax - y * ay;\n", index(m, m));
				tprint("Y[%i] = detail::fma(y, ax, x * ay);\n", index(m, -m));
				flops += 6 - fmaops;
			} else {
				tprint("ax = Y[%i] * TCAST(%.20e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax;\n", index(m, m));
				tprint("Y[%i] = y * ax;\n", index(m, -m));
				flops += 3;
			}
		}
		if (m + 1 <= P) {
//			Y[index(m + 1, m)] = z * Y[index(m, m)];
			if (m == 0) {
				tprint("Y[%i] = z * Y[%i];\n", index(m + 1, m), index(m, m));
				flops += 1;
			} else {
				tprint("Y[%i] = z * Y[%i];\n", index(m + 1, m), index(m, m));
				tprint("Y[%i] = z * Y[%i];\n", index(m + 1, -m), index(m, -m));
				flops += 2;
			}
		}
		for (int n = m + 2; n <= P; n++) {
			const double inv = double(1) / (double(n * n) - double(m * m));
//			Y[index(n, m)] = inv * (TCAST(2 * n - 1) * z * Y[index(n - 1, m)] - r2 * Y[index(n - 2, m)]);
			tprint("ax = TCAST(%.20e) * z;\n", inv * double(2 * n - 1));
			tprint("ay = TCAST(%.20e) * r2;\n", -(double) inv);
			tprint("Y[%i] = detail::fma(ax, Y[%i], ay * Y[%i]);\n", index(n, m), index(n - 1, m), index(n - 2, m));
			flops += 5 - fmaops;
			if (m != 0) {
				tprint("Y[%i] = detail::fma(ax, Y[%i], ay * Y[%i]);\n", index(n, -m), index(n - 1, -m), index(n - 2, -m));
				flops += 3 - fmaops;
			}
		}
	}
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("}\n");
	tprint("\n");
	return flops;
}

void cuda_header() {
	tprint("const int& tid = threadIdx.x;\n");
	tprint("const int& bsz = blockDim.x;\n");
}

int regular_harmonic_xz(int P) {
	int flops = 0;
	func_header("regular_harmonic_xz", P, false, false, true, "", "Y", EXP, "x", LIT, "z", LIT);
	init_real("ax");
	init_real("ay");
	init_real("r2");
	tprint("r2 = detail::fma(x, x, z * z);\n");
	flops += 3 - fmaops;
	tprint("Y[0] = TCAST(1);\n");
	tprint("Y_st.trace2() = r2;\n");
	const auto index = [](int l, int m) {
		return l*(l+1)/2+m;
	};
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
//	Y[index(m, m)] = Y[index(m - 1, m - 1)] * R / TCAST(2 * m);
			if (m - 1 > 0) {
				tprint("ax = Y[%i] * TCAST(%.20e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax;\n", index(m, m));
				flops += 2;
			} else {
				tprint("ax = Y[%i] * TCAST(%.20e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax;\n", index(m, m));
				flops += 2;
			}
		}
		if (m + 1 <= P) {
//			Y[index(m + 1, m)] = z * Y[index(m, m)];
			if (m == 0) {
				tprint("Y[%i] = z * Y[%i];\n", index(m + 1, m), index(m, m));
				flops += 1;
			} else {
				tprint("Y[%i] = z * Y[%i];\n", index(m + 1, m), index(m, m));
				flops += 1;
			}
		}
		for (int n = m + 2; n <= P; n++) {
			const double inv = double(1) / (double(n * n) - double(m * m));
//			Y[index(n, m)] = inv * (TCAST(2 * n - 1) * z * Y[index(n - 1, m)] - r2 * Y[index(n - 2, m)]);
			tprint("ax = TCAST(%.20e) * z;\n", inv * double(2 * n - 1));
			tprint("ay = TCAST(%.20e) * r2;\n", -(double) inv);
			tprint("Y[%i] = detail::fma(ax, Y[%i], ay * Y[%i]);\n", index(n, m), index(n - 1, m), index(n - 2, m));
			flops += 5 - fmaops;
		}
	}
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("}\n");
	tprint("\n");
	return flops;
}

int M2M_norot(int P) {
	int flops = 0;
	const auto c = tprint_on;
	set_tprint(false);
	flops += regular_harmonic(P);
	set_tprint(c);
	func_header("M2M", P + 1, true, true, true, "", "M", MUL, "x", LIT, "y", LIT, "z", LIT);
//const auto Y = spherical_regular_harmonic<T, P>(-x, -y, -z);
	init_real("tmp1");
	tprint("tmp1 = TCAST(1) / M_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");
	tprint("expansion_type<%s, %i> Y_st;\n", type.c_str(), P);
	tprint("T* Y = Y_st.data();\n", type.c_str(), P);
	tprint("detail::regular_harmonic(Y_st, -x, -y, -z, flags);\n");
	flops += 3;
	if (P > 1 && !nophi && periodic) {
		tprint("if( calcpot ) {\n");
		indent();
		tprint("M_st.trace2() = detail::fma(TCAST(-4) * x, M[%i], M_st.trace2());\n", index(1, 1));
		tprint("M_st.trace2() = detail::fma(TCAST(-4) * y, M[%i], M_st.trace2());\n", index(1, -1));
		tprint("M_st.trace2() = detail::fma(TCAST(-2) * z, M[%i], M_st.trace2());\n", index(1, 0));
		tprint("M_st.trace2() = detail::fma(x * x, M[%i], M_st.trace2());\n", index(0, 0));
		tprint("M_st.trace2() = detail::fma(y * y, M[%i], M_st.trace2());\n", index(0, 0));
		tprint("M_st.trace2() = detail::fma(z * z, M[%i], M_st.trace2());\n", index(0, 0));
		flops += 18 - 6 * fmaops;
		deindent();
		tprint("}\n");
	}
	for (int n = P; n >= 0; n--) {
		int sw = 0;
		for (int m = 0; m <= n; m++) {
			sw = sw == 0 ? 1 : 0;
			std::vector<std::pair<std::string, std::string>> pos_real;
			std::vector<std::pair<std::string, std::string>> neg_real;
			std::vector<std::pair<std::string, std::string>> pos_imag;
			std::vector<std::pair<std::string, std::string>> neg_imag;
			const auto add_work = [&pos_real,&pos_imag,&neg_real,&neg_imag](int sgn, int m, char* mstr, char* gstr) {
				if( sgn == 1) {
					if( m >= 0 ) {
						pos_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					} else {
						pos_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					}
				} else {
					if( m >= 0 ) {
						neg_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					} else {
						neg_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					}
				}
			};
			for (int k = 1; k <= n; k++) {
				const int lmin = std::max(-k, m - n + k);
				const int lmax = std::min(k, m + n - k);
				for (int l = -k; l <= k; l++) {
					char* mxstr = nullptr;
					char* mystr = nullptr;
					char* gxstr = nullptr;
					char* gystr = nullptr;
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
					if (m - l > 0) {
						ASPRINTF(&mxstr, "M[%i]", index(n - k, abs(m - l)));
						ASPRINTF(&mystr, "M[%i]", index(n - k, -abs(m - l)));
					} else if (m - l < 0) {
						if (abs(m - l) % 2 == 0) {
							ASPRINTF(&mxstr, "M[%i]", index(n - k, abs(m - l)));
							ASPRINTF(&mystr, "M[%i]", index(n - k, -abs(m - l)));
							mysgn = -1;
						} else {
							ASPRINTF(&mxstr, "M[%i]", index(n - k, abs(m - l)));
							ASPRINTF(&mystr, "M[%i]", index(n - k, -abs(m - l)));
							mxsgn = -1;
						}
					} else {
						ASPRINTF(&mxstr, "M[%i]", index(n - k, 0));
					}
					if (l > 0) {
						ASPRINTF(&gxstr, "Y[%i]", index(k, abs(l)));
						ASPRINTF(&gystr, "Y[%i]", index(k, -abs(l)));
					} else if (l < 0) {
						if (abs(l) % 2 == 0) {
							ASPRINTF(&gxstr, "Y[%i]", index(k, abs(l)));
							ASPRINTF(&gystr, "Y[%i]", index(k, -abs(l)));
							gysgn = -1;
						} else {
							ASPRINTF(&gxstr, "Y[%i]", index(k, abs(l)));
							ASPRINTF(&gystr, "Y[%i]", index(k, -abs(l)));
							gxsgn = -1;
						}
					} else {
						ASPRINTF(&gxstr, "Y[%i]", index(k, 0));
					}
					add_work(mxsgn * gxsgn, +1, mxstr, gxstr);
					if (gystr && mystr) {
						add_work(-mysgn * gysgn, 1, mystr, gystr);
					}
					if (m > 0) {
						if (gystr) {
							add_work(mxsgn * gysgn, -1, mxstr, gystr);
						}
						if (mystr) {
							add_work(mysgn * gxsgn, -1, mystr, gxstr);
						}
					}
					if (gxstr) {
						free(gxstr);
					}
					if (mxstr) {
						free(mxstr);
					}
					if (gystr) {
						free(gystr);
					}
					if (mystr) {
						free(mystr);
					}
				}
			}
			if (fmaops && neg_real.size() >= 2) {
				tprint(sw, "M[%i] = -M[%i];\n", index(n, m), index(n, m));
				for (int i = 0; i < neg_real.size(); i++) {
					tprint(sw, "M[%i] = detail::fma(%s, %s, M[%i]);\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), index(n, m));
					flops++;
				}
				tprint(sw, "M[%i] = -M[%i];\n", index(n, m), index(n, m));
				flops += 2;
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint(sw, "M[%i] -= %s * %s;\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint(sw, "M[%i] = detail::fma(%s, %s, M[%i]);\n", index(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), index(n, m));
				flops += 2 - fmaops;
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint(sw, "M[%i] = -M[%i];\n", index(n, -m), index(n, -m));
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint(sw, "M[%i] = detail::fma(%s, %s, M[%i]);\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), index(n, -m));
					flops++;
				}
				tprint(sw, "M[%i] = -M[%i];\n", index(n, -m), index(n, -m));
				flops += 2;
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint(sw, "M[%i] -= %s * %s;\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint(sw, "M[%i] = detail::fma(%s, %s, M[%i]);\n", index(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), index(n, -m));
				flops += 2 - fmaops;
			}
		}
		tprint_flush();
	}
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return flops;
}

int M2M_rot1(int P) {
	int flops = 0;
	flops_t fps;
	const auto c = tprint_on;
	set_tprint(false);
	flops += regular_harmonic_xz(P);
	set_tprint(c);

	func_header("M2M", P + 1, true, true, true, "", "M", MUL, "x", LIT, "y", LIT, "z", LIT);
	tprint("expansion_type<%s, %i> Y_st;\n", type.c_str(), P);
	tprint("T* Y = Y_st.data();\n", type.c_str(), P);
	tprint("T rx[%i];\n", P);
	tprint("T ry[%i];\n", P);
	init_real("tmp0");
	init_real("R");
	init_real("Rinv");
	init_real("tmp1");
	init_real("R2");
	init_real("Rzero");
	init_real("cosphi");
	init_real("sinphi");
	tprint("tmp1 = TCAST(1) / M_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");
	tprint("R2 = detail::fma(x, x, y * y);\n");
	flops += 3 - fmaops;
	tprint("Rzero = (R2<TCAST(1e-37));\n");
	tprint("tmp1 = R2 + Rzero;\n");
	tprint("Rinv = detail::rsqrt(tmp1);\n");
	flops += rsqrt_flops;
	flops++;
	tprint("R = TCAST(1) / Rinv;\n");
	flops += divops;
	tprint("cosphi = detail::fma(x, Rinv, Rzero);\n");
	flops += 2 - fmaops;
	tprint("sinphi = -y * Rinv;\n");
	flops += 2;
	flops += z_rot(P, "M", false, false, false, fps);
	if (P > 1 && !nophi && periodic) {
		tprint("if( calcpot ) {\n");
		indent();
		tprint("M_st.trace2() = detail::fma(TCAST(-4)*R, M[%i], M_st.trace2());\n", index(1, 1));
		tprint("M_st.trace2() = detail::fma(TCAST(-2)*z, M[%i], M_st.trace2());\n", index(1, 0));
		tprint("M_st.trace2() = detail::fma(R * R, M[%i], M_st.trace2());\n", index(0, 0));
		tprint("M_st.trace2() = detail::fma(z * z, M[%i], M_st.trace2());\n", index(0, 0));
		deindent();
		tprint("}\n");
		flops += 12;
	}
	const auto yindex = [](int l, int m) {
		return l*(l+1)/2+m;
	};

	tprint("detail::regular_harmonic_xz(Y_st, -R, -z, flags);\n");
	flops += 2;
	for (int n = P; n >= 0; n--) {
		for (int m = 0; m <= n; m++) {
			std::vector<std::pair<std::string, std::string>> pos_real;
			std::vector<std::pair<std::string, std::string>> neg_real;
			std::vector<std::pair<std::string, std::string>> pos_imag;
			std::vector<std::pair<std::string, std::string>> neg_imag;
			const auto add_work = [&pos_real,&pos_imag,&neg_real,&neg_imag](int sgn, int m, char* mstr, char* gstr) {
				if( sgn == 1) {
					if( m >= 0 ) {
						pos_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					} else {
						pos_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					}
				} else {
					if( m >= 0 ) {
						neg_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					} else {
						neg_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					}
				}
			};
			for (int k = 1; k <= n; k++) {
				const int lmin = std::max(-k, m - n + k);
				const int lmax = std::min(k, m + n - k);
				for (int l = -k; l <= k; l++) {
					char* mxstr = nullptr;
					char* mystr = nullptr;
					char* gxstr = nullptr;
					int mxsgn = 1;
					int mysgn = 1;
					int gxsgn = 1;
					if (abs(m - l) > n - k) {
						continue;
					}
					if (-abs(m - l) < k - n) {
						continue;
					}
					if (m - l > 0) {
						ASPRINTF(&mxstr, "M[%i]", index(n - k, abs(m - l)));
						ASPRINTF(&mystr, "M[%i]", index(n - k, -abs(m - l)));
					} else if (m - l < 0) {
						if (abs(m - l) % 2 == 0) {
							ASPRINTF(&mxstr, "M[%i]", index(n - k, abs(m - l)));
							ASPRINTF(&mystr, "M[%i]", index(n - k, -abs(m - l)));
							mysgn = -1;
						} else {
							ASPRINTF(&mxstr, "M[%i]", index(n - k, abs(m - l)));
							ASPRINTF(&mystr, "M[%i]", index(n - k, -abs(m - l)));
							mxsgn = -1;
						}
					} else {
						ASPRINTF(&mxstr, "M[%i]", index(n - k, 0));
					}
					ASPRINTF(&gxstr, "Y[%i]", yindex(k, abs(l)));
					if (l < 0 && abs(l) % 2 != 0) {
						gxsgn = -1;
					}
					add_work(mxsgn * gxsgn, +1, mxstr, gxstr);
					if (m > 0) {
						if (mystr) {
							add_work(mysgn * gxsgn, -1, mystr, gxstr);
						}
					}
					if (gxstr) {
						free(gxstr);
					}
					if (mxstr) {
						free(mxstr);
					}
					if (mystr) {
						free(mystr);
					}
				}
			}
			if (fmaops && neg_real.size() >= 2) {
				tprint("M[%i] = -M[%i];\n", index(n, m), index(n, m));
				for (int i = 0; i < neg_real.size(); i++) {
					tprint("M[%i] = detail::fma(%s, %s, M[%i]);\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), index(n, m));
					flops++;
				}
				tprint("M[%i] = -M[%i];\n", index(n, m), index(n, m));
				flops += 2;
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint("M[%i] -= %s * %s;\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint("M[%i] = detail::fma(%s, %s, M[%i]);\n", index(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), index(n, m));
				flops += 2 - fmaops;
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint("M[%i] = -M[%i];\n", index(n, -m), index(n, -m));
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint("M[%i] = detail::fma(%s, %s, M[%i]);\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), index(n, -m));
					flops++;
				}
				tprint("M[%i] = -M[%i];\n", index(n, -m), index(n, -m));
				flops += 2;
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint("M[%i] -= %s * %s;\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint("M[%i] = detail::fma(%s, %s, M[%i]);\n", index(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), index(n, -m));
				flops += 2 - fmaops;
			}
		}
	}
	tprint("sinphi = -sinphi;\n");
	flops++;
	flops += z_rot(P, "M", false, false, false, fps);
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return flops;
}

int M2M_rot2(int P) {
	int flops = 0;
	flops_t fps;
	func_header("M2M", P + 1, true, true, true, "", "M", MUL, "x", LIT, "y", LIT, "z", LIT);
	init_reals("A", 2 * P + 1);
	tprint("T rx[%i];\n", P);
	tprint("T ry[%i];\n", P);
	init_real("tmp0");
	init_real("R");
	init_real("Rinv");
	init_real("R2");
	init_real("Rzero");
	init_real("tmp1");
	init_real("r2");
	init_real("rzero");
	init_real("r");
	init_real("rinv");
	init_real("cosphi");
	init_real("cosphi0");
	init_real("sinphi");
	init_real("sinphi0");
	tprint("tmp1 = TCAST(1) / M_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");
	tprint("R2 = detail::fma(x, x, y * y);\n");
	flops += 3 - fmaops;
	tprint("Rzero = T(R2<TCAST(1e-37));\n");
	flops++;
	tprint("tmp1 = R2 + Rzero;\n");
	flops++;
	tprint("Rinv = detail::rsqrt(tmp1);\n");
	flops += rsqrt_flops;
	tprint("r2 = detail::fma(z, z, R2);\n");
	flops += 2 - fmaops;
	tprint("rzero = T(r2<TCAST(1e-37));\n");
	flops += 1;
	tprint("tmp1 = r2 + rzero;\n");
	tprint("rinv = detail::rsqrt(tmp1);\n");
	flops += rsqrt_flops;
	tprint("R = TCAST(1) / Rinv;\n");
	flops += divops;
	tprint("r = TCAST(1) / rinv;\n");
	flops += divops;
	tprint("cosphi = y * Rinv;\n");
	flops++;
	tprint("sinphi = detail::fma(x, Rinv, Rzero);\n");
	flops += 2 - fmaops;
	flops += z_rot(P, "M", false, false, false, fps);
	flops += xz_swap(P, "M", false, false, false, false, fps);
	tprint("cosphi0 = cosphi;\n");
	tprint("sinphi0 = sinphi;\n");
	tprint("cosphi = detail::fma(z, rinv, rzero);\n");
	flops += 2 - fmaops;
	tprint("sinphi = -R * rinv;\n");
	flops += z_rot(P, "M", false, false, false, fps);
	flops += xz_swap(P, "M", false, false, false, false, fps);
	if (P > 1 && !nophi && periodic) {
		tprint("if( calcpot ) {\n");
		indent();
		tprint("M_st.trace2() = detail::fma(TCAST(-2) * r, M[%i], M_st.trace2());\n", index(1, 0));
		tprint("M_st.trace2() = detail::fma(r * r, M[%i], M_st.trace2());\n", index(0, 0));
		flops += 6 - 2 * fmaops;
		deindent();
		tprint("}\n");
	}
	tprint("A[0] = TCAST(1);\n");
	for (int n = 1; n <= P; n++) {
		tprint("A[%i] = -r * A[%i];\n", n, n - 1);
		flops += 2;
	}
	for (int n = 2; n <= P; n++) {
		tprint("A[%i] *= TCAST(%.20e);\n", n, 1.0 / factorial(n));
		flops += 1;
	}
	for (int n = P; n >= 0; n--) {
		for (int m = 0; m <= n; m++) {
			for (int k = 1; k <= n; k++) {
				if (abs(m) > n - k) {
					continue;
				}
				if (-abs(m) < k - n) {
					continue;
				}
				tprint("M[%i] = detail::fma(M[%i], A[%i], M[%i]);\n", index(n, m), index(n - k, m), k, index(n, m));
				flops += 2 - fmaops;
				if (m > 0) {
					tprint("M[%i] = detail::fma(M[%i], A[%i], M[%i]);\n", index(n, -m), index(n - k, -m), k, index(n, -m));
					flops += 2 - fmaops;
				}
			}

		}
	}
	flops += xz_swap(P, "M", false, false, false, false, fps);
	tprint("sinphi = -sinphi;\n");
	flops += z_rot(P, "M", false, false, false, fps);
	flops += xz_swap(P, "M", false, false, false, false, fps);
	tprint("cosphi = cosphi0;\n");
	tprint("sinphi = -sinphi0;\n");
	flops += 1;
	flops += z_rot(P, "M", false, false, false, fps);
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return flops;
}

int L2L_norot(int P) {
	int flops = 0;
	const auto c = tprint_on;
	set_tprint(false);
	flops += regular_harmonic(P);
	set_tprint(c);
	func_header("L2L", P, true, true, true, "", "L", EXP, "x", LIT, "y", LIT, "z", LIT);
	init_real("tmp1");
	tprint("tmp1 = TCAST(1) / L_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");
	tprint("expansion_type<%s, %i> Y_st;\n", type.c_str(), P);
	tprint("T* Y = Y_st.data();\n", type.c_str(), P);
	tprint("detail::regular_harmonic(Y_st, -x, -y, -z, flags);\n");
	flops += 3;
	int sw = 0;
	tprint(sw, "if( calcpot ) {\n");
	indent();
	for (int n = 0; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			if (n > 0) {
				sw = sw == 0 ? 1 : 0;
			}
			std::vector<std::pair<std::string, std::string>> pos_real;
			std::vector<std::pair<std::string, std::string>> neg_real;
			std::vector<std::pair<std::string, std::string>> pos_imag;
			std::vector<std::pair<std::string, std::string>> neg_imag;
			const auto add_work = [&pos_real,&pos_imag,&neg_real,&neg_imag](int sgn, int m, char* mstr, char* gstr) {
				if( sgn == 1) {
					if( m >= 0 ) {
						pos_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					} else {
						pos_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					}
				} else {
					if( m >= 0 ) {
						neg_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					} else {
						neg_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					}
				}
			};
			for (int k = 1; k <= P - n; k++) {
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					char* mxstr = nullptr;
					char* mystr = nullptr;
					char* gxstr = nullptr;
					char* gystr = nullptr;
					int mxsgn = 1;
					int mysgn = 1;
					int gxsgn = 1;
					int gysgn = 1;
					if (m + l > 0) {
						ASPRINTF(&mxstr, "L[%i]", index(n + k, abs(m + l)));
						ASPRINTF(&mystr, "L[%i]", index(n + k, -abs(m + l)));
					} else if (m + l < 0) {
						if (abs(m + l) % 2 == 0) {
							ASPRINTF(&mxstr, "L[%i]", index(n + k, abs(m + l)));
							ASPRINTF(&mystr, "L[%i]", index(n + k, -abs(m + l)));
							mysgn = -1;
						} else {
							ASPRINTF(&mxstr, "L[%i]", index(n + k, abs(m + l)));
							ASPRINTF(&mystr, "L[%i]", index(n + k, -abs(m + l)));
							mxsgn = -1;
						}
					} else {
						ASPRINTF(&mxstr, "L[%i]", index(n + k, 0));
					}
					if (l > 0) {
						ASPRINTF(&gxstr, "Y[%i]", index(k, abs(l)));
						ASPRINTF(&gystr, "Y[%i]", index(k, -abs(l)));
						gysgn = -1;
					} else if (l < 0) {
						if (abs(l) % 2 == 0) {
							ASPRINTF(&gxstr, "Y[%i]", index(k, abs(l)));
							ASPRINTF(&gystr, "Y[%i]", index(k, -abs(l)));
						} else {
							ASPRINTF(&gxstr, "Y[%i]", index(k, abs(l)));
							ASPRINTF(&gystr, "Y[%i]", index(k, -abs(l)));
							gxsgn = -1;
							gysgn = -1;
						}
					} else {
						ASPRINTF(&gxstr, "Y[%i]", index(k, 0));
					}
					const auto csgn = [](int i) {
						return i > 0 ? '+' : '-';
					};

					add_work(mxsgn * gxsgn, +1, mxstr, gxstr);
					if (gystr && mystr) {
						add_work(-mysgn * gysgn, 1, mystr, gystr);
					}
					if (m > 0) {
						if (gystr) {
							add_work(mxsgn * gysgn, -1, mxstr, gystr);
						}
						if (mystr) {
							add_work(mysgn * gxsgn, -1, mystr, gxstr);
						}
					}
					if (gxstr) {
						free(gxstr);
					}
					if (mxstr) {
						free(mxstr);
					}
					if (gystr) {
						free(gystr);
					}
					if (mystr) {
						free(mystr);
					}
				}
			}
			if (fmaops && neg_real.size() >= 2) {
				tprint(sw, "L[%i] = -L[%i];\n", index(n, m), index(n, m));
				for (int i = 0; i < neg_real.size(); i++) {
					tprint(sw, "L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), index(n, m));
					flops++;
				}
				tprint(sw, "L[%i] = -L[%i];\n", index(n, m), index(n, m));
				flops += 2;
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint(sw, "L[%i] -= %s * %s;\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint(sw, "L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), index(n, m));
				flops += 2 - fmaops;
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint(sw, "L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint(sw, "L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), index(n, -m));
					flops++;
				}
				tprint(sw, "L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
				flops += 2;
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint(sw, "L[%i] -= %s * %s;\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint(sw, "L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), index(n, -m));
				flops += 2 - fmaops;
			}
		}
		if (n == 0) {
			deindent();
			tprint(sw, "}\n");
			tprint_flush();
		}
		tprint_flush();
	}
	if (P > 1 && periodic) {
		tprint("L[%i] = detail::fma(TCAST(-2) * x, L_st.trace2(), L[%i]);\n", index(1, 1), index(1, 1));
		tprint("L[%i] = detail::fma(TCAST(-2) * y, L_st.trace2(), L[%i]);\n", index(1, -1), index(1, -1));
		tprint("L[%i] = detail::fma(TCAST(-2) * z, L_st.trace2(), L[%i]);\n", index(1, 0), index(1, 0));
		flops += 9 - 3 * fmaops;
		if (!nophi) {
			tprint("if( calcpot ) {\n");
			indent();
			tprint("L[%i] = detail::fma(x * x, L_st.trace2(), L[%i]);\n", index(0, 0), index(0, 0));
			tprint("L[%i] = detail::fma(y * y, L_st.trace2(), L[%i]);\n", index(0, 0), index(0, 0));
			tprint("L[%i] = detail::fma(z * z, L_st.trace2(), L[%i]);\n", index(0, 0), index(0, 0));
			deindent();
			tprint("}\n");
			flops += 9 - 3 * fmaops;
		}
	}

	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return flops;
}

int L2L_rot1(int P) {
	int flops = 0;
	flops_t fps;
	const auto c = tprint_on;
	set_tprint(false);
	flops += regular_harmonic_xz(P);
	set_tprint(c);

	func_header("L2L", P, true, true, true, "", "L", EXP, "x", LIT, "y", LIT, "z", LIT);
	tprint("T rx[%i];\n", P);
	tprint("T ry[%i];\n", P);
	init_real("tmp0");
	init_real("R");
	init_real("Rinv");
	init_real("R2");
	init_real("Rzero");
	init_real("tmp1");
	init_real("cosphi");
	init_real("sinphi");
	tprint("tmp1 = TCAST(1) / L_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");

	tprint("R2 = detail::fma(x, x, y * y);\n");
	flops += 3 - fmaops;
	tprint("Rzero = (R2<TCAST(1e-37));\n");
	tprint("tmp1 = R2 + Rzero;\n");
	tprint("Rinv = detail::rsqrt(tmp1);\n");
	flops += rsqrt_flops;
	flops++;
	tprint("R = TCAST(1) / Rinv;\n");
	flops += divops;
	tprint("cosphi = detail::fma(x, Rinv, Rzero);\n");
	flops += 2 - fmaops;
	tprint("sinphi = -y * Rinv;\n");
	flops += 2;
	flops += z_rot(P, "L", false, false, false, fps);
	const auto yindex = [](int l, int m) {
		return l*(l+1)/2+m;
	};

	tprint("expansion_type<%s, %i> Y_st;\n", type.c_str(), P);
	tprint("T* Y = Y_st.data();\n", type.c_str(), P);
	tprint("detail::regular_harmonic_xz(Y_st, -R, -z, flags);\n");
	flops += 2;
	for (int n = 0; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			std::vector<std::pair<std::string, std::string>> pos_real;
			std::vector<std::pair<std::string, std::string>> neg_real;
			std::vector<std::pair<std::string, std::string>> pos_imag;
			std::vector<std::pair<std::string, std::string>> neg_imag;
			const auto add_work = [&pos_real,&pos_imag,&neg_real,&neg_imag](int sgn, int m, char* mstr, char* gstr) {
				if( sgn == 1) {
					if( m >= 0 ) {
						pos_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					} else {
						pos_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					}
				} else {
					if( m >= 0 ) {
						neg_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					} else {
						neg_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					}
				}
			};
			for (int k = 1; k <= P - n; k++) {
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					char* mxstr = nullptr;
					char* mystr = nullptr;
					char* gxstr = nullptr;
					int mxsgn = 1;
					int mysgn = 1;
					int gxsgn = 1;
					if (m + l > 0) {
						ASPRINTF(&mxstr, "L[%i]", index(n + k, abs(m + l)));
						ASPRINTF(&mystr, "L[%i]", index(n + k, -abs(m + l)));
					} else if (m + l < 0) {
						if (abs(m + l) % 2 == 0) {
							ASPRINTF(&mxstr, "L[%i]", index(n + k, abs(m + l)));
							ASPRINTF(&mystr, "L[%i]", index(n + k, -abs(m + l)));
							mysgn = -1;
						} else {
							ASPRINTF(&mxstr, "L[%i]", index(n + k, abs(m + l)));
							ASPRINTF(&mystr, "L[%i]", index(n + k, -abs(m + l)));
							mxsgn = -1;
						}
					} else {
						ASPRINTF(&mxstr, "L[%i]", index(n + k, 0));
					}
					ASPRINTF(&gxstr, "Y[%i]", yindex(k, abs(l)));
					if (l < 0 && abs(l) % 2 != 0) {
						gxsgn = -1;
					}
					//	L[index(n, m)] += Y(k, l) * M(n + k, m + l);
					const auto csgn = [](int i) {
						return i > 0 ? '+' : '-';
					};
					add_work(mxsgn * gxsgn, +1, mxstr, gxstr);
					if (m > 0) {
						if (mystr) {
							add_work(mysgn * gxsgn, -1, mystr, gxstr);
						}
					}
					if (gxstr) {
						free(gxstr);
					}
					if (mxstr) {
						free(mxstr);
					}
					if (mystr) {
						free(mystr);
					}
				}
			}
			if (fmaops && neg_real.size() >= 2) {
				tprint("L[%i] = -L[%i];\n", index(n, m), index(n, m));
				for (int i = 0; i < neg_real.size(); i++) {
					tprint("L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), index(n, m));
					flops++;
				}
				tprint("L[%i] = -L[%i];\n", index(n, m), index(n, m));
				flops += 2;
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint("L[%i] -= %s * %s;\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint("L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), index(n, m));
				flops += 2 - fmaops;
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint("L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint("L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), index(n, -m));
					flops++;
				}
				tprint("L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
				flops += 2;
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint("L[%i] -= %s * %s;\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint("L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), index(n, -m));
				flops += 2 - fmaops;
			}
		}
	}
	tprint("sinphi = -sinphi;\n");
	flops++;
	if (P > 1 && periodic) {
		tprint("L[%i] = detail::fma(TCAST(-2) * R, L_st.trace2(), L[%i]);\n", index(1, 1), index(1, 1));
		tprint("L[%i] = detail::fma(TCAST(-2) * z, L_st.trace2(), L[%i]);\n", index(1, 0), index(1, 0));
		flops += 6;
		if (!nophi) {
			tprint("if( calcpot ) {\n");
			indent();
			tprint("L[%i] = detail::fma(R2, L_st.trace2(), L[%i]);\n", index(0, 0), index(0, 0));
			tprint("L[%i] = detail::fma(z * z, L_st.trace2(), L[%i]);\n", index(0, 0), index(0, 0));
			deindent();
			tprint("}\n");
			flops += 5;
		}
	}
	flops += z_rot(P, "L", false, false, false, fps);
	flops++;
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return flops;
}

int L2L_rot2(int P) {
	int flops = 0;
	flops_t fps;
	func_header("L2L", P, true, true, true, "", "L", EXP, "x", LIT, "y", LIT, "z", LIT);
	init_reals("A", 2 * P + 1);
	tprint("T rx[%i];\n", P);
	tprint("T ry[%i];\n", P);
	init_real("tmp0");
	init_real("R");
	init_real("Rinv");
	init_real("R2");
	init_real("Rzero");
	init_real("tmp1");
	init_real("r2");
	init_real("rzero");
	init_real("r");
	init_real("rinv");
	init_real("cosphi");
	init_real("cosphi0");
	init_real("sinphi");
	init_real("sinphi0");
	tprint("tmp1 = TCAST(1) / L_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");
	tprint("R2 = detail::fma(x, x, y * y);\n");
	flops += 3 - fmaops;
	tprint("Rzero = T(R2<TCAST(1e-37));\n");
	flops++;
	tprint("tmp1 = R2 + Rzero;\n");
	flops++;
	tprint("Rinv = detail::rsqrt(tmp1);\n");
	flops += rsqrt_flops;
	tprint("r2 = detail::fma(z, z, R2);\n");
	flops += 2 - fmaops;
	tprint("rzero = T(r2<TCAST(1e-37));\n");
	flops += 1;
	tprint("tmp1 = r2 + rzero;\n");
	tprint("rinv = detail::rsqrt(tmp1);\n");
	flops += rsqrt_flops;
	tprint("R = TCAST(1) / Rinv;\n");
	flops += divops;
	tprint("r = TCAST(1) / rinv;\n");
	flops += divops;
	tprint("cosphi = y * Rinv;\n");
	flops++;
	tprint("sinphi = detail::fma(x, Rinv, Rzero);\n");
	flops += 2 - fmaops;
	flops += z_rot(P, "L", false, false, false, fps);
	flops += xz_swap(P, "L", true, false, false, false, fps);
	tprint("cosphi0 = cosphi;\n");
	tprint("sinphi0 = sinphi;\n");
	tprint("cosphi = detail::fma(z, rinv, rzero);\n");
	flops += 2 - fmaops;
	tprint("sinphi = -R * rinv;\n");
	flops += z_rot(P, "L", false, false, false, fps);
	flops += xz_swap(P, "L", true, false, false, false, fps);
	tprint("A[0] = TCAST(1);\n");
	for (int n = 1; n <= P; n++) {
		tprint("A[%i] = -r * A[%i];\n", n, n - 1);
		flops += 2;
	}
	for (int n = 2; n <= P; n++) {
		tprint("A[%i] *= TCAST(%.20e);\n", n, 1.0 / factorial(n));
		flops += 1;
	}
	tprint("if( calcpot ) {\n");
	indent();
	for (int n = 0; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			for (int k = 1; k <= P - n; k++) {
				if (abs(m) > n + k) {
					continue;
				}
				if (-abs(m) < -(k + n)) {
					continue;
				}
				tprint("L[%i] = detail::fma(L[%i], A[%i], L[%i]);\n", index(n, m), index(n + k, m), k, index(n, m));
				flops += 2 - fmaops;
				if (m > 0) {
					tprint("L[%i] = detail::fma(L[%i], A[%i], L[%i]);\n", index(n, -m), index(n + k, -m), k, index(n, -m));
					flops += 2 - fmaops;
				}
			}

		}
		if (n == 0) {
			deindent();
			tprint("}\n");
		}
	}
	if (P > 1 && periodic) {
		tprint("L[%i] = detail::fma(TCAST(-2) * r, L_st.trace2(), L[%i]);\n", index(1, 0), index(1, 0));
		flops += 3 - fmaops;
		if (!nophi) {
			tprint("if( calcpot ) {\n");
			indent();
			tprint("L[%i] = detail::fma(r * r, L_st.trace2(), L[%i]);\n", index(0, 0), index(0, 0));
			deindent();
			tprint("}\n");
			flops += 3 - fmaops;
		}
	}
	flops += xz_swap(P, "L", true, false, false, false, fps);
	tprint("sinphi = -sinphi;\n");
	flops += z_rot(P, "L", false, false, false, fps);
	flops += xz_swap(P, "L", true, false, false, false, fps);
	tprint("cosphi = cosphi0;\n");
	tprint("sinphi = -sinphi0;\n");
	flops += 1;
	flops += z_rot(P, "L", false, false, false, fps);
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return flops;
}

int L2P(int P) {
	int flops = 0;
	const auto c = tprint_on;
	set_tprint(false);
	flops += regular_harmonic(P);
	set_tprint(c);
	const char* fstr[4] = { "f.potential", "f.force[2]", "f.force[0]", "f.force[1]" };
	func_header("L2P", P, true, true, true, "", "f0", FORCE, "L", EXP, "x", LIT, "y", LIT, "z", LIT);
	init_real("rinv");
	init_real("r2inv");
	init_real("tmp1");
	tprint("force_type<%s> f;\n", type.c_str());
	tprint("f.init();\n");
	tprint("tmp1 = TCAST(1) / L_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");
//tprint("expansion_type<T,1> L1;\n");
	tprint("expansion_type<%s, %i> Y_st;\n", type.c_str(), P);
	tprint("T* Y = Y_st.data();\n", type.c_str(), P);
	tprint("detail::regular_harmonic(Y_st, -x, -y, -z, flags);\n");
	flops += 3;
	int sw = 0;
	tprint(sw, "if( calcpot ) {\n");
	indent();
	for (int n = 0; n <= 1; n++) {
		for (int m = 0; m <= n; m++) {
			sw = sw == 0 ? 1 : 0;
			std::vector<std::pair<std::string, std::string>> pos_real;
			std::vector<std::pair<std::string, std::string>> neg_real;
			std::vector<std::pair<std::string, std::string>> pos_imag;
			std::vector<std::pair<std::string, std::string>> neg_imag;
			const auto add_work = [&pos_real,&pos_imag,&neg_real,&neg_imag](int sgn, int m, char* mstr, char* gstr) {
				if( sgn == 1) {
					if( m >= 0 ) {
						pos_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					} else {
						pos_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					}
				} else {
					if( m >= 0 ) {
						neg_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					} else {
						neg_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					}
				}
			};
			bool pfirst = true;
			bool nfirst = true;
			for (int k = 0; k <= P - n; k++) {
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					char* mxstr = nullptr;
					char* mystr = nullptr;
					char* gxstr = nullptr;
					char* gystr = nullptr;
					int mxsgn = 1;
					int mysgn = 1;
					int gxsgn = 1;
					int gysgn = 1;
					if (m + l > 0) {
						ASPRINTF(&mxstr, "L[%i]", index(n + k, abs(m + l)));
						ASPRINTF(&mystr, "L[%i]", index(n + k, -abs(m + l)));
					} else if (m + l < 0) {
						if (abs(m + l) % 2 == 0) {
							ASPRINTF(&mxstr, "L[%i]", index(n + k, abs(m + l)));
							ASPRINTF(&mystr, "L[%i]", index(n + k, -abs(m + l)));
							mysgn = -1;
						} else {
							ASPRINTF(&mxstr, "L[%i]", index(n + k, abs(m + l)));
							ASPRINTF(&mystr, "L[%i]", index(n + k, -abs(m + l)));
							mxsgn = -1;
						}
					} else {
						ASPRINTF(&mxstr, "L[%i]", index(n + k, 0));
					}
					if (l > 0) {
						ASPRINTF(&gxstr, "Y[%i]", index(k, abs(l)));
						ASPRINTF(&gystr, "Y[%i]", index(k, -abs(l)));
						gysgn = -1;
					} else if (l < 0) {
						if (abs(l) % 2 == 0) {
							ASPRINTF(&gxstr, "Y[%i]", index(k, abs(l)));
							ASPRINTF(&gystr, "Y[%i]", index(k, -abs(l)));
						} else {
							ASPRINTF(&gxstr, "Y[%i]", index(k, abs(l)));
							ASPRINTF(&gystr, "Y[%i]", index(k, -abs(l)));
							gxsgn = -1;
							gysgn = -1;
						}
					} else {
						ASPRINTF(&gxstr, "Y[%i]", index(k, 0));
					}
					if (n > 0) {
						mxsgn = -mxsgn;
						mysgn = -mysgn;
					}
					add_work(mxsgn * gxsgn, 1, mxstr, gxstr);
					if (gystr && mystr) {
						add_work(-mysgn * gysgn, 1, mystr, gystr);
					}
					if (m > 0) {
						if (gystr) {
							add_work(mxsgn * gysgn, -1, mxstr, gystr);
						}
						if (mystr) {
							add_work(mysgn * gxsgn, -1, mystr, gxstr);
						}
					}
					if (gxstr) {
						free(gxstr);
					}
					if (mxstr) {
						free(mxstr);
					}
					if (gystr) {
						free(gystr);
					}
					if (mystr) {
						free(mystr);
					}
				}
			}
			if (fmaops && neg_real.size() >= 2) {
				tprint(sw, "%s = -%s;\n", fstr[index(n, m)], fstr[index(n, m)]);
				for (int i = 0; i < neg_real.size(); i++) {
					tprint(sw, "%s = detail::fma(%s, %s, %s);\n", fstr[index(n, m)], neg_real[i].first.c_str(), neg_real[i].second.c_str(), fstr[index(n, m)]);
					flops++;
				}
				tprint(sw, "%s = -%s;\n", fstr[index(n, m)], fstr[index(n, m)]);
				flops += 2;
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint(sw, "%s -= %s * %s;\n", fstr[index(n, m)], neg_real[i].first.c_str(), neg_real[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint(sw, "%s = detail::fma(%s, %s, %s);\n", fstr[index(n, m)], pos_real[i].first.c_str(), pos_real[i].second.c_str(), fstr[index(n, m)]);
				flops += 2 - fmaops;
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint(sw, "%s = -%s;\n", fstr[index(n, -m)], fstr[index(n, -m)]);
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint(sw, "%s = detail::fma(%s, %s, %s);\n", fstr[index(n, -m)], neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), fstr[index(n, -m)]);
					flops++;
				}
				tprint(sw, "%s = -%s;\n", fstr[index(n, -m)], fstr[index(n, -m)]);
				flops += 2;
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint(sw, "%s -= %s * %s;\n", fstr[index(n, -m)], neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint(sw, "%s = detail::fma(%s, %s, %s);\n", fstr[index(n, -m)], pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), fstr[index(n, -m)]);
				flops += 2 - fmaops;
			}
		}
		if (n == 0) {
			deindent();
			tprint(sw, "}\n");
			tprint_flush();
		}
	}
	tprint_flush();
	if (P >= 1 && periodic) {
		tprint("%s = detail::fma(TCAST(2) * x, L_st.trace2(), %s);\n", fstr[index(1, 1)], fstr[index(1, 1)]);
		tprint("%s = detail::fma(TCAST(2) * y, L_st.trace2(), %s);\n", fstr[index(1, -1)], fstr[index(1, -1)]);
		tprint("%s = detail::fma(TCAST(2) * z, L_st.trace2(), %s);\n", fstr[index(1, 0)], fstr[index(1, 0)]);
		flops += 9;
		if (!nophi) {
			tprint("if( calcpot ) {\n");
			indent();
			tprint("%s = detail::fma(x * x, L_st.trace2(), %s);\n", fstr[index(0, 0)], fstr[index(0, 0)]);
			tprint("%s = detail::fma(y * y, L_st.trace2(), %s);\n", fstr[index(0, 0)], fstr[index(0, 0)]);
			tprint("%s = detail::fma(z * z, L_st.trace2(), %s);\n", fstr[index(0, 0)], fstr[index(0, 0)]);
			deindent();
			tprint("}\n");
			flops += 9;
		}
	}
	tprint("rinv = TCAST(1) / L_st.scale();\n");
	tprint("r2inv = rinv * rinv;\n");
	tprint("f0.potential += f.potential * rinv;\n");
	tprint("f0.force[0] += f.force[0] * r2inv;\n");
	tprint("f0.force[1] += f.force[1] * r2inv;\n");
	tprint("f0.force[2] += f.force[2] * r2inv;\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");

	return flops;
}

void scaling(int P) {
}

int P2M(int P) {
	int flops = 0;
	tprint("\n");
	func_header("P2M", P + 1, true, false, true, "", "M", MUL, "m", LIT, "x", LIT, "y", LIT, "z", LIT);
	init_real("ax");
	init_real("ay");
	init_real("r2");
	init_real("tmp1");
	tprint("x = -x;");
	tprint("y = -y;");
	tprint("z = -z;");
	flops += 3;
	tprint("tmp1 = TCAST(1) / M_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");
	tprint("r2 = detail::fma(x, x, detail::fma(y, y, z * z));\n");
	flops += 5 - 2 * fmaops;
	tprint("M[0] = m;\n");
	if (!nophi && periodic) {
		tprint("M_st.trace2() = m * r2;\n");
		flops++;
	}
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
//	M[index(m, m)] = M[index(m - 1, m - 1)] * R / TCAST(2 * m);
			if (m - 1 > 0) {
				tprint("ax = M[%i] * TCAST(%.20e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("ay = M[%i] * TCAST(%.20e);\n", index(m - 1, -(m - 1)), 1.0 / (2.0 * m));
				tprint("M[%i] = x * ax - y * ay;\n", index(m, m));
				tprint("M[%i] = detail::fma(y, ax, x * ay);\n", index(m, -m));
				flops += 6 - fmaops;
			} else {
				tprint("ax = M[%i] * TCAST(%.20e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("M[%i] = x * ax;\n", index(m, m));
				tprint("M[%i] = y * ax;\n", index(m, -m));
				flops += 3;
			}
		}
		if (m + 1 <= P) {
//			M[index(m + 1, m)] = z * M[index(m, m)];
			if (m == 0) {
				tprint("M[%i] = z * M[%i];\n", index(m + 1, m), index(m, m));
				flops += 1;
			} else {
				tprint("M[%i] = z * M[%i];\n", index(m + 1, m), index(m, m));
				tprint("M[%i] = z * M[%i];\n", index(m + 1, -m), index(m, -m));
				flops += 2;
			}
		}
		for (int n = m + 2; n <= P; n++) {
			const double inv = double(1) / (double(n * n) - double(m * m));
//			M[index(n, m)] = inv * (TCAST(2 * n - 1) * z * M[index(n - 1, m)] - r2 * M[index(n - 2, m)]);
			tprint("ax = TCAST(%.20e) * z;\n", inv * double(2 * n - 1));
			tprint("ay = TCAST(%.20e) * r2;\n", -(double) inv);
			tprint("M[%i] = detail::fma(ax, M[%i], ay * M[%i]);\n", index(n, m), index(n - 1, m), index(n - 2, m));
			flops += 5 - fmaops;
			if (m != 0) {
				tprint("M[%i] = detail::fma(ax, M[%i], ay * M[%i]);\n", index(n, -m), index(n - 1, -m), index(n - 2, -m));
				flops += 3 - fmaops;
			}
		}
	}
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return flops + 3;
}

void math_functions() {
	if (fp) {
		fclose(fp);
	}
	fp = fopen("./generated_code/include/spherical_fmm.hpp", "at");
	tprint("\n");

	tprint("#ifndef __CUDACC__\n");
#if defined(VEC_DOUBLE) || defined(VEC_FLOAT)
	tprint("%s\n", vec_header.c_str());
#endif
#ifdef VEC_DOUBLE
	tprint("create_vec_types(double, int64_t, uint64_t, %i);\n", VEC_DOUBLE_SIZE);
#endif
#ifdef VEC_FLOAT
	tprint("create_vec_types(float, int32_t, uint32_t, %i);\n", VEC_FLOAT_SIZE);
#endif
	tprint("\n#endif");
	tprint("\n");
	fclose(fp);
	const char* sout = "*s";
	const char* cout = "*c";
#ifdef FLOAT
	fp = fopen("./generated_code/include/spherical_fmm.hpp", "at");
	tprint("\n");
	tprint("namespace detail {\n");
	tprint("CUDA_EXPORT inline float fma(float a, float b, float c) {\n");
	indent();
	tprint("return fmaf(a, b, c);\n");
	deindent();
	tprint("}\n");
	tprint("CUDA_EXPORT float rsqrt(float);\n");
	tprint("CUDA_EXPORT float sqrt(float);\n");
	tprint("CUDA_EXPORT void sincos(float, float*, float*);\n");
	tprint("CUDA_EXPORT void erfcexp(float, float*, float*);\n");
	tprint("}\n");
	fclose(fp);
	flops_t flops;
	for (int cuda = 0; cuda < 2; cuda++) {
		if (cuda) {
			fp = fopen("./generated_code/src/math/math_float.cu", "at");
		} else {
			fp = fopen("./generated_code/src/math/math_float.cpp", "at");
		}
		tprint("\n");
		tprint("#include \"typecast_float.hpp\"\n");
		tprint("#include <math.h>\n");
		tprint("\n");
		tprint("namespace fmm {\n");
		tprint("namespace detail {\n");
		tprint("\n");
		if (cuda) {
			tprint("__device__\n");
		}
		tprint("T rsqrt(T x) {\n");
		indent();
		tprint("V i;\n");
		tprint("T y;\n");
		tprint("x += TCAST(%.20e);\n", std::numeric_limits<float>::min());
		flops.r++;
		tprint("i = *((V*) &x);\n");
		flops.a++;
		tprint("x *= TCAST(0.5);");
		flops.r++;
		tprint("i >>= VCAST(1);\n");
		flops.i++;
		tprint("i = VCAST(0x5F3759DF) - i;\n");
		flops.i++;
		tprint("y = *((T*) &i);\n");
		tprint("y *= fmaf(x, y * y, TCAST(-1.5));\n");
		flops.f++;
		flops.r += 2;
		tprint("y *= fmaf(x, y * y, TCAST(-1.5));\n");
		flops.f++;
		flops.r += 2;
		tprint("y *= TCAST(1.5) - x * y * y;\n");
		flops.r += 4;
		tprint("return y;\n");
		deindent();
		tprint("}\n");
		tprint("\n");
		flops_map["rsqrt_float"] = flops;
		flops.reset();
		if (cuda) {
			tprint("__device__\n");
		}
		tprint("T sqrt(float x) {\n");
		flops.d++;
		flops += flops_map["rsqrt_float"];
		flops_map["sqrt_float"] = flops;
		flops.reset();
		indent();
		tprint("return TCAST(1) / rsqrt(x);\n");
		deindent();
		tprint("}\n");
		tprint("\n");
		if (cuda) {
			tprint("__device__\n");
		}
		tprint("void sincos(T x, T* s, T* c) {\n");
		indent();
		tprint("V ssgn, j, i, k;\n");
		tprint("T x2;\n");
		tprint("ssgn = VCAST(((*((U*) &x) & UCAST(0x80000000)) >> UCAST(30)) - UCAST(1));\n");
		flops.i += 3;
		tprint("j = V((*((U*) &x) & UCAST(0x7FFFFFFF)));\n");
		flops.i += 1;
		tprint("x = *((T*) &j);\n");
		flops.a++;
		tprint("i = x * TCAST(%.20e);\n", 1.0 / M_PI);
		flops.r += 2;
		tprint("x -= T(i) * TCAST(%.20e);\n", M_PI);
		flops.r += 2;
		tprint("x -= TCAST(%.20e);\n", 0.5 * M_PI);
		flops.r++;
		tprint("x2 = x * x;\n");
		flops.r++;
		{
			constexpr int N = 11;
			tprint("%s = TCAST(%.20e);\n", cout, nonepow(N / 2) / factorial(N));
			tprint("%s = TCAST(%.20e);\n", sout, cout, nonepow((N - 1) / 2) / factorial(N - 1));
			flops.a += 2;
			for (int n = N - 2; n >= 0; n -= 2) {
				tprint("%s = fmaf(%s, x2, TCAST(%.20e));\n", cout, cout, nonepow(n / 2) / factorial(n));
				tprint("%s = fmaf(%s, x2, TCAST(%.20e));\n", sout, sout, nonepow((n - 1) / 2) / factorial(n - 1));
				flops.f += 2;
			}
		}
		tprint("%s *= x;\n", cout);
		flops.r++;
		tprint("k = (((i & VCAST(1)) << VCAST(1)) - VCAST(1));\n");
		flops.i += 3;
		tprint("%s *= TCAST(ssgn * k);\n", sout);
		flops.r += 2;
		flops.i++;
		tprint("%s *= TCAST(k);\n", cout);
		flops.r += 2;
		deindent();
		tprint("}\n");
		tprint("\n");
		flops_map["sincos_float"] = flops;
		flops.reset();

		if (cuda) {
			tprint("__device__\n");
		}
		tprint("T exp(T x) {\n");
		{
			indent();
			constexpr int N = 7;
			tprint("V k;\n");
			tprint("T y, xxx;\n");
			tprint("k =  x / TCAST(0.6931471805599453094172) + TCAST(0.5);\n"); // 1 + divops
			flops.d++;
			flops.r + -2;
			tprint("k -= x < TCAST(0);\n");
			flops.r++;
			tprint("xxx = x - T(k) * TCAST(0.6931471805599453094172);\n");
			flops.r += 3;
			tprint("y = TCAST(%.20e);\n", 1.0 / factorial(N));
			flops.a++;
			for (int i = N - 1; i >= 0; i--) {
				tprint("y = fmaf(y, xxx, TCAST(%.20e));\n", 1.0 / factorial(i)); //17*(2-fmaops);
				flops.f++;
			}
			tprint("k = (k + VCAST(127)) << VCAST(23);\n");
			flops.i += 2;
			tprint("y *= *((T*) (&k));\n"); //1
			flops.r++;
			flops_map["exp_float"] = flops;
			flops.reset();
			tprint("return y;\n");
			deindent();
			tprint("}\n");
			tprint("\n");
		}
		{
			if (cuda) {
				tprint("__device__\n");
			}
			tprint("void erfcexp(T x, T* erfc0, T* exp0) {\n");
			indent();
			tprint("T x2, nx2, q;\n");
			tprint("x2 = x * x;\n");
			flops.r++;
			tprint("nx2 = -x2;\n");
			flops.r++;
			tprint("*exp0 = exp(nx2);\n");
			flops += flops_map["exp_float"];

			constexpr double x0 = 2.75;
			tprint("if (x < TCAST(%.20e)) {\n", x0);
			flops.r++;
			{
				indent();
				constexpr int N = 25;
				tprint("q = TCAST(2) * x * x;\n");
				flops.r += 2;
				tprint("*erfc0 = TCAST(%.20e);\n", 1.0 / dfactorial(2 * N + 1));
				flops.a++;
				for (int n = N - 1; n >= 0; n--) {
					tprint("*erfc0 = fmaf(*erfc0, q, TCAST(%.20e));\n", 1.0 / dfactorial(2 * n + 1));
					flops.f++;
				}
				tprint("*erfc0 *= TCAST(%.20e) * x * *exp0;\n", 2.0 / sqrt(M_PI));
				flops.r += 3;
				tprint("*erfc0 = TCAST(1) - *erfc0;\n");
				flops.r++;
				deindent();
				flops_map["erfcexp"] = flops;
				flops.reset();
			}
			tprint("} else  {\n");
			{
				indent();
				constexpr int N = x0 * x0 + 0.5;
				tprint("q = TCAST(1) / (TCAST(2) * x * x);\n");
				tprint("*erfc0 = TCAST(%.20e);\n", dfactorial(2 * N - 1) * nonepow(N));
				for (int i = N - 1; i >= 1; i--) {
					tprint("*erfc0 = fmaf(*erfc0, q, TCAST(%.20e));\n", dfactorial(2 * i - 1) * nonepow(i));
				}
				tprint("*erfc0 = fmaf(*erfc0, q, TCAST(1));\n");
				tprint("*erfc0 *= *exp0 * TCAST(%.20e) / x;\n", 1.0 / sqrt(M_PI));
				deindent();
			}
			tprint("}\n");
			deindent();
			tprint("}\n");
			tprint("\n");
		}
		tprint("\n");
		tprint("}\n");
		tprint("}\n");
		tprint("\n");
		fclose(fp);
	}
#endif

#ifdef VEC_FLOAT
	fp = fopen("./generated_code/include/spherical_fmm.hpp", "at");
	tprint("\n");
	tprint("#ifndef __CUDACC__\n");
	tprint("namespace detail {\n");
	tprint("inline vec_float fma(vec_float a, vec_float b, vec_float c) {\n");
	indent();
	tprint("return a * b + c;\n");
	deindent();
	tprint("}\n");
	tprint("vec_float rsqrt(vec_float);\n");
	tprint("vec_float sqrt(vec_float);\n");
	tprint("void sincos(vec_float, vec_float*, vec_float*);\n");
	tprint("void erfcexp(vec_float, vec_float*, vec_float*);\n");
	tprint("}\n");
	tprint("#endif\n");
	fclose(fp);
	if (cuda) {
		fp = fopen("./generated_code/src/math/math_vec_float.cu", "at");
	} else {
		fp = fopen("./generated_code/src/math/math_vec_float.cpp", "at");
	}
	tprint("\n");
	tprint("#include \"spherical_fmm.hpp\"\n");
	tprint("#include \"typecast_vec_float.hpp\"\n");
	tprint("\n");
	tprint("namespace fmm {\n");
	tprint("namespace detail {\n");
	tprint("\n");
	tprint("T rsqrt(T x) {\n");
	indent();
	tprint("V i;\n");
	tprint("T y;\n");
	tprint("x += TCAST(%.20e);\n", std::numeric_limits<float>::min());
	tprint("i = *((V*) &x);\n");
	tprint("x *= TCAST(0.5);");
	tprint("i >>= VCAST(1);\n");
	tprint("i = VCAST(0x5F3759DF) - i;\n");
	tprint("y = *((T*) &i);\n");
	tprint("y *= detail::fma(x, y * y, TCAST(-1.5));\n");
	tprint("y *= detail::fma(x, y * y, TCAST(-1.5));\n");
	tprint("y *= TCAST(1.5) - x * y * y;\n");
	flops_map["rsqrt_vec_float"] = flops_map["rsqrt_float"];
	tprint("return y;\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("T sqrt(vec_float x) {\n");
	indent();
	flops_map["sqrt_vec_float"] = flops_map["sqrt_float"];
	tprint("return TCAST(1) / rsqrt(x);\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("void sincos(T x, T* s, T* c) {\n");
	indent();
	tprint("V ssgn, j, i, k;\n");
	tprint("T x2;\n");
	tprint("ssgn = V(((*((U*) &x) & UCAST(0x80000000)) >> UCAST(30)) - UCAST(1));\n");
	tprint("j = V((*((U*) &x) & UCAST(0x7FFFFFFF)));\n");
	tprint("x = *((T*) &j);\n");
	tprint("i = x * TCAST(%.20e);\n", 1.0 / M_PI);
	tprint("x -= T(i) * TCAST(%.20e);\n", M_PI);
	tprint("x -= TCAST(%.20e);\n", 0.5 * M_PI);
	tprint("x2 = x * x;\n");
	{
		constexpr int N = 11;
		tprint("%s = TCAST(%.20e);\n", cout, nonepow(N / 2) / factorial(N));
		tprint("%s = TCAST(%.20e);\n", sout, cout, nonepow((N - 1) / 2) / factorial(N - 1));
		for (int n = N - 2; n >= 0; n -= 2) {
			tprint("%s = detail::fma(%s, x2, TCAST(%.20e));\n", cout, cout, nonepow(n / 2) / factorial(n));
			tprint("%s = detail::fma(%s, x2, TCAST(%.20e));\n", sout, sout, nonepow((n - 1) / 2) / factorial(n - 1));
		}
	}
	tprint("%s *= x;\n", cout);
	tprint("k = (((i & VCAST(1)) << VCAST(1)) - VCAST(1));\n");
	tprint("%s *= T(ssgn * k);\n", sout);
	tprint("%s *= T(k);\n", cout);
	deindent();
	flops_map["sincos_vec_float"] = flops_map["sqrt_float"];
	tprint("}\n");
	tprint("\n");

	tprint("T exp(T x) {\n");
	{
		indent();
		constexpr int N = 7;
		tprint("V k;\n");
		tprint("T y, xxx;\n");
		tprint("k =  x / TCAST(0.6931471805599453094172) + TCAST(0.5);\n"); // 1 + divops
		tprint("k -= x < TCAST(0);\n");
		tprint("xxx = x - T(k) * TCAST(0.6931471805599453094172);\n");
		tprint("y = TCAST(%.20e);\n", 1.0 / factorial(N));
		for (int i = N - 1; i >= 0; i--) {
			tprint("y = detail::fma(y, xxx, TCAST(%.20e));\n", 1.0 / factorial(i)); //17*(2-fmaops);
		}
		tprint("k = (k + VCAST(127)) << VCAST(23);\n");
		tprint("y *= *((T*) (&k));\n"); //1
		flops_map["exp_vec_float"] = flops_map["exp_float"];
		flops_map["exp_vec_float"].i++;

		tprint("return y;\n");
		deindent();
		tprint("}\n");
		tprint("\n");
	}
	{

		tprint("void erfcexp(T x, T* erfc0, T* exp0) {\n");
		indent();
		flops.reset();
		tprint("T x2, nx2, q0, q1, sw0, sw1, tmp1, tmp0;\n");
		tprint("x2 = x * x;\n");
		flops.r++;
		tprint("nx2 = -x2;\n");
		flops.r++;
		tprint("*exp0 = exp(nx2);\n");
		flops += flops_map["exp_vec_float"];

		constexpr double x0 = 2.75;
		tprint("sw0 = (x < TCAST(%.20e));\n", x0);
		flops.r++;
		flops.i++;
		tprint("sw1 = TCAST(1) - sw0;\n", x0);
		flops.r++;
		constexpr int N0 = 25;
		constexpr int N1 = x0 * x0 + 0.5;
		tprint("q0 = TCAST(2) * x * x;\n");
		flops.r += 2;
		tprint("q1 = TCAST(1) / (TCAST(2) * x * x);\n");
		flops.d++;
		flops.r++;
		tprint("tmp0 = TCAST(%.20e);\n", 1.0 / dfactorial(2 * N0 + 1));
		tprint("tmp1 = TCAST(%.20e);\n", dfactorial(2 * N1 - 1) * nonepow(N1));
		flops.a += 2;
		int n1 = N1 - 1;
		int n0 = N0 - 1;
		while (n1 >= 1 || n0 >= 0) {
			if (n0 >= 0) {
				tprint("tmp0 = detail::fma(tmp0, q0, TCAST(%.20e));\n", 1.0 / dfactorial(2 * n0 + 1));
				flops.f++;
			}
			if (n1 >= 1) {
				tprint("tmp1 = detail::fma(tmp1, q1, TCAST(%.20e));\n", dfactorial(2 * n1 - 1) * nonepow(n1));
				flops.f++;
			}
			n1--;
			n0--;
		}
		tprint("tmp0 *= TCAST(%.20e) * x * *exp0;\n", 2.0 / sqrt(M_PI));
		flops.r += 3;
		tprint("tmp1 = detail::fma(tmp1, q1, TCAST(1));\n");
		flops.f++;
		tprint("tmp0 = TCAST(1) - tmp0;\n");
		flops.r++;
		tprint("tmp1 *= *exp0 * TCAST(%.20e) / x;\n", 1.0 / sqrt(M_PI));
		flops.r += 3;
		flops.d++;
		tprint("*erfc0 = sw0 * tmp0 + sw1 * tmp1;\n");
		flops.r += 3;
		deindent();
		tprint("}\n");
		tprint("\n");
	}
	flops_map["erfcexp_vec_float"] = flops;
	tprint("\n");
	tprint("}\n");
	tprint("}\n");
	tprint("\n");
	fclose(fp);
#endif

#ifdef DOUBLE
	fp = fopen("./generated_code/include/spherical_fmm.hpp", "at");
	tprint("\n");
	tprint("namespace detail {\n");
	tprint("CUDA_EXPORT inline double fma(double a, double b, double c) {\n");
	indent();
	tprint("return std::fma(a, b, c);\n");
	deindent();
	tprint("}\n");
	tprint("CUDA_EXPORT double rsqrt(double);\n");
	tprint("CUDA_EXPORT double sqrt(double);\n");
	tprint("CUDA_EXPORT void sincos(double, double*, double*);\n");
	tprint("CUDA_EXPORT void erfcexp(double, double*, double*);\n");
	tprint("}\n");
	fclose(fp);
	constexpr double x0 = 1.0;
	constexpr double x1 = 4.7;
	for (int cuda = 0; cuda < 2; cuda++) {
		if (cuda) {
			fp = fopen("./generated_code/src/math/math_double.cu", "at");
		} else {
			fp = fopen("./generated_code/src/math/math_double.cpp", "at");
		}
		tprint("\n");
		tprint("#include <math.h>\n");
		tprint("#include \"typecast_double.hpp\"\n");
		tprint("\n");
		tprint("namespace fmm {\n");
		tprint("namespace detail {\n");
		tprint("\n");
		if (cuda) {
			tprint("__device__\n");
		}
		tprint("T rsqrt(T x) {\n");
		indent();
		flops.reset();
		tprint("V i;\n");
		tprint("T y;\n");
		tprint("x += TCAST(%.20e);\n", std::numeric_limits<double>::min());
		flops.r++;
		tprint("i = *((V*) &x);\n");
		flops.a++;
		tprint("x *= TCAST(0.5);");
		flops.r++;
		tprint("i >>= VCAST(1);\n");
		flops.i++;
		tprint("i = VCAST(0x5FE6EB50C7B537A9) - i;\n");
		flops.i++;
		tprint("y = *((T*) &i);\n");
		flops.a++;
		tprint("y *= fma(x, y * y, TCAST(-1.5));\n");
		flops.f++;
		flops.r++;
		tprint("y *= fma(x, y * y, TCAST(-1.5));\n");
		flops.f++;
		flops.r++;
		tprint("y *= fma(x, y * y, TCAST(-1.5));\n");
		flops.f++;
		flops.r++;
		tprint("y *= fma(x, y * y, TCAST(-1.5));\n");
		flops.f++;
		flops.r++;
		flops_map["rsqrt_double"] = flops;
		tprint("return y;\n");
		deindent();
		tprint("}\n");
		tprint("\n");
		if (cuda) {
			tprint("__device__\n");
		}
		tprint("T sqrt(T x) {\n");
		indent();
		tprint("return TCAST(1) / rsqrt(x);\n");
		flops.d++;
		flops_map["sqrt_double"] = flops;
		flops.reset();
		deindent();
		tprint("}\n");
		tprint("\n");
		if (cuda) {
			tprint("__device__\n");
		}
		tprint("void sincos(T x, T* s, T* c) {\n");
		indent();
		tprint("V ssgn, j, i, k;\n");
		tprint("T x2;\n");
		tprint("ssgn = VCAST(((*((U*) &x) & UCAST(0x8000000000000000LL)) >> UCAST(62LL)) - UCAST(1LL));\n");
		flops.i += 3;
		tprint("j = V((*((U*) &x) & UCAST(0x7FFFFFFFFFFFFFFFLL)));\n");
		flops.i++;
		tprint("x = *((T*) &j);\n");
		flops.a++;
		tprint("i = x * TCAST(%.20e);\n", 1.0 / M_PI);
		flops.r += 2;
		tprint("x -= T(i) * TCAST(%.20e);\n", M_PI);
		flops.r += 2;
		tprint("x -= TCAST(%.20e);\n", 0.5 * M_PI);
		flops.r++;
		tprint("x2 = x * x;\n");
		flops.r++;
		{
			constexpr int N = 21;
			tprint("%s = TCAST(%.20e);\n", cout, nonepow(N / 2) / factorial(N));
			tprint("%s = TCAST(%.20e);\n", sout, cout, nonepow((N - 1) / 2) / factorial(N - 1));
			flops.a += 2;
			for (int n = N - 2; n >= 0; n -= 2) {
				tprint("%s = fma(%s, x2, TCAST(%.20e));\n", cout, cout, nonepow(n / 2) / factorial(n));
				tprint("%s = fma(%s, x2, TCAST(%.20e));\n", sout, sout, nonepow((n - 1) / 2) / factorial(n - 1));
				flops.f += 2;
			}
		}
		tprint("%s *= x;\n", cout);
		flops.r++;
		tprint("k = (((i & VCAST(1)) << VCAST(1)) - VCAST(1));\n");
		flops.i += 3;
		tprint("%s *= T(ssgn * k);\n", sout);
		flops.i++;
		flops.r += 2;
		tprint("%s *= T(k);\n", cout);
		flops.r += 2;
		deindent();
		tprint("}\n");
		tprint("\n");
		flops_map["sincos_double"] = flops;
		flops.reset();
		if (cuda) {
			tprint("__device__\n");
		}
		tprint("T exp(T x) {\n");
		{
			indent();
			constexpr int N = 18;
			tprint("V k;\n");
			tprint("T xxx, y;\n");
			tprint("k =  x / TCAST(0.6931471805599453094172) + TCAST(0.5);\n"); // 1 + divops
			flops.d++;
			flops.r += 2;
			tprint("k -= x < TCAST(0);\n");
			flops.r++;
			flops.i++;
			tprint("xxx = x - T(k) * TCAST(0.6931471805599453094172);\n");
			flops.r += 2;
			flops.i++;
			tprint("y = TCAST(%.20e);\n", 1.0 / factorial(N));
			flops.a++;
			for (int i = N - 1; i >= 0; i--) {
				tprint("y = fma(y, xxx, TCAST(%.20e));\n", 1.0 / factorial(i)); //17*(2-fmaops);
				flops.f++;
			}
			tprint("k = (k + VCAST(1023)) << VCAST(52);\n");
			flops.i += 2;
			tprint("y *= *((T*) (&k));\n"); //1
			flops.r++;
			flops_map["exp_double"] = flops;
			tprint("return y;\n");
			deindent();
			tprint("}\n");
			tprint("\n");
		}
		if (cuda) {
			tprint("__device__\n");
		}
		tprint("void erfcexp(T x, T* erfc0, T* exp0) {\n");
		indent();
		flops.reset();
		tprint("T x2, nx2, q, x0;\n");
		tprint("x2 = x * x;\n");
		flops.r++;
		tprint("nx2 = -x2;\n");
		flops.r++;
		tprint("*exp0 = exp(nx2);\n");
		flops += flops_map["exp_double"];

		tprint("if (x < TCAST(%.20e)) {\n", x0);
		{
			indent();
			constexpr int N = 17;
			tprint("q = TCAST(2) * x * x;\n");
			tprint("*erfc0 = TCAST(%.20e);\n", 1.0 / dfactorial(2 * N + 1));
			for (int n = N - 1; n >= 0; n--) {
				tprint("*erfc0 = fma(*erfc0, q, TCAST(%.20e));\n", 1.0 / dfactorial(2 * n + 1));
			}
			tprint("*erfc0 *= TCAST(%.20e) * x * *exp0;\n", 2.0 / sqrt(M_PI));
			tprint("*erfc0 = TCAST(1) - *erfc0;\n");
			deindent();
		}
		tprint("} else if (x < TCAST(%.20e)) {\n", x1);
		indent();
		{
			flops.r += 2;
			constexpr int N = 35;
			constexpr double a = (x1 - x0) * 0.5 + x0;
			static double c0[N + 1];
			static bool init = false;
			if (!init) {
				double q[N + 1];
				init = true;
				c0[0] = (a) * exp(a * a) * erfc(a);
				q[0] = exp(a * a) * erfc(a);
				q[1] = -2.0 / sqrt(M_PI) + 2 * a * exp(a * a) * erfc(a);
				for (int n = 2; n <= N; n++) {
					q[n] = 2 * (a * q[n - 1] + q[n - 2]) / n;
				}
				for (int n = 1; n <= N; n++) {
					c0[n] = q[n - 1] + (a) * q[n];
				}
			}
			tprint("x0 = x;\n");
			flops.r++;
			tprint("x -= TCAST(%.20e);\n", a);
			flops.r++;
			tprint("*erfc0 = TCAST(%.20e);\n", c0[N]);
			flops.a++;
			for (int n = N - 1; n >= 0; n--) {
				tprint("*erfc0 = fma(*erfc0, x, TCAST(%.20e));\n", c0[n]);
				flops.f++;
			}
			tprint("*erfc0 *= *exp0 / x0;\n");
			flops.r++;
			flops.d++;
			flops_map["erfcexp_double"] = flops;
			flops.reset();
			deindent();
		}
		tprint("} else  {\n");
		{
			indent();
			constexpr int N = x1 * x1 + 0.5;
			tprint("q = TCAST(1) / (TCAST(2) * x * x);\n");
			tprint("*erfc0 = TCAST(%.20e);\n", dfactorial(2 * N - 1) * nonepow(N));
			for (int i = N - 1; i >= 1; i--) {
				tprint("*erfc0 = fma(*erfc0, q, TCAST(%.20e));\n", dfactorial(2 * i - 1) * nonepow(i));
			}
			tprint("*erfc0 = fma(*erfc0, q, TCAST(1));\n");
			tprint("*erfc0 *= *exp0 * TCAST(%.20e) / x;\n", 1.0 / sqrt(M_PI));
			deindent();
		}
		tprint("}\n");

		deindent();
		tprint("}\n");
		tprint("\n");
		tprint("\n");
		tprint("}\n");
		tprint("}\n");
		tprint("\n");
		fclose(fp);
	}
#endif
#ifdef VEC_DOUBLE
	fp = fopen("./generated_code/include/spherical_fmm.hpp", "at");
	tprint("\n");
	tprint("#ifndef __CUDACC__\n");
	tprint("namespace detail {\n");
	tprint("inline vec_double fma(vec_double a, vec_double b, vec_double c) {\n");
	indent();
	tprint("return a * b + c;\n");
	deindent();
	tprint("}\n");
	tprint("vec_double rsqrt(vec_double);\n");
	tprint("vec_double sqrt(vec_double);\n");
	tprint("void sincos(vec_double, vec_double*, vec_double*);\n");
	tprint("void erfcexp(vec_double, vec_double*, vec_double*);\n");
	tprint("}\n");
	tprint("#endif\n");
	fclose(fp);
	if (cuda) {
		fp = fopen("./generated_code/src/math/math_vec_double.cu", "at");
	} else {
		fp = fopen("./generated_code/src/math/math_vec_double.cpp", "at");
	}
	tprint("\n");
	tprint("#include \"spherical_fmm.hpp\"\n");
	tprint("#include \"typecast_vec_double.hpp\"\n");
	tprint("\n");
	tprint("namespace fmm {\n");
	tprint("namespace detail {\n");
	tprint("\n");
	flops_map["rsqrt_vec_double"] = flops_map["rsqrt_double"];
	flops_map["sqrt_vec_double"] = flops_map["sqrt_double"];
	flops_map["exp_vec_double"] = flops_map["exp_double"];
	flops_map["exp_vec_double"].i++;
	tprint("T rsqrt(T x) {\n");
	indent();
	tprint("V i;\n");
	tprint("T y;\n");
	tprint("x += TCAST(%.20e);\n", std::numeric_limits<double>::min());
	tprint("i = *((V*) &x);\n");
	tprint("x *= TCAST(0.5);");
	tprint("i >>= VCAST(1);\n");
	tprint("i = VCAST(0x5FE6EB50C7B537A9) - i;\n");
	tprint("y = *((T*) &i);\n");
	tprint("y *= detail::fma(x, y * y, TCAST(-1.5));\n");
	tprint("y *= detail::fma(x, y * y, TCAST(-1.5));\n");
	tprint("y *= detail::fma(x, y * y, TCAST(-1.5));\n");
	tprint("y *= detail::fma(x, y * y, TCAST(-1.5));\n");
	tprint("return y;\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("T sqrt(T x) {\n");
	indent();
	tprint("return TCAST(1) / rsqrt(x);\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("void sincos(T x, T* s, T* c) {\n");
	indent();
	tprint("V ssgn, j, i, k;\n");
	tprint("T x2;\n");
	tprint("ssgn = V(((*((U*) &x) & UCAST(0x8000000000000000LL)) >> UCAST(62LL)) - UCAST(1LL));\n");
	tprint("j = V((*((U*) &x) & UCAST(0x7FFFFFFFFFFFFFFFLL)));\n");
	tprint("x = *((T*) &j);\n");
	tprint("i = x * TCAST(%.20e);\n", 1.0 / M_PI);
	tprint("x -= T(i) * TCAST(%.20e);\n", M_PI);
	tprint("x -= TCAST(%.20e);\n", 0.5 * M_PI);
	tprint("x2 = x * x;\n");
	{
		constexpr int N = 21;
		tprint("%s = TCAST(%.20e);\n", cout, nonepow(N / 2) / factorial(N));
		tprint("%s = TCAST(%.20e);\n", sout, cout, nonepow((N - 1) / 2) / factorial(N - 1));
		for (int n = N - 2; n >= 0; n -= 2) {
			tprint("%s = detail::fma(%s, x2, TCAST(%.20e));\n", cout, cout, nonepow(n / 2) / factorial(n));
			tprint("%s = detail::fma(%s, x2, TCAST(%.20e));\n", sout, sout, nonepow((n - 1) / 2) / factorial(n - 1));
		}
	}
	tprint("%s *= x;\n", cout);
	tprint("k = (((i & VCAST(1)) << VCAST(1)) - VCAST(1));\n");
	tprint("%s *= T(ssgn * k);\n", sout);
	tprint("%s *= T(k);\n", cout);
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("T exp(T x) {\n");
	{
		indent();
		constexpr int N = 18;
		tprint("V k;\n");
		tprint("T xxx, y;\n");
		tprint("k =  x / TCAST(0.6931471805599453094172) + TCAST(0.5);\n"); // 1 + divops
		tprint("k -= x < TCAST(0);\n");
		tprint("xxx = x - T(k) * TCAST(0.6931471805599453094172);\n");
		tprint("y = TCAST(%.20e);\n", 1.0 / factorial(N));
		for (int i = N - 1; i >= 0; i--) {
			tprint("y = detail::fma(y, xxx, TCAST(%.20e));\n", 1.0 / factorial(i)); //17*(2-detail::fmaops);
		}
		tprint("k = (k + VCAST(1023)) << VCAST(52);\n");
		tprint("y *= *((T*) (&k));\n"); //1
		tprint("return y;\n");
		deindent();
		tprint("}\n");
		tprint("\n");
	}
	tprint("void erfcexp(T x, T* erfc0, T* exp0) {\n");
	indent();
	flops.reset();
	tprint("T x2, nx2, q0, q2, x0, x1, sw0, sw1, sw2, res0, res1, res2;\n");
	tprint("x2 = x * x;\n");
	flops.r++;
	tprint("nx2 = -x2;\n");
	flops.r++;
	tprint("*exp0 = exp(nx2);\n");
	flops += flops_map["exp_vec_double"];

	tprint("sw0 = T(x < TCAST(%.20e));\n", x0);
	flops.r += 2;
	tprint("sw1 = T(x >= TCAST(%.20e)) * T(x <= TCAST(%.20e));\n", x0, x1);
	flops.r += 5;
	tprint("sw2 = T(x > TCAST(%.20e));\n", x1);
	flops.r += 2;
	{
		constexpr int N0 = 17;
		tprint("x0 = x;\n");
		flops.a++;
		tprint("q0 = TCAST(2) * x0 * x0;\n");
		flops.r += 2;
		tprint("res0 = TCAST(%.20e);\n", 1.0 / dfactorial(2 * N0 + 1));
		flops.a++;
		constexpr int N1 = 35;
		constexpr double a = (x1 - x0) * 0.5 + x0;
		static double c0[N1 + 1];
		static bool init = false;
		double q[N1 + 1];
		init = true;
		c0[0] = (a) * exp(a * a) * erfc(a);
		q[0] = exp(a * a) * erfc(a);
		q[1] = -2.0 / sqrt(M_PI) + 2 * a * exp(a * a) * erfc(a);
		for (int n = 2; n <= N1; n++) {
			q[n] = 2 * (a * q[n - 1] + q[n - 2]) / n;
		}
		for (int n = 1; n <= N1; n++) {
			c0[n] = q[n - 1] + (a) * q[n];
		}
		tprint("x1 = x0 - TCAST(%.20e);\n", a);
		flops.r++;
		tprint("res1 = TCAST(%.20e);\n", c0[N1]);
		flops.a++;
		constexpr int N2 = x1 * x1 + 0.5;
		tprint("x2 = x0;\n");
		flops.a++;
		tprint("q2 = TCAST(1) / (TCAST(2) * x2 * x2);\n");
		flops.d++;
		flops.r += 2;
		tprint("res2 = TCAST(%.20e);\n", dfactorial(2 * N2 - 1) * nonepow(N2));
		flops.a++;
		int n0 = N0 - 1;
		int n1 = N1 - 1;
		int n2 = N2 - 1;
		while (n0 >= 0 || n1 >= 0 || n2 >= 1) {
			if (n0 >= 0) {
				tprint("res0 = detail::fma(res0, q0, TCAST(%.20e));\n", 1.0 / dfactorial(2 * n0 + 1));
				flops.f++;
			}
			if (n1 >= 0) {
				tprint("res1 = detail::fma(res1, x1, TCAST(%.20e));\n", c0[n1]);
				flops.f++;
			}
			if (n2 >= 1) {
				tprint("res2 = detail::fma(res2, q2, TCAST(%.20e));\n", dfactorial(2 * n2 - 1) * nonepow(n2));
				flops.f++;
			}
			n0--;
			n1--;
			n2--;
		}
		tprint("res0 *= TCAST(%.20e) * x0 * *exp0;\n", 2.0 / sqrt(M_PI));
		flops.r += 3;
		tprint("res0 = TCAST(1) - res0;\n");
		flops.r++;
		tprint("res1 *= *exp0 / x0;\n");
		flops.d++;
		flops.r++;
		tprint("res2 = detail::fma(res2, q2, TCAST(1));\n");
		flops.f++;
		tprint("res2 *= *exp0 * TCAST(%.20e) / x2;\n", 1.0 / sqrt(M_PI));
		flops.r += 2;
	}
	tprint("*erfc0 = detail::fma(sw0, res0, detail::fma(sw1, res1, sw2 * res2));\n");
	flops.r += 1;
	flops.f += 2;
	flops_map["erfcexp_vec_double"] = flops;
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("\n");
	tprint("}\n");
	tprint("}\n");
	tprint("\n");
	fclose(fp);
#endif
	fp = nullptr;
}

void typecast_functions() {
	if (fp) {
		fclose(fp);
	}
#ifdef FLOAT
	fp = fopen("./generated_code/include/typecast_float.hpp", "at");
	tprint("#ifndef SPHERICAL_FMM_TYPECAST_FLOAT\n");
	tprint("#define SPHERICAL_FMM_TYPECAST_FLOAT\n");
	tprint("\n");
	tprint("#define TCAST(a) ((float)(a))\n");
	tprint("#define UCAST(a) ((unsigned)(a))\n");
	tprint("#define VCAST(a) ((int)(a))\n");
	tprint("\n");
	tprint("namespace fmm {\n");
	tprint("typedef float T;\n");
	tprint("typedef unsigned U;\n");
	tprint("typedef int V;\n");
	tprint("}\n");
	tprint("\n");
	tprint("#endif\n");
	fclose(fp);
#endif
#ifdef VEC_FLOAT
	fp = fopen("./generated_code/include/typecast_vec_float.hpp", "at");
	tprint("#ifndef SPHERICAL_FMM_TYPECAST_VEC_FLOAT\n");
	tprint("#define SPHERICAL_FMM_TYPECAST_VEC_FLOAT\n");
	tprint("\n");
	tprint("#define TCAST(a) (vec_float(float(a)))\n");
	tprint("#define UCAST(a) (vec_uint32_t(unsigned(a)))\n");
	tprint("#define VCAST(a) (vec_int32_t(int(a)))\n");
	tprint("\n");
	tprint("namespace fmm {\n");
	tprint("typedef vec_float T;\n");
	tprint("typedef vec_uint32_t U;\n");
	tprint("typedef vec_int32_t V;\n");
	tprint("}\n");
	tprint("\n");
	tprint("#endif\n");
	fclose(fp);
#endif
#ifdef DOUBLE
	fp = fopen("./generated_code/include/typecast_double.hpp", "at");
	tprint("#ifndef SPHERICAL_FMM_TYPECAST_DOUBLE\n");
	tprint("#define SPHERICAL_FMM_TYPECAST_DOUBLE\n");
	tprint("\n");
	tprint("#define TCAST(a) ((double)(a))\n");
	tprint("#define UCAST(a) ((unsigned long long)(a))\n");
	tprint("#define VCAST(a) ((long long)(a))\n");
	tprint("\n");
	tprint("namespace fmm {\n");
	tprint("typedef double T;\n");
	tprint("typedef unsigned long long U;\n");
	tprint("typedef long long V;\n");
	tprint("}\n");
	tprint("\n");
	tprint("#endif\n");
	fclose(fp);
#endif
#ifdef VEC_DOUBLE
	fp = fopen("./generated_code/include/typecast_vec_double.hpp", "at");
	tprint("#ifndef SPHERICAL_FMM_TYPECAST_VEC_DOUBLE\n");
	tprint("#define SPHERICAL_FMM_TYPECAST_VEC_DOUBLE\n");
	tprint("\n");
	tprint("#define TCAST(a) (vec_double(double(a)))\n");
	tprint("#define UCAST(a) (vec_uint64_t(uint64_t(a)))\n");
	tprint("#define VCAST(a) (vec_int64_t(int64_t(a)))\n");
	tprint("\n");
	tprint("namespace fmm {\n");
	tprint("typedef vec_double T;\n");
	tprint("typedef vec_uint64_t U;\n");
	tprint("typedef vec_int64_t V;\n");
	tprint("}\n");
	tprint("\n");
	tprint("#endif\n");
	fclose(fp);
#endif
	fp = nullptr;
}

int main() {
	SYSTEM("[ -e ./generated_code ] && rm -rf  ./generated_code\n");
	SYSTEM("mkdir generated_code\n");
	SYSTEM("mkdir ./generated_code/include\n");
	SYSTEM("mkdir ./generated_code/src\n");
	SYSTEM("mkdir ./generated_code/src/math\n");
	tprint("\n");
	set_file("./generated_code/include/spherical_fmm.hpp");
	tprint("#pragma once\n");
	tprint("\n");
	tprint("#ifdef __CUDA_ARCH__\n");
	tprint("#define CUDA_EXPORT __device__\n");
	tprint("#else\n");
	tprint("#define CUDA_EXPORT\n");
	tprint("#endif\n");
	tprint("\n");
	tprint("#include <math.h>\n");
	tprint("#include <cmath>\n");
	tprint("#include <cstdint>\n");
	tprint("\n");
	tprint("#define FMM_NOCALC_POT (0x1)\n");
	tprint("\n");
	tprint("namespace fmm {\n");
	tprint("\n");
	tprint("template<class T>\n");
	tprint("struct force_type {\n");
	indent();
	tprint("T potential;\n");
	tprint("T force[3];\n");
	tprint("CUDA_EXPORT inline void init() {\n");
	indent();
	tprint("potential = force[0] = force[1] = force[2] = T(0);\n");
	deindent();
	tprint("}\n");
	deindent();
	tprint("};\n");
	tprint("\n");
	tprint("template<class T, int P>\n");
	tprint("class expansion_type {\n");
	indent();
	tprint("T o[(P+1)*(P+1)];\n");
	tprint("T t;\n");
	tprint("T r;\n");
	deindent();
	tprint("public:\n");
	indent();
	tprint("CUDA_EXPORT expansion_type(T=T(1));\n");
	tprint("CUDA_EXPORT expansion_type(const expansion_type&);\n");
	tprint("CUDA_EXPORT expansion_type& operator=(const expansion_type&);\n");
	tprint("CUDA_EXPORT expansion_type& operator+=(expansion_type);\n");
	tprint("CUDA_EXPORT void init(T r0 = T(1));\n");
	tprint("CUDA_EXPORT void rescale(T);\n");
	tprint("CUDA_EXPORT T* data();\n");
	tprint("CUDA_EXPORT const T* data() const;\n");
	tprint("CUDA_EXPORT T scale() const;\n");
	tprint("CUDA_EXPORT T& trace2();\n");
	tprint("CUDA_EXPORT T trace2() const;\n");
	deindent();
	tprint("};\n");
	tprint("\n");

	tprint("template<class T, int P>\n");
	tprint("class multipole_type {\n");
	indent();
	tprint("T o[P*P];\n");
	tprint("T t;\n");
	tprint("T r;\n");
	deindent();
	tprint("public:\n");
	indent();
	tprint("CUDA_EXPORT multipole_type(T=T(1));\n");
	tprint("CUDA_EXPORT multipole_type(const multipole_type&);\n");
	tprint("CUDA_EXPORT multipole_type& operator=(const multipole_type&);\n");
	tprint("CUDA_EXPORT multipole_type& operator+=(multipole_type);\n");
	tprint("CUDA_EXPORT void init(T r0 = T(1));\n");
	tprint("CUDA_EXPORT void rescale(T);\n");
	tprint("CUDA_EXPORT T* data();\n");
	tprint("CUDA_EXPORT const T* data() const;\n");
	tprint("CUDA_EXPORT T scale() const;\n");
	tprint("CUDA_EXPORT T& trace2();\n");
	tprint("CUDA_EXPORT T trace2() const;\n");
	deindent();
	tprint("};\n");
	tprint("\n");

	for (int P = pmin - 1; P <= pmax; P++) {

		tprint("\n");
		tprint("template<class T>\n");
		tprint("class expansion_type<T,%i> {\n", P);
		indent();
		tprint("T o[%i];\n", exp_sz(P));
		tprint("T t;\n");
		tprint("T r;\n");
		deindent();
		tprint("public:\n");
		indent();
		tprint("CUDA_EXPORT expansion_type(T r0 = T(1)) {\n");
		indent();
		tprint("r = r0;\n");
		deindent();
		tprint("}\n");
		tprint("CUDA_EXPORT void init(T r0 = T(1)) {\n");
		indent();
		tprint("for( int n = 0; n < %i; n++ ) {\n", exp_sz(P));
		indent();
		tprint("o[n] = T(0);\n");
		deindent();
		tprint("}\n");
		tprint("t = T(0);\n");
		tprint("r = r0;\n");
		deindent();
		tprint("}\n");
		tprint("CUDA_EXPORT expansion_type(const expansion_type& other) {\n");
		indent();
		tprint("*this = other;\n");
		deindent();
		tprint("}\n");

		tprint("CUDA_EXPORT expansion_type& operator=(const expansion_type& other) {\n");
		indent();
		tprint("for( int n = 0; n < %i; n++ ) {\n", exp_sz(P));
		indent();
		tprint("o[n] = other.o[n];\n");
		deindent();
		tprint("}\n");
		tprint("t = other.t;\n");
		tprint("r = other.r;\n");
		tprint("return *this;\n");
		deindent();
		tprint("}\n");

		tprint("CUDA_EXPORT expansion_type& operator+=(expansion_type other) {\n");
		indent();
		tprint("other.rescale(r);\n");
		tprint("for( int n = 0; n < %i; n++ ) {\n", exp_sz(P));
		indent();
		tprint("o[n] += other.o[n];\n");
		deindent();
		tprint("}\n");
		tprint("t += other.t;\n");
		tprint("r = other.r;\n");
		tprint("return *this;\n");
		deindent();
		tprint("}\n");

		tprint("CUDA_EXPORT void rescale(T r0) {\n");
		indent();
		tprint("const T a = r0 / r;\n");
		tprint("T b = a;\n");
		tprint("r = r0;\n");
		for (int n = 0; n <= P; n++) {
			for (int m = -n; m <= n; m++) {
				tprint("o[%i] *= b;\n", index(n, m));
			}
			if (n == 2) {
				tprint("t *= b;\n", exp_sz(P));
			}
			if (n != P) {
				tprint("b *= a;\n");
			}
		}
		deindent();
		tprint("}\n");
		tprint("CUDA_EXPORT T* data() {\n");
		indent();
		tprint("return o;\n");
		deindent();
		tprint("}\n");
		tprint("CUDA_EXPORT const T* data() const {\n");
		indent();
		tprint("return o;\n");
		deindent();
		tprint("}\n");
		tprint("CUDA_EXPORT T scale() const {\n");
		indent();
		tprint("return r;\n");
		deindent();
		tprint("}\n");
		tprint("CUDA_EXPORT T& trace2() {\n");
		indent();
		tprint("return t;\n");
		deindent();
		tprint("}\n");
		tprint("CUDA_EXPORT T trace2() const {\n");
		indent();
		tprint("return t;\n");
		deindent();
		tprint("}\n");
		deindent();
		tprint("};\n");
		tprint("\n");
		if (P > pmin - 1) {
			tprint("\n");
			tprint("template<class T>\n");
			tprint("class multipole_type<T,%i> {\n", P);
			indent();
			tprint("T o[%i];\n", mul_sz(P));
			tprint("T t;\n");
			tprint("T r;\n");
			deindent();
			tprint("public:\n");
			indent();
			tprint("CUDA_EXPORT multipole_type(T r0 = T(1)) {\n");
			indent();
			tprint("r = r0;\n");
			deindent();
			tprint("}\n");
			tprint("CUDA_EXPORT void init(T r0 = T(1)) {\n");
			indent();
			tprint("for( int n = 0; n < %i; n++ ) {\n", mul_sz(P));
			indent();
			tprint("o[n] = T(0);\n");
			deindent();
			tprint("}\n");
			tprint("t = T(0);\n");
			tprint("r = r0;\n");
			deindent();
			tprint("}\n");
			tprint("CUDA_EXPORT multipole_type(const multipole_type& other) {\n");
			indent();
			tprint("*this = other;\n");
			deindent();
			tprint("}\n");

			tprint("CUDA_EXPORT multipole_type& operator=(const multipole_type& other) {\n");
			indent();
			tprint("for( int n = 0; n < %i; n++ ) {\n", mul_sz(P));
			indent();
			tprint("o[n] = other.o[n];\n");
			deindent();
			tprint("}\n");
			tprint("t = other.t;\n");
			tprint("r = other.r;\n");
			tprint("return *this;\n");
			deindent();
			tprint("}\n");

			tprint("CUDA_EXPORT multipole_type& operator+=(multipole_type other) {\n");
			indent();
			tprint("if( r != other.r ) {\n");
			indent();
			tprint("other.rescale(r);\n");
			deindent();
			tprint("}\n");
			tprint("for( int n = 0; n < %i; n++ ) {\n", mul_sz(P));
			indent();
			tprint("o[n] += other.o[n];\n");
			deindent();
			tprint("}\n");
			tprint("t += other.t;\n");
			tprint("r = other.r;\n");
			tprint("return *this;\n");
			deindent();
			tprint("}\n");

			tprint("CUDA_EXPORT void rescale(T r0) {\n");
			indent();
			tprint("const T a = r / r0;\n");
			tprint("T b = a;\n");
			tprint("r = r0;\n");
			for (int n = 1; n < P; n++) {
				for (int m = -n; m <= n; m++) {
					tprint("o[%i] *= b;\n", index(n, m));
				}
				if (n == 2) {
					tprint("t *= b;\n");
				}
				if (n != P - 1) {
					tprint("b *= a;\n");
				}
			}
			deindent();
			tprint("}\n");
			tprint("CUDA_EXPORT T* data() {\n");
			indent();
			tprint("return o;\n");
			deindent();
			tprint("}\n");
			tprint("CUDA_EXPORT const T* data() const {\n");
			indent();
			tprint("return o;\n");
			deindent();
			tprint("}\n");
			tprint("CUDA_EXPORT T scale() const {\n");
			indent();
			tprint("return r;\n");
			deindent();
			tprint("}\n");
			tprint("CUDA_EXPORT T& trace2() {\n");
			indent();
			tprint("return t;\n");
			deindent();
			tprint("}\n");
			tprint("CUDA_EXPORT T trace2() const {\n");
			indent();
			tprint("return t;\n");
			deindent();
			tprint("}\n");
			deindent();
			tprint("};\n");
			tprint("\n");
		}
	}
	tprint("\n");
	math_functions();
	set_file("./generated_code/include/spherical_fmm.hpp");
	tprint("\n");
	typecast_functions();

	static int rsqrt_float_flops = 15 - 3 * fmaops;
	static int rsqrt_double_flops = 15 - 3 * fmaops;
	static int sqrt_float_flops = 15 - 3 * fmaops + divops;
	static int sqrt_double_flops = 15 - 3 * fmaops + divops;
	static int sincos_float_flops = 31 - fmaops * 10;
	static int sincos_double_flops = 41 - fmaops * 20;
	static int exp_double_flops = 4 + divops + 17 * (2 - fmaops);
	static int exp_float_flops = 4 + divops + 7 * (2 - fmaops);
	static int erfcexp_float_flops = exp_float_flops + 56 - 24 * fmaops;
	static int erfcexp_double_flops = exp_double_flops + 72 - 35 * fmaops + divops;

	const int rsqrt_flops_array[] = { rsqrt_float_flops, rsqrt_double_flops, rsqrt_float_flops, rsqrt_double_flops, rsqrt_float_flops, rsqrt_double_flops };
	const int sqrt_flops_array[] = { sqrt_float_flops, sqrt_double_flops, sqrt_float_flops, sqrt_float_flops, sqrt_double_flops, sqrt_float_flops,
			sqrt_double_flops };
	const int sincos_flops_array[] =
			{ sincos_float_flops, sincos_double_flops, sincos_float_flops, sincos_double_flops, sincos_float_flops, sincos_double_flops };
	const int erfcexp_flops_array[] = { erfcexp_float_flops, erfcexp_double_flops, erfcexp_float_flops, erfcexp_double_flops, erfcexp_float_flops,
			erfcexp_double_flops };
	int ntypenames = 0;
	std::vector<std::string> rtypenames;
	std::vector<std::string> sitypenames;
	std::vector<std::string> uitypenames;
	std::vector<int> ucuda;
#ifdef FLOAT
	rtypenames.push_back("float");
	sitypenames.push_back("int32_t");
	uitypenames.push_back("uint32_t");
	ucuda.push_back(true);
	ntypenames++;
#endif
#ifdef DOUBLE
	rtypenames.push_back("double");
	sitypenames.push_back("int64_t");
	uitypenames.push_back("uint64_t");
	ucuda.push_back(true);
	ntypenames++;
#endif
#ifdef FLOAT
	rtypenames.push_back("float");
	sitypenames.push_back("int32_t");
	uitypenames.push_back("uint32_t");
	ucuda.push_back(false);
	ntypenames++;
#endif
#ifdef DOUBLE
	rtypenames.push_back("double");
	sitypenames.push_back("int64_t");
	uitypenames.push_back("uint64_t");
	ucuda.push_back(false);
	ntypenames++;
#endif
#ifdef VEC_FLOAT
	rtypenames.push_back("vec_float");
	sitypenames.push_back("vec_int32_t");
	uitypenames.push_back("vec_uint32_t");
	ucuda.push_back(false);
	ntypenames++;
#endif

#ifdef VEC_DOUBLE
	rtypenames.push_back("vec_double");
	sitypenames.push_back("vec_int64_t");
	uitypenames.push_back("vec_uint64_t");
	ucuda.push_back(false);
	ntypenames++;
#endif

	for (int ti = 0; ti < ntypenames; ti++) {
		fprintf( stderr, "%s cuda:%i\n", rtypenames[ti].c_str(), ucuda[ti]);
		cuda = ucuda[ti];
		prefix = ucuda[ti] ? "CUDA_EXPORT" : "";
		type = rtypenames[ti];
		sitype = sitypenames[ti];
		uitype = uitypenames[ti];
		rsqrt_flops = rsqrt_flops_array[ti];
		sqrt_flops = sqrt_flops_array[ti];
		sincos_flops = sincos_flops_array[ti];
		erfcexp_flops = erfcexp_flops_array[ti];
		std::vector<double> alphas(pmax + 1);
		if (is_vec(type)) {
			set_file("./generated_code/include/spherical_fmm.hpp");
			tprint("#ifndef __CUDACC__\n");
		}
		flops_t fps;
		for (int b = 0; b < 1; b++) {
			nophi = b != 0;
			std::vector<int> pc_flops(pmax + 1);
			std::vector<int> cp_flops(pmax + 1);
			std::vector<int> cc_flops(pmax + 1);
			std::vector<int> m2m_flops(pmax + 1);
			std::vector<int> l2l_flops(pmax + 1);
			std::vector<int> l2p_flops(pmax + 1);
			std::vector<int> p2m_flops(pmax + 1);
			std::vector<int> pc_rot(pmax + 1);
			std::vector<int> cc_rot(pmax + 1);
			std::vector<int> m2m_rot(pmax + 1);
			std::vector<int> l2l_rot(pmax + 1);

			set_tprint(false);
			fprintf(stderr, "%2s %5s %5s %2s %5s %5s %2s %5s %5s %5s %5s %2s %5s %5s %2s %5s %5s %5s %5s ", "p", "M2L", "eff", "-r", "M2P", "eff", "-r", "P2L",
					"eff", "M2M", "eff", "-r", "L2L", "eff", "-r", "P2M", "eff", "L2P", "eff");
			if (b == 0) {
				fprintf(stderr, " %8s %8s %8s %8s\n", "CC_ewald", "green", "m2l", "alpha");
			} else {
				fprintf(stderr, " \n");

			}
			int eflopsg;
			for (int P = pmin; P <= pmax; P++) {
				auto r0 = M2L_norot(P, P, fps);
				auto r1 = M2L_rot1(P, P);
				auto r2 = M2L_rot2(P, P);
				if (r0 <= r1 && r0 <= r2) {
					cc_flops[P] = r0;
					cc_rot[P] = 0;
				} else if (r1 <= r0 && r1 <= r2) {
					cc_flops[P] = r1;
					cc_rot[P] = 1;
				} else {
					cc_flops[P] = r2;
					cc_rot[P] = 2;
				}
				r0 = M2L_norot(P, 1, fps);
				r1 = M2L_rot1(P, 1);
				r2 = M2L_rot2(P, 1);
				if (r0 <= r1 && r0 <= r2) {
					pc_flops[P] = r0;
					pc_rot[P] = 0;
				} else if (r1 <= r0 && r1 <= r2) {
					pc_flops[P] = r1;
					pc_rot[P] = 1;
				} else {
					pc_flops[P] = r2;
					pc_rot[P] = 2;
				}
				cp_flops[P] = p2l(P, fps);
				r0 = M2M_norot(P - 1);
				r1 = M2M_rot1(P - 1);
				r2 = M2M_rot2(P - 1);
				if (r0 <= r1 && r0 <= r2) {
					m2m_flops[P] = r0;
					m2m_rot[P] = 0;
				} else if (r1 <= r0 && r1 <= r2) {
					m2m_flops[P] = r1;
					m2m_rot[P] = 1;
				} else {
					m2m_flops[P] = r2;
					m2m_rot[P] = 2;
				}
				r0 = L2L_norot(P);
				r1 = L2L_rot1(P);
				r2 = L2L_rot2(P);
				if (r0 <= r1 && r0 <= r2) {
					l2l_flops[P] = r0;
					l2l_rot[P] = 0;
				} else if (r1 <= r0 && r1 <= r2) {
					l2l_flops[P] = r1;
					l2l_rot[P] = 1;
				} else {
					l2l_flops[P] = r2;
					l2l_rot[P] = 2;
				}
				l2p_flops[P] = L2P(P);
				p2m_flops[P] = P2M(P - 1);
				int eflopsm = m2lg(P, P, fps);
				if (b == 0) {
					double best_alpha;
					int best_ops = 1000000000;
					double alpha = is_float(type) ? 2.4 : 2.25;
					alphas[P] = alpha;
					int ops = greens_ewald(P, alpha, fps);
					eflopsg = greens_ewald(P, alpha, fps);
					/*					for (double alpha = 2.2; alpha <= 2.45; alpha += 0.05) {

					 //					printf( "%i %e %i\n", P, alpha, ops);
					 if (ops < best_ops) {
					 best_ops = ops;
					 best_alpha = alpha;
					 }
					 }
					 alphas[P] = best_alpha;
					 */
				}
				fprintf(stderr, "%2i %5i %5.2f %2i %5i %5.2f %2i %5i %5.2f %5i %5.2f %2i %5i %5.2f %2i %5i %5.2f %5i %5.2f ", P, cc_flops[P],
						cc_flops[P] / pow(P + 1, 3), cc_rot[P], pc_flops[P], pc_flops[P] / pow(P + 1, 2), pc_rot[P], cp_flops[P], cp_flops[P] / pow(P + 1, 2),
						m2m_flops[P], m2m_flops[P] / pow(P + 1, 3), m2m_rot[P], l2l_flops[P], l2l_flops[P] / pow(P + 1, 3), l2l_rot[P], p2m_flops[P],
						p2m_flops[P] / pow(P + 1, 2), l2p_flops[P], l2p_flops[P] / pow(P + 1, 2));
				if (b == 0) {
					fprintf(stderr, " %8i %8i %8i %f \n", eflopsg + eflopsm, eflopsg, eflopsm, alphas[P]);
				} else {
					fprintf(stderr, "\n");
				}
			}
			set_tprint(true);
			regular_harmonic(pmin - 1);
			regular_harmonic_xz(pmin - 1);
			for (int P = 3; P <= pmax; P++) {
				if (b == 0)
					greens(P, fps);
				if (b == 0)
					greens_xz(P, fps);
				switch (cc_rot[P]) {
				case 0:
					M2L_norot(P, P, fps);
					break;
				case 1:
					M2L_rot1(P, P);
					break;
				case 2:
					M2L_rot2(P, P);
					break;
				};
				if (P > 1) {
					switch (pc_rot[P]) {
					case 0:
						M2L_norot(P, 1, fps);
						break;
					case 1:
						M2L_rot1(P, 1);
						break;
					case 2:
						M2L_rot2(P, 1);
						break;
					};
				}
				p2l(P, fps);
				regular_harmonic(P);
				if (cuda) {
//					regular_harmonic_cuda(P);
				}
				regular_harmonic_xz(P);
				switch (m2m_rot[P]) {
				case 0:
					M2M_norot(P - 1);
					break;
				case 1:
					M2M_rot1(P - 1);
					break;
				case 2:
					M2M_rot2(P - 1);
					break;
				};
				switch (l2l_rot[P]) {
				case 0:
					L2L_norot(P);
					break;
				case 1:
					L2L_rot1(P);
					break;
				case 2:
					L2L_rot2(P);
					break;
				};
				L2P(P);
				P2M(P - 1);
				greens_ewald(P, alphas[P], fps);
				m2lg(P, P, fps);
				M2L_ewald(P, fps);
				scaling(P);
			}
		}
		if (is_vec(type)) {
			set_file("./generated_code/include/spherical_fmm.hpp");
			tprint("#endif\n");
		}

	}
//	printf("./generated_code/include/spherical_fmm.h");
	fflush(stdout);
	set_file("./generated_code/include/spherical_fmm.hpp");
	tprint("namespace detail {\n");
	tprint("%s", detail_header.c_str());
	tprint("#ifndef __CUDACC__\n");
	tprint("%s", detail_header_vec.c_str());
	tprint("#endif\n");
	tprint("}\n");
	tprint("}\n");
	tprint("\n");

	return 0;
}
