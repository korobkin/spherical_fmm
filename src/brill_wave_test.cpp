#include <stdio.h>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <fenv.h>

#include "spherical_fmm.hpp"

using real = double;

struct Point {
	real x;
	real y;
	real z;
	real volume;
	real V;
};

struct IterationInfo {
	int iteration;
	real scale;
	real lInfDelta;
	real l2Delta;
	real massEstimate;
};

template<class T>
T sqr(T const a) {
	return a * a;
}

template<class T>
T sqr(T const a, T const b, T const c) {
	return a * a + b * b + c * c;
}

real gaussianQ(real const x, real const y, real const z, real const amplitude, real const sigma) {
	auto const r2 = sqr(x, y, z);
	return amplitude * std::exp(-r2 / sqr(sigma));
}

real brillPotential3D(real const x, real const y, real const z, real const amplitude, real const sigma) {
	auto const r2 = sqr(x, y, z);
	auto const sigma2 = sqr(sigma);
	auto const sigma4 = sqr(sigma2);
	auto const e = std::exp(-r2 / sigma2);
	auto const laplacianQ = amplitude * e * (4.0 * r2 / sigma4 - 6.0 / sigma2);
	return -0.25 * laplacianQ;
}

std::vector<Point> makePointCloud(int const nPerDim, real const boxHalfWidth, real const amplitude, real const sigma, real const vCut) {
	auto points = std::vector<Point>{};
	auto const dx = 2.0 * boxHalfWidth / real(nPerDim);
	auto const dv = dx * dx * dx;
	points.reserve(nPerDim * nPerDim * nPerDim);
	for (int iz = 0; iz < nPerDim; iz++) {
		for (int iy = 0; iy < nPerDim; iy++) {
			for (int ix = 0; ix < nPerDim; ix++) {
				auto const x = -boxHalfWidth + (real(ix) + 0.5) * dx;
				auto const y = -boxHalfWidth + (real(iy) + 0.5) * dx;
				auto const z = -boxHalfWidth + (real(iz) + 0.5) * dx;
				auto const V = brillPotential3D(x, y, z, amplitude, sigma);
				if (std::abs(V) >= vCut) {
					points.push_back(Point{x, y, z, dv, V});
				}
			}
		}
	}
	return points;
}

real initialGuessTildeChi(real const x, real const y, real const z, real const M) {
	auto const r2 = sqr(x, y, z);
	return std::pow(1.0 + 4.0 * r2 / sqr(M), -0.5);
}

real computeW(std::vector<Point> const& points) {
	auto W = 0.0;
	for (auto const& p: points) {
		W += p.volume * p.V;
	}
	return W;
}

real normalizeChi(std::vector<Point> const& points, std::vector<real>& chi, real const M, real const W) {
	auto denom = 0.0;
	for (size_t i = 0; i < points.size(); i++) {
		denom += points[i].volume * points[i].V * chi[i];
	}
	if (std::abs(denom) < 1.0e-300) {
		return 1.0;
	}
	auto const C = -(4.0 * M_PI * M + W) / denom;
	for (auto& x: chi) {
		x *= C;
	}
	return C;
}

template<int P>
real potentialAtTargetUsingP2L(std::vector<Point> const& points, std::vector<real> const& masses, Point const& target, bool const excludeSelf, int const targetIndex) {
	using namespace fmm;
	auto L = expansion_type<real, P>{};
	auto f = force_type<real>{};
	auto const flags = 0;
	L.init(1.0e-30);
	f.init();
	for (size_t j = 0; j < points.size(); j++) {
		if (excludeSelf && int(j) == targetIndex) {
			continue;
		}
		auto const dx = points[j].x - target.x;
		auto const dy = points[j].y - target.y;
		auto const dz = points[j].z - target.z;
		P2L(L, masses[j], dx, dy, dz, flags);
	}
	L2P(f, L, 0.0, 0.0, 0.0, flags);
	return f.potential;
}

template<int P>
std::vector<real> applyFixedPointMap(std::vector<Point> const& points, std::vector<real> const& chi, bool const excludeSelf) {
	auto masses = std::vector<real>(points.size());
	for (size_t i = 0; i < points.size(); i++) {
		masses[i] = points[i].volume * points[i].V * (1.0 + chi[i]) / (4.0 * M_PI);
	}
	auto tildeChiNext = std::vector<real>(points.size());
	for (size_t i = 0; i < points.size(); i++) {
		auto const phi = potentialAtTargetUsingP2L<P>(points, masses, points[i], excludeSelf, int(i));
		tildeChiNext[i] = -phi;
	}
	return tildeChiNext;
}

IterationInfo computeIterationInfo(std::vector<Point> const& points, std::vector<real> const& oldChi, std::vector<real> const& newChi, int const iteration, real const scale) {
	auto info = IterationInfo{};
	info.iteration = iteration;
	info.scale = scale;
	info.lInfDelta = 0.0;
	info.l2Delta = 0.0;
	info.massEstimate = 0.0;
	for (size_t i = 0; i < points.size(); i++) {
		auto const d = newChi[i] - oldChi[i];
		info.lInfDelta = std::max(info.lInfDelta, std::abs(d));
		info.l2Delta += d * d;
		info.massEstimate += -(points[i].volume * points[i].V * (1.0 + newChi[i])) / (4.0 * M_PI);
	}
	info.l2Delta = std::sqrt(info.l2Delta / std::max<size_t>(1, points.size()));
	return info;
}

template<int P>
std::vector<real> solveBrillByFixedPoint(std::vector<Point> const& points, real const M, int const maxIter, real const tol, real const relax, bool const excludeSelf) {
	auto const W = computeW(points);
	auto chi = std::vector<real>(points.size());
	for (size_t i = 0; i < points.size(); i++) {
		chi[i] = initialGuessTildeChi(points[i].x, points[i].y, points[i].z, M);
	}
	normalizeChi(points, chi, M, W);
	printf("# points=%zu W=% .16e\n", points.size(), W);
	for (int iter = 0; iter < maxIter; iter++) {
		auto tildeChiNext = applyFixedPointMap<P>(points, chi, excludeSelf);
		auto const C = normalizeChi(points, tildeChiNext, M, W);
		auto chiNext = chi;
		for (size_t i = 0; i < chi.size(); i++) {
			chiNext[i] = (1.0 - relax) * chi[i] + relax * tildeChiNext[i];
		}
		auto const info = computeIterationInfo(points, chi, chiNext, iter, C);
		printf("iter=%3d  C=% .6e  linf=% .6e  l2=% .6e  M=% .6e\n",
			info.iteration,
			info.scale,
			info.lInfDelta,
			info.l2Delta,
			info.massEstimate);
		chi = std::move(chiNext);
		if (info.lInfDelta < tol) {
			break;
		}
	}
	for (auto& x: chi) {
		x += 1.0;
	}
	return chi;
}

template<int P>
void sampleAlongXAxis(std::vector<Point> const& points, std::vector<real> const& psi, real const amplitude, real const sigma) {
	printf("\n# x  psi(x,0,0)  V(x,0,0)\n");
	for (size_t i = 0; i < points.size(); i++) {
		if (std::abs(points[i].y) < 1.0e-12 && std::abs(points[i].z) < 1.0e-12) {
			printf("% .16e % .16e % .16e\n", points[i].x, psi[i], brillPotential3D(points[i].x, 0.0, 0.0, amplitude, sigma));
		}
	}
}

int main() {
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_INVALID);
	feenableexcept(FE_OVERFLOW);

	constexpr int P = 4;
	auto const nPerDim = 16;
	auto const boxHalfWidth = 4.0;
	auto const amplitude = 0.20;
	auto const sigma = 1.0;
	auto const vCut = 1.0e-8;
	auto const M = 0.25;
	auto const maxIter = 50;
	auto const tol = 1.0e-8;
	auto const relax = 0.7;
	auto const excludeSelf = true;

	auto const points = makePointCloud(nPerDim, boxHalfWidth, amplitude, sigma, vCut);
	auto const psi = solveBrillByFixedPoint<P>(points, M, maxIter, tol, relax, excludeSelf);
	sampleAlongXAxis<P>(points, psi, amplitude, sigma);
	return 0;
}