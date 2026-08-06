#include "sfmm.hpp"
#include <stdio.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fenv.h>
#include <future>
#include <limits>
#include <thread>
#include <vector>

#define DEBUG

#define NDIM 3
#define BUCKET_SIZE (64)
#define MIN_CLOUD 4
#define LEFT 0
#define RIGHT 1
#define NCHILD 2
#define MIN_THREAD 1024
#define TEST_SIZE 32
// #define FLAGS (sfmmWithRandomOptimization | sfmmProfilingOn)

using rtype = double;

double rand1() {
	return (rand() + 0.5) / RAND_MAX;
}

void prefetch(void *ptr, int cnt) {
	for (int i = 0; i < cnt; i += 32) {
		__builtin_prefetch((char *)ptr + i);
	}
}

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
class tree {

	template <class W>
	using multipole_type = sfmm::multipole<W, ORDER>;
	template <class W>
	using expansion_type = sfmm::expansion<W, ORDER>;
	template <class W>
	using force_type = sfmm::force_type<W>;

	multipole_type<T> multipole;
	std::vector<tree> children;
	std::pair<int, int> part_range;
	sfmm::vec3<T> begin;
	sfmm::vec3<T> end;
	sfmm::vec3<FT> center;
	tree *parent;
	T radius;
	T scale;

	static const T theta_max;
	static const T hsoft;
	static const int Ngrid;
	static std::atomic<long long> p2p;
	static std::atomic<long long> m2p;
	static std::atomic<long long> p2l;
	static std::atomic<long long> m2l;
	static std::atomic<long long> p2p_ewald;
	static std::atomic<long long> m2p_ewald;
	static std::atomic<long long> p2l_ewald;
	static std::atomic<long long> m2l_ewald;
	static std::atomic<long long> node_count;
	static std::atomic<int> threads_avail;

	static std::vector<sfmm::vec3<FT>> parts;
	static std::vector<force_type<T>> forces;
	static std::vector<T> masses;
	static std::vector<T> chi;
	static std::vector<T> sourceV;
	static std::vector<T> volume;
	static std::vector<tree> forest;

	struct check_type {
		tree *ptr;
		bool opened;
	};

public:
	tree() {
	}

	static int partition_parts(std::pair<int, int> part_range, int xdim, T xmid) {
		int lo = part_range.first;
		int hi = part_range.second;
		while (lo < hi) {
			if (parts[lo][xdim].to_double() >= xmid) {
				while (lo != hi) {
					hi--;
					if (parts[hi][xdim].to_double() < xmid) {
						std::swap(parts[lo], parts[hi]);
						std::swap(forces[lo], forces[hi]);
						std::swap(masses[lo], masses[hi]);
						std::swap(chi[lo], chi[hi]);
						std::swap(sourceV[lo], sourceV[hi]);
						std::swap(volume[lo], volume[hi]);
						break;
					}
				}
			}
			lo++;
		}
		return hi;
	}

	static void sort_grid(sfmm::vec3<int> cell_begin = {0, 0, 0}, sfmm::vec3<int> cell_end = {Ngrid, Ngrid, Ngrid},
						  std::pair<int, int> part_range = std::make_pair(0, parts.size())) {
#ifdef DEBUG
		bool flag = false;
		for (int i = part_range.first; i < part_range.second; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				T begin = (T)cell_begin[dim] / Ngrid;
				T end = (T)cell_end[dim] / Ngrid;
				if (parts[i][dim].to_double() < begin || parts[i][dim].to_double() > end) {
					flag = true;
				}
			}
		}
		for (int i = part_range.first; i < part_range.second; i++) {
			if (flag) {
				for (int dim = 0; dim < NDIM; dim++) {
					T begin = (T)cell_begin[dim] / Ngrid;
					T end = (T)cell_end[dim] / Ngrid;
					printf("particle out of range! %e %e %e \n", begin, parts[i][dim].to_double(), end);
				}
			}
		}
		if (flag) {
			abort();
		}
#endif
		sfmm::vec3<int> cell_end_left;
		sfmm::vec3<int> cell_begin_right;
		std::pair<int, int> range_left;
		std::pair<int, int> range_right;
		int cmid;
		int xdim;
		T xmid;
		int largest = 0;
		for (int dim = 0; dim < SFMM_NDIM; dim++) {
			const int span = cell_end[dim] - cell_begin[dim];
			if (span > largest) {
				largest = span;
				xdim = dim;
			}
		}
		if (largest == 1) {
			// printf( "%i %i %i : %i %i\n", cell_begin[0], cell_begin[1], cell_begin[2], part_range.first, part_range.second);
			tree root;
			root.part_range = part_range;
			for (int dim = 0; dim < SFMM_NDIM; dim++) {
				root.begin[dim] = (T)cell_begin[dim] / Ngrid;
				root.end[dim] = (T)cell_end[dim] / Ngrid;
			}
			forest.push_back(root);
		} else {
			cmid = (cell_end[xdim] + cell_begin[xdim]) / 2;
			xmid = (T)cmid / Ngrid;
			int pmid = partition_parts(part_range, xdim, xmid);
			range_left = range_right = part_range;
			range_left.second = range_right.first = pmid;
			cell_end_left = cell_end;
			cell_begin_right = cell_begin;
			cell_end_left[xdim] = cell_begin_right[xdim] = cmid;
			sort_grid(cell_begin, cell_end_left, range_left);
			sort_grid(cell_begin_right, cell_end, range_right);
		}
	}

	static size_t form_trees() {
		size_t flops = 0;
		std::vector<std::future<size_t>> futs;
		for (auto &tr : forest) {
			auto *ptr = &tr;
			futs.push_back(std::async([ptr]() {
				return ptr->form_tree();
			}));
		}
		for (auto &f : futs) {
			flops += f.get();
		}
		return flops;
	}

	size_t form_tree(tree *par = nullptr, int depth = 0) {
		size_t flops = 0;
		const int xdim = depth % SFMM_NDIM;
		parent = par;
		const T xmid = T(0.5) * (begin[xdim] + end[xdim]);
		const int nparts = part_range.second - part_range.first;
		scale = end[0] - begin[0];
		multipole.init(scale);
#ifdef DEBUG
		for (int i = part_range.first; i < part_range.second; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				if (parts[i][dim].to_double() < begin[dim] || parts[i][dim].to_double() > end[dim]) {
					printf("particle out of range! %e %e %e %i\n", begin[dim], parts[i][dim].to_double(), end[dim], depth);
				}
			}
		}
#endif
		//		center = (begin + end) * 0.5;
		sfmm::vec3<double> center0;
		if (nparts > BUCKET_SIZE) {
			sfmm::vec3<T> dx;
			multipole_type<T> M;
			children.resize(NCHILD);
			int imid = partition_parts(part_range, xdim, xmid);
			auto &left = children[LEFT];
			auto &right = children[RIGHT];
			left.part_range = right.part_range = part_range;
			left.part_range.second = right.part_range.first = imid;
			right.begin = left.begin = begin;
			left.end = right.end = end;
			left.end[xdim] = right.begin[xdim] = 0.5 * (begin[xdim] + end[xdim]);
			if (threads_avail-- >= 0) {
				auto fut = std::async([this, depth]() {
					size_t flops = children[LEFT].form_tree(this, depth + 1);
					threads_avail++;
					return flops;
				});
				flops += children[RIGHT].form_tree(this, depth + 1);
				flops += fut.get();
			} else {
				threads_avail++;
				flops += children[LEFT].form_tree(this, depth + 1);
				flops += children[RIGHT].form_tree(this, depth + 1);
			}
			const auto &cleft = children[LEFT];
			const auto &cright = children[RIGHT];
			center0 = (begin + end) * T(0.5);
			radius = 0.0;
			for (int ci = 0; ci < NCHILD; ci++) {
				for (int dim = 0; dim < NDIM; dim++) {
					dx[dim] = children[ci].center[dim].to_double() - center0[dim];
				}
				M = children[ci].multipole;
				radius = std::max(radius, sfmm::reduce_max(abs(dx) + children[ci].radius));
				flops += sfmm::M2M(M, dx, FLAGS);
				multipole += sfmm::reduce_sum(M);
			}
			center = center0;
		} else {
			if (nparts) {
				node_count++;
				center0 = T(0);
				for (int i = part_range.first; i < part_range.second; i++) {
					for (int dim = 0; dim < SFMM_NDIM; dim++) {
						center0[dim] += parts[i][dim].to_double();
					}
				}
				center0 /= nparts;
				radius = 0.0;
				multipole_type<V> M;
				M.init(scale);
				for (int i = part_range.first; i < part_range.second; i += V::size()) {
					sfmm::vec3<V> dx;
					const int cnt = std::min((int)V::size(), part_range.second - i);
					for (int j = 0; j < cnt; j++) {
						const auto &part = parts[i + j];
						for (int dim = 0; dim < SFMM_NDIM; dim++) {
							sfmm::load(dx[dim], T(part[dim].to_double() - center0[dim]), j);
						}
					}
					sfmm::apply_padding(dx, cnt);
					radius = std::max(radius, sfmm::reduce_max(abs(dx)));
					V pmass;
					for (int j = 0; j < cnt; j++) {
						sfmm::load(pmass, masses[i + j], j);
					}
					sfmm::apply_padding(pmass, cnt);
					flops += V::size() * sfmm::P2M(M, sfmm::create_mask<V>(cnt) * pmass, dx);
				}
				multipole = sfmm::reduce_sum(M);
				center = center0;
				radius = std::max(hsoft, radius);
			} else {
				center0 = (begin + end) * 0.5;
				center = center0;
				radius = hsoft;
			}
		}
		return flops;
	}

	static size_t compute_gravity() {
		std::vector<check_type> checklist;
		size_t flops = 0;
		for (auto &root : forest) {
			check_type entry;
			entry.ptr = &root;
			entry.opened = false;
			checklist.push_back(entry);
		}
		std::vector<std::future<size_t>> futs;
		for (auto &tr : forest) {
			auto *ptr = &tr;
			futs.push_back(std::async([ptr, checklist]() {
				size_t flops = 0;
				expansion_type<T> expansion;
				expansion.init();
				return flops + ptr->compute_cell_gravity(expansion, checklist, checklist);
			}));
		}
		for (auto &f : futs) {
			flops += f.get();
		}
		return flops;
	}

	static size_t compute_gravity_bruteforce() {
		const size_t nparts = parts.size();
		if (nparts == 0) {
			return 0;
		}
		const size_t nthreads = std::min<size_t>(std::max(1u, 2u * std::thread::hardware_concurrency()), nparts);
		std::vector<std::future<size_t>> futs;
		futs.reserve(nthreads);
		for (size_t thread = 0; thread < nthreads; thread++) {
			const size_t sink_begin = thread * nparts / nthreads;
			const size_t sink_end = (thread + 1) * nparts / nthreads;
			futs.push_back(std::async(std::launch::async, [sink_begin, sink_end, nparts]() {
				size_t flops = 0;
				for (size_t i = sink_begin; i < sink_end; i++) {
					force_type<V> F;
					F.init();
					sfmm::vec3<FV> xsnk;
					for (int dim = 0; dim < SFMM_NDIM; dim++) {
						xsnk[dim] = FV(parts[i][dim]);
					}
					for (size_t j = 0; j < nparts; j += V::size()) {
						const int cnt = std::min<size_t>(V::size(), nparts - j);
						sfmm::vec3<FV> xsrc;
						V pmass;
						for (int lane = 0; lane < cnt; lane++) {
							sfmm::load(xsrc, parts[j + lane], lane);
							sfmm::load(pmass, masses[j + lane], lane);
						}
						sfmm::apply_padding(xsrc, cnt);
						sfmm::apply_padding(pmass, cnt);
						flops += cnt * sfmm::P2P(F, sfmm::create_mask<V>(cnt) * pmass, xsrc, xsnk, FLAGS);
					}
					forces[i] = sfmm::reduce_sum(F);
					force_type<T> self;
					self.init();
					flops += sfmm::P2P(self, masses[i], sfmm::vec3<T>({T(0), T(0), T(0)}), FLAGS);
					forces[i].potential -= self.potential;
				}
				return flops;
			}));
		}
		size_t flops = 0;
		for (auto &f : futs) {
			flops += f.get();
		}
		return flops;
	}

	size_t compute_cell_gravity(expansion_type<T> expansion, std::vector<check_type> dchecklist, std::vector<check_type> echecklist) {
		size_t flops = 0;
		expansion_type<V> L(scale);
		multipole_type<V> M(scale);
		force_type<V> F;
		sfmm::vec3<FV> xsnk;
		sfmm::vec3<FV> xsrc;
		if (parent) {
			expansion.rescale(scale);
			flops += sfmm::L2L(expansion, parent->center, center, FLAGS);
		}
		if (!children.size()) {
			for (int i = part_range.first; i < part_range.second; i++) {
				force_type<T> F;
				F.init();
				flops += sfmm::P2P(F, masses[i], sfmm::vec3<T>({0, 0, 0}), FLAGS);
				forces[i].potential = -F.potential;
				for (int dim = 0; dim < NDIM; dim++) {
					forces[i].force[dim] = T(0);
				}
			}
		}
		static thread_local std::vector<tree *> M2Llist;
		static thread_local std::vector<tree *> P2Llist;
		static thread_local std::vector<tree *> M2Plist;
		static thread_local std::vector<tree *> P2Plist;
		static thread_local std::vector<check_type> nextlist;
		for (int ewald = 0; ewald <= 0; ewald++) {
			const T c0(1.0 / theta_max);
			auto &checklist = ewald ? echecklist : dchecklist;
			M2Llist.resize(0);
			P2Llist.resize(0);
			M2Plist.resize(0);
			P2Plist.resize(0);
			do {
				for (int i = 0; i < checklist.size(); i += V::size()) {
					sfmm::vec3<V> dx;
					V rsum;
					V rm2l;
					V rp2l;
					V rm2p;
					const int end = std::min(V::size(), checklist.size() - i);
					for (int j = 0; j < end; j++) {
						const auto &check = checklist[i + j];
						sfmm::load(dx, sfmm::distance(center, check.ptr->center), j);
						sfmm::load(rsum, radius + check.ptr->radius, j);
						sfmm::load(rm2l, c0 * radius + c0 * check.ptr->radius, j);
						sfmm::load(rm2p, radius + c0 * check.ptr->radius, j);
						sfmm::load(rp2l, c0 * radius + check.ptr->radius, j);
					}
					sfmm::apply_padding(dx, end);
					sfmm::apply_padding(rsum, end);
					sfmm::apply_padding(rm2l, end);
					sfmm::apply_padding(rp2l, end);
					sfmm::apply_padding(rm2p, end);
					V Cm2l, Cl2p, Cm2p;
					auto D = abs(dx);
					if (ewald) {
						D = fmax(D, V(0.5) - rsum);
					}
					Cm2l = D - rm2l;
					Cl2p = D - rp2l;
					Cm2p = D - rm2p;
					for (int j = 0; j < end; j++) {
						const auto check = checklist[j + i];
						bool used = false;
						if (Cm2l[j] > 0.0) {
							M2Llist.push_back(check.ptr);
							used = true;
						} else if (!children.size() && check.ptr->children.size()) {
							if (Cm2p[j] > T(0)) {
								M2Plist.push_back(check.ptr);
								used = true;
							}
						} else if (children.size() && !check.ptr->children.size()) {
							if (Cl2p[j] > T(0)) {
								P2Llist.push_back(check.ptr);
								used = true;
							}
						} else if (!children.size() && !check.ptr->children.size()) {
							if (Cl2p[j] > T(0) && Cl2p[j] > Cm2p[j]) {
								P2Llist.push_back(check.ptr);
							} else if (Cm2p[j] > T(0) && Cl2p[j] < Cm2p[j]) {
								M2Plist.push_back(check.ptr);
							} else {
								P2Plist.push_back(check.ptr);
							}
							used = true;
						}
						if (!used) {
							if (check.ptr->children.size()) {
								for (int ci = 0; ci < NCHILD; ci++) {
									check_type chk;
									chk.ptr = &(check.ptr->children[ci]);
									nextlist.push_back(chk);
								}
							} else {
								nextlist.push_back(check);
							}
						}
					}
				}
				checklist = std::move(nextlist);
				nextlist.resize(0);
			} while (!children.size() && checklist.size());
			xsnk = center;
			int inters;
			inters = 0;
			L.init(scale);
			for (int i = 0; i < M2Llist.size(); i += V::size()) {
				const int cnt = std::min(V::size(), M2Llist.size() - i);
				inters += cnt;
				for (int j = 0; j < cnt; j++) {
					const auto &src = M2Llist[i + j];
					sfmm::load(xsrc, src->center, j);
					sfmm::load(M, src->multipole, j);
				}
				sfmm::apply_padding(xsrc, cnt);
				apply_mask(M, cnt);
				if (ewald) {
					flops += cnt * sfmm::M2L_ewald(L, M, xsrc, xsnk, FLAGS);
				} else {
					flops += cnt * sfmm::M2L(L, M, xsrc, xsnk, FLAGS);
				}
			}
			expansion += reduce_sum(L);
			(ewald ? m2l_ewald : m2l) += inters;
			inters = 0;
			L.init(scale);
			for (int i = 0; i < P2Llist.size(); i++) {
				for (int j = P2Llist[i]->part_range.first; j < P2Llist[i]->part_range.second; j += V::size()) {
					const int cnt = std::min((int)V::size(), P2Llist[i]->part_range.second - j);
					inters += cnt;
					for (int k = 0; k < cnt; k++) {
						sfmm::load(xsrc, parts[j + k], k);
					}
					sfmm::apply_padding(xsrc, cnt);
					if (ewald) {
						V pmass;
						for (int k = 0; k < cnt; k++) {
							sfmm::load(pmass, masses[j + k], k);
						}
						sfmm::apply_padding(pmass, cnt);
						flops += cnt * sfmm::P2L_ewald(L, sfmm::create_mask<V>(cnt) * pmass, xsrc, xsnk, FLAGS);
					} else {
						V pmass;
						for (int k = 0; k < cnt; k++) {
							sfmm::load(pmass, masses[j + k], k);
						}
						sfmm::apply_padding(pmass, cnt);
						flops += cnt * sfmm::P2L(L, sfmm::create_mask<V>(cnt) * pmass, xsrc, xsnk, FLAGS);
					}
				}
			}
			expansion += (reduce_sum(L));
			(ewald ? p2l_ewald : p2l) += inters;
			if (!children.size()) {
				inters = 0;
				for (int i = 0; i < M2Plist.size(); i += V::size()) {
					const int cnt = std::min(V::size(), M2Plist.size() - i);
					for (int j = 0; j < cnt; j++) {
						const auto &src = M2Plist[i + j];
						sfmm::load(xsrc, src->center, j);
						sfmm::load(M, src->multipole, j);
					}
					sfmm::apply_padding(xsrc, cnt);
					apply_mask(M, cnt);
					for (int k = part_range.first; k < part_range.second; k++) {
						auto &part = parts[k];
						for (int dim = 0; dim < SFMM_NDIM; dim++) {
							xsnk[dim] = FV(part[dim]);
						}
						inters += cnt;
						F.init();
						if (ewald) {
							flops += cnt * sfmm::M2P_ewald(F, M, xsrc, xsnk, FLAGS);
						} else {
							flops += cnt * sfmm::M2P(F, M, xsrc, xsnk, FLAGS);
						}
						forces[k] += reduce_sum(F);
					}
				}
				(ewald ? m2p_ewald : m2p) += inters;
				inters = 0;
				const auto nparts = part_range.second - part_range.first;
				static thread_local std::array<force_type<V>, BUCKET_SIZE> Forces;
				static thread_local std::array<sfmm::vec3<FV>, BUCKET_SIZE> xsnk;
				for (int l = part_range.first; l < part_range.second; l++) {
					for (int dim = 0; dim < SFMM_NDIM; dim++) {
						xsnk[l - part_range.first][dim] = FV(parts[l][dim]);
					}
				}
				for (int i = 0; i < nparts; i++) {
					Forces[i].init();
				}
				for (int i = 0; i < P2Plist.size(); i++) {
					for (int j = P2Plist[i]->part_range.first; j < P2Plist[i]->part_range.second; j += V::size()) {
						const int cnt = std::min((int)V::size(), P2Plist[i]->part_range.second - j);
						inters += cnt;
						for (int k = 0; k < cnt; k++) {
							sfmm::load(xsrc, parts[j + k], k);
						}
						sfmm::apply_padding(xsrc, cnt);
						if (ewald) {
							for (int l = 0; l < nparts; l++) {
								V pmass;
								for (int k = 0; k < cnt; k++) {
									sfmm::load(pmass, masses[j + k], k);
								}
								sfmm::apply_padding(pmass, cnt);
								flops += cnt * sfmm::P2P_ewald(Forces[l], sfmm::create_mask<V>(cnt) * pmass, xsrc, xsnk[l], FLAGS);
							}
						} else {
							for (int l = 0; l < nparts; l++) {
								V pmass;
								for (int k = 0; k < cnt; k++) {
									sfmm::load(pmass, masses[j + k], k);
								}
								sfmm::apply_padding(pmass, cnt);
								flops += cnt * sfmm::P2P(Forces[l], sfmm::create_mask<V>(cnt) * pmass, xsrc, xsnk[l], FLAGS);
							}
						}
					}
				}
				for (int i = 0; i < nparts; i++) {
					forces[part_range.first + i] += reduce_sum(Forces[i]);
				}
				(ewald ? p2p_ewald : p2p) += inters;
				inters = 0;
			}
		}
		if (children.size()) {
			if (dchecklist.size() || echecklist.size()) {
				if (threads_avail-- >= 0) {
					auto fut = std::async(
						[this, expansion](std::vector<check_type> dchecklist, std::vector<check_type> echecklist) {
							size_t flops = children[LEFT].compute_cell_gravity(expansion, std::move(dchecklist), std::move(echecklist));
							threads_avail++;
							return flops;
						},
						dchecklist, echecklist);
					flops += children[RIGHT].compute_cell_gravity(expansion, std::move(dchecklist), std::move(echecklist));
					flops += fut.get();
				} else {
					threads_avail++;
					flops += children[LEFT].compute_cell_gravity(expansion, dchecklist, echecklist);
					flops += children[RIGHT].compute_cell_gravity(expansion, std::move(dchecklist), std::move(echecklist));
				}
			}
		} else {
			sfmm::load(L, (expansion));
			xsnk = center;
			for (int i = part_range.first; i < part_range.second; i += V::size()) {
				force_type<V> F;
				F.init();
				const int cnt = std::min((int)V::size(), part_range.second - i);
				for (int j = 0; j < cnt; j++) {
					sfmm::load(xsrc, parts[j + i], j);
				}
				sfmm::apply_padding(xsrc, cnt);
				flops += cnt * sfmm::L2P(F, L, xsnk, xsrc, FLAGS);
				for (int j = 0; j < cnt; j++) {
					accumulate(forces[i + j], F, j);
				}
			}
		}
		return flops;
	}

	static std::pair<T, T> compare_analytic(T sample_odds) {
		double err = 0.0;
		double norm = 0.0;
		double perr = 0.0;
		double pnorm = 0.0;
		force_type<T> f;
		f.init();
		P2P(f, masses.empty() ? T(0) : masses[0], {0, 0, 0});
		double pot0 = f.potential;
		for (int i = 0; i < parts.size(); i++) {
			if (rand1() > sample_odds) {
				continue;
			}
			const auto &snk_part = parts[i];
			force_type<T> fa;
			fa.init();
			fa.potential = -pot0;
			const int nthreads = 2 * std::thread::hardware_concurrency();
			std::vector<std::future<void>> futs;
			std::mutex mutex;
			for (int proc = 0; proc < nthreads; proc++) {
				const int b = (size_t)proc * parts.size() / nthreads;
				const int e = (size_t)(proc + 1) * parts.size() / nthreads;
				futs.push_back(std::async([i, b, e, &fa, &mutex, snk_part]() {
					sfmm::vec3<FV> xsrc, xsnk;
					force_type<V> fe;
					force_type<V> f;
					xsnk = sfmm::vec3<FV>(parts[i]);
					for (int j = b; j < e; j += sfmm::simd_size<FV>()) {
						int cnt = std::min((int)sfmm::simd_size<FV>(), e - j);
						for (int k = j; k < j + cnt; k++) {
							sfmm::load(xsrc, parts[k], k - j);
						}
						f.init();
						sfmm::apply_padding(xsrc, cnt);
						V pmass;
						for (int k = 0; k < cnt; k++) {
							sfmm::load(pmass, masses[j + k], k);
						}
						sfmm::apply_padding(pmass, cnt);
						P2P(f, sfmm::create_mask<V>(cnt) * pmass, xsrc, xsnk);
						P2P_ewald(f, sfmm::create_mask<V>(cnt) * pmass, xsrc, xsnk);
						std::lock_guard<std::mutex> lock(mutex);
						fa += reduce_sum(f);
					}
				}));
			}
			for (auto &f : futs) {
				f.get();
			}
			double famag = 0.0;
			double fnmag = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				famag += sfmm::sqr(fa.force[0]) + sfmm::sqr(fa.force[1]) + sfmm::sqr(fa.force[2]);
				fnmag += sfmm::sqr(forces[i].force[0]) + sfmm::sqr(forces[i].force[1]) + sfmm::sqr(forces[i].force[2]);
			}
			famag = sqrt(famag);
			fnmag = sqrt(fnmag);
			printf("%e %e %e\n", (famag - fnmag) / famag, fa.potential, forces[i].potential);
			std::lock_guard<std::mutex> lock(mutex);
			perr += sfmm::sqr(forces[i].potential - fa.potential);
			pnorm += sfmm::sqr(fa.potential);
			norm += sfmm::sqr(famag);
			err += sfmm::sqr(famag - fnmag);
		}
		err = sqrt(err / norm);
		perr = sqrt(perr / pnorm);
		return std::make_pair(perr, err);
	}

	static T brill_source(T x, T y, T z) {
		const T A = T(4.0);
		const T sigma = T(0.15);
		const T x0 = x - T(0.5);
		const T y0 = y - T(0.5);
		const T z0 = z - T(0.5);
		const T rho2 = x0 * x0 + y0 * y0;
		const T s2 = sigma * sigma;
		const T r2 = rho2 + z0 * z0;
		const T e = std::exp(-r2 / s2);

		// q = A rho^2 exp(-(rho^2 + z^2) / sigma^2)
		// V = -1/4 (q_{,rho rho} + q_{,zz})
		const T lap_q = A * e * (T(2) - (T(10) * rho2 + T(2) * z0 * z0) / s2 + T(4) * rho2 * r2 / (s2 * s2));
		return -T(0.25) * lap_q;
	}

	static void initialize() {
		forest.resize(0);
		parts.resize(0);
		forces.resize(0);
		masses.resize(0);
		chi.resize(0);
		sourceV.resize(0);
		volume.resize(0);
		const int N = TEST_SIZE;
		const T dx = T(1) / N;
		const T dV = dx * dx * dx;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				for (int k = 0; k < N; k++) {
					const T x = (T(i) + T(0.5)) * dx;
					const T y = (T(j) + T(0.5)) * dx;
					const T z = (T(k) + T(0.5)) * dx;
					const T Vq = brill_source(x, y, z);
					if (std::abs(Vq) > T(1.0e-10)) {
						parts.push_back(sfmm::vec3<FT>({FT(x), FT(y), FT(z)}));
						sourceV.push_back(Vq);
						volume.push_back(dV);
					}
				}
			}
		}
		forces.resize(parts.size());
		masses.resize(parts.size(), T(0));
		chi.resize(parts.size(), T(0));
		for (int i = 0; i < parts.size(); i++) {
			const T x = parts[i][0].to_double() - T(0.5);
			const T y = parts[i][1].to_double() - T(0.5);
			const T z = parts[i][2].to_double() - T(0.5);
			const T r2 = x * x + y * y + z * z;
			chi[i] = T(1) / std::sqrt(T(1) + T(4) * r2 / T(0.01 * 0.01));
		}
	}

	static T normalize_chi(T target_mass) {
		T W = 0;
		T denom = 0;
		for (int i = 0; i < parts.size(); i++) {
			W += volume[i] * sourceV[i];
			denom += volume[i] * sourceV[i] * chi[i];
		}
		const T C = -(T(4) * T(M_PI) * target_mass + W) / denom;
		for (auto &c : chi) {
			c *= C;
		}
		return W;
	}

	static void update_masses() {
		for (int i = 0; i < parts.size(); i++) {
			masses[i] = volume[i] * sourceV[i] * (T(1) + chi[i]) / (T(4) * T(M_PI));
		}
	}

	enum class gravity_method { fmm, bruteforce };

	static void iterate_brill(gravity_method method, int iterations = 40, T target_mass = T(0.01), T omega = T(0.25)) {
		std::vector<T> chi_old(parts.size());
		std::vector<T> chi_next(parts.size());
		const char *name = method == gravity_method::fmm ? "FMM" : "brute";
		for (int iter = 0; iter < iterations; iter++) {
			normalize_chi(target_mass);
			chi_old = chi;
			update_masses();
			for (auto &f : forces) {
				f.init();
			}
			auto const t0 = std::chrono::steady_clock::now();

			size_t flops = 0;
			if (method == gravity_method::fmm) {
				reset_counters();
				flops += form_trees();
				flops += compute_gravity();
			} else {
				flops += compute_gravity_bruteforce();
			}

			auto const t1 = std::chrono::steady_clock::now();
			const double seconds = std::chrono::duration<double>(t1 - t0).count();
			for (size_t i = 0; i < parts.size(); i++) {
				chi[i] = forces[i].potential;
			}
			normalize_chi(target_mass);
			chi_next = chi;
			T l2 = 0;
			T norm = 0;
			for (size_t i = 0; i < parts.size(); i++) {
				const T damped = (T(1) - omega) * chi_old[i] + omega * chi_next[i];
				l2 += sfmm::sqr(damped - chi_old[i]);
				norm += sfmm::sqr(damped);
				chi[i] = damped;
			}
			const T rel_update = norm > T(0) ? std::sqrt(l2 / norm) : T(0);
			printf("%s iter %02i rel_update = %.12e flops = %zu time = %.6f s\n", name, iter, double(rel_update), flops, seconds);
		}
		normalize_chi(target_mass);
	}

	static void compare_final_chi(const std::vector<T> &chi_fmm, const std::vector<T> &chi_exact) {
		if (chi_fmm.size() != chi_exact.size() || chi_fmm.size() != parts.size()) {
			fprintf(stderr,
				"comparison size mismatch: FMM=%zu exact=%zu parts=%zu\n",
				chi_fmm.size(), chi_exact.size(), parts.size());
			abort();
		}

		T error_l1 = 0;
		T error_l2_squared = 0;
		T error_linf = 0;
		T exact_l1 = 0;
		T exact_l2_squared = 0;
		T exact_linf = 0;
		size_t linf_index = 0;

		printf("# pointwise FMM versus direct solution\n");
		printf("# index x y z chi_exact chi_fmm error abs_error\n");
		for (size_t i = 0; i < chi_fmm.size(); i++) {
			const T error = chi_fmm[i] - chi_exact[i];
			const T abs_error = std::abs(error);
			const T abs_exact = std::abs(chi_exact[i]);

			error_l1 += abs_error;
			error_l2_squared += error * error;
			exact_l1 += abs_exact;
			exact_l2_squared += chi_exact[i] * chi_exact[i];
			exact_linf = std::max(exact_linf, abs_exact);
			if (abs_error > error_linf) {
				error_linf = abs_error;
				linf_index = i;
			}

			printf("%zu %.16e %.16e %.16e %.16e %.16e %+.16e %.16e\n",
				i,
				double(parts[i][0].to_double()),
				double(parts[i][1].to_double()),
				double(parts[i][2].to_double()),
				double(chi_exact[i]),
				double(chi_fmm[i]),
				double(error),
				double(abs_error));
		}

		const T count = T(chi_fmm.size());
		const T l1 = count > T(0) ? error_l1 / count : T(0);
		const T l2 = count > T(0) ? std::sqrt(error_l2_squared / count) : T(0);
		const T linf = error_linf;

		const T relative_l1 = exact_l1 > T(0) ? error_l1 / exact_l1 : T(0);
		const T relative_l2 = exact_l2_squared > T(0)
			? std::sqrt(error_l2_squared / exact_l2_squared)
			: T(0);
		const T relative_linf = exact_linf > T(0) ? error_linf / exact_linf : T(0);

		printf("# FMM error norms relative to direct solution\n");
		printf("L1_abs   = %.12e\n", double(l1));
		printf("L2_abs   = %.12e\n", double(l2));
		printf("Linf_abs = %.12e\n", double(linf));
		printf("L1_rel   = %.12e\n", double(relative_l1));
		printf("L2_rel   = %.12e\n", double(relative_l2));
		printf("Linf_rel = %.12e\n", double(relative_linf));
		printf("Linf_index = %zu\n", linf_index);
		if (!chi_fmm.empty()) {
			printf("Linf_exact = %.12e\n", double(chi_exact[linf_index]));
			printf("Linf_fmm   = %.12e\n", double(chi_fmm[linf_index]));
		}
	}

	static void run_brill_comparison(int iterations = 40, T target_mass = T(0.01), T omega = T(0.25)) {
		// Build the tree once to establish the final particle permutation before
		// saving the common initial iterate used by both solvers.
		form_trees();
		const std::vector<T> chi_initial = chi;
		chi = chi_initial;
		iterate_brill(gravity_method::fmm, iterations, target_mass, omega);
		const std::vector<T> chi_fmm = chi;
		chi = chi_initial;
		iterate_brill(gravity_method::bruteforce, iterations, target_mass, omega);
		const std::vector<T> chi_exact = chi;
		compare_final_chi(chi_fmm, chi_exact);
		chi = chi_fmm;
	}

	static size_t particle_count() {
		return parts.size();
	}

	static void show_counters() {
		printf("Node count = %lli\n", (long long)node_count);
		printf("P2P = %lli (%e)\n", (long long)p2p, (long long)p2p / (double)node_count);
		printf("P2L = %lli (%e)\n", (long long)p2l, (long long)p2l / (double)node_count);
		printf("M2P = %lli (%e)\n", (long long)m2p, (long long)m2p / (double)node_count);
		printf("M2L = %lli (%e)\n", (long long)m2l, (long long)m2l / (double)node_count);
		printf("P2P_ewald = %lli (%e)\n", (long long)p2p_ewald, (long long)p2p_ewald / (double)node_count);
		printf("P2L_ewald = %lli (%e)\n", (long long)p2l_ewald, (long long)p2l_ewald / (double)node_count);
		printf("M2P_ewald = %lli (%e)\n", (long long)m2p_ewald, (long long)m2p_ewald / (double)node_count);
		printf("M2L_ewald = %lli (%e)\n", (long long)m2l_ewald, (long long)m2l_ewald / (double)node_count);
	}

	static void reset_counters() {
		node_count = p2p = m2p = p2l = m2l = p2p_ewald = m2p_ewald = p2l_ewald = m2l_ewald = 0;
	}
};

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::vector<sfmm::vec3<FT>> tree<T, V, FT, FV, ORDER, FLAGS>::parts;

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::vector<sfmm::force_type<T>> tree<T, V, FT, FV, ORDER, FLAGS>::forces;

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::vector<T> tree<T, V, FT, FV, ORDER, FLAGS>::masses;

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::vector<T> tree<T, V, FT, FV, ORDER, FLAGS>::chi;

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::vector<T> tree<T, V, FT, FV, ORDER, FLAGS>::sourceV;

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::vector<T> tree<T, V, FT, FV, ORDER, FLAGS>::volume;

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
const T tree<T, V, FT, FV, ORDER, FLAGS>::theta_max = 0.3;

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::atomic<int> tree<T, V, FT, FV, ORDER, FLAGS>::threads_avail(2 * std::thread::hardware_concurrency() - 1);

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
const T tree<T, V, FT, FV, ORDER, FLAGS>::hsoft = 1e-6;

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
const int tree<T, V, FT, FV, ORDER, FLAGS>::Ngrid = 1;

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::vector<tree<T, V, FT, FV, ORDER, FLAGS>> tree<T, V, FT, FV, ORDER, FLAGS>::forest;

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, FT, FV, ORDER, FLAGS>::p2p(0);

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, FT, FV, ORDER, FLAGS>::m2p(0);

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, FT, FV, ORDER, FLAGS>::p2l(0);

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, FT, FV, ORDER, FLAGS>::m2l(0);

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, FT, FV, ORDER, FLAGS>::p2p_ewald(0);

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, FT, FV, ORDER, FLAGS>::m2p_ewald(0);

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, FT, FV, ORDER, FLAGS>::p2l_ewald(0);

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, FT, FV, ORDER, FLAGS>::m2l_ewald(0);

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, FT, FV, ORDER, FLAGS>::node_count(0);

template <class T, class V, class FT, class FV, int ORDER, int FLAGS>
struct run_tests {
	void operator()() const {
		using tree_type = tree<T, V, FT, FV, ORDER, FLAGS>;
		srand(1);
		tree_type::initialize();
		printf("Brill source particles = %zu, order = %i\n", tree_type::particle_count(), ORDER);
		tree_type::sort_grid();
		tree_type::run_brill_comparison();
		tree_type::show_counters();
		tree_type::reset_counters();
		run_tests<T, V, FT, FV, ORDER + 1, FLAGS> run;
		run();
	}
};

template <class T, class V, class FT, class FV, int FLAGS>
struct run_tests<T, V, FT, FV, PMAX + 1, FLAGS> {
	void operator()() const {
	}
};

void ewald() {
	/*	constexpr int N = 8;
	 sfmm::multipole<double, N> M;
	 sfmm::expansion<double, N> L;
	 M.init();
	 M[0] = 1.0;
	 sfmm::vec3<double> dx;
	 for (double x = 0.0; x < 0.5; x += 0.001) {
	 dx[2] = dx[1] = double(0.0);
	 dx[0] = double(x);
	 sfmm::M2L_ewald(L, M, dx);
	 printf("%e ", x);
	 for (int i = 0; i < L.size(); i++) {
	 printf("%e ", L[i]);
	 }
	 printf("\n");
	 }*/
}

void random_unit(double &x, double &y, double &z) {
	const double theta = acos(2 * rand1() - 1.0);
	const double phi = rand1() * 2.0 * M_PI;
	x = cos(phi) * sin(theta);
	y = sin(phi) * sin(theta);
	z = cos(theta);
}

template <int P>
void test2() {
	/*sfmm::vec3<double> dx0;
	 sfmm::vec3<double> dx1;
	 sfmm::vec3<double> dx2;
	 sfmm::vec3<double> dx3;
	 sfmm::vec3<double> dx4;
	 sfmm::vec3<double> x0;
	 sfmm::vec3<double> x1;
	 sfmm::vec3<double> x2;
	 sfmm::vec3<double> x3;
	 sfmm::vec3<double> x4;
	 sfmm::vec3<double> x5;
	 constexpr int n = 1000;
	 double poterr = 0, potnorm = 0;
	 double ferr = 0, fnorm = 0;
	 for (int i = 0; i < n; i++) {
	 rand1();
	 random_unit(dx0[0], dx0[1], dx0[2]);
	 random_unit(dx1[0], dx1[1], dx1[2]);
	 random_unit(dx2[0], dx2[1], dx2[2]);
	 random_unit(dx3[0], dx3[1], dx3[2]);
	 random_unit(dx4[0], dx4[1], dx4[2]);
	 const auto alpha = 0.25;
	 dx0 *= alpha * 0.125;
	 dx1 *= alpha * 0.125;
	 dx2 *= alpha;
	 dx3 *= alpha * 0.125;
	 dx4 *= alpha * 0.125;
	 x0 = sfmm::vec3<double>( { 0, 0, 0 });
	 x1 = x0 + dx0;
	 x2 = x1 + dx1;
	 x3 = x2 + dx2;
	 x4 = x3 + dx3;
	 x5 = x4 + dx4;

	 sfmm::multipole<double, P> M, M1;
	 sfmm::expansion<double, P> L1;
	 sfmm::expansion<double, P> L;
	 sfmm::force_type<double> f1;
	 M.init();
	 L.init();
	 M1.init();
	 //		x0 *= 0.0;
	 //	x1 *= 0.0;
	 //x3 *= 0.0;
	 //	x4 *= 0.0;
	 sfmm::P2M(M, double(0.5), x0 - x1);
	 sfmm::P2M(M1, double(0.5), { 0, 0, 0 });
	 sfmm::M2M(M, x1 - x2);
	 sfmm::M2L_ewald(L1, M1, { 0, 0, 0 });
	 L += L1;
	 sfmm::M2L_ewald(L1, M, x3 - x2);
	 L += L1;
	 sfmm::M2L(L1, M, x3 - x2, sfmmWithSingleRotationOptimization);
	 L += L1;
	 f1.init();
	 sfmm::L2L(L, x3 - x4);
	 sfmm::L2P(f1, L, x4 - x5);
	 sfmm::force_type<double> f2;
	 f2.init();
	 //	f1.init();
	 P2P(f2, 0.5, x5 - x0);
	 P2P_ewald(f2, 0.5, { 0, 0, 0 });
	 P2P_ewald(f2, 0.5, x5 - x0);
	 poterr += sfmm::sqr(f1.potential - f2.potential);
	 potnorm += sfmm::sqr(f1.potential);
	 const double fabs1 = sfmm::sqr(f1.force[0]) + sfmm::sqr(f1.force[1]) + sfmm::sqr(f1.force[2]);
	 const double fabs2 = sfmm::sqr(f2.force[0]) + sfmm::sqr(f2.force[1]) + sfmm::sqr(f2.force[2]);
	 ferr += sfmm::sqr(fabs1 - fabs2);
	 fnorm += sfmm::sqr(fabs2);
	 //	printf( "%e\n", f1.force[0]*f2.force[0]+f1.force[1]*f2.force[1]+f1.force[2]*f2.force[2]);
	 }
	 printf("%i %e %e\n", P, poterr / potnorm, ferr / fnorm);
	 */
}

int main(int argc, char **argv) {
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_OVERFLOW);
	feenableexcept(FE_INVALID);
	for (double a = 0.0; a < 1.0; a += 0.1) {
		for (double b = 0.0; b < 1.0; b += 0.1) {
			sfmm::fixed32 A(a);
			sfmm::fixed32 B(b);
			sfmm::simd_fixed32 aa, bb;
			aa[0] = A;
			bb[0] = B;
			auto d = sfmm::distance(aa, bb)[0];
		}
	}

	sfmm::vec3<double> t;
	sfmm::vec3<float> t1(t);
	//	ewald();
	//	return 0;
	// return 0;
	// test2<5>();
	// test2<6>();
	// test2<7>();
	// test2<8>();
	//	 return 0;
	// ewald();
	// return 0;
	run_tests<float, simd::simd_f32, sfmm::fixed32, sfmm::simd_fixed32, PMIN, sfmmWithBestOptimization> run5;
	run_tests<double, simd::simd_f64, sfmm::fixed64, sfmm::simd_fixed64, PMIN, sfmmWithDoubleRotationOptimization> run4;
	run5();
	run4();
	return 0;
}