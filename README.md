# spherical_fmm

## The `brille` executable

Source: [`src/brille.cpp`](src/brille.cpp) — a single translation unit, built by
`add_executable(brille ./src/brille.cpp)` in [`CMakeLists.txt`](CMakeLists.txt).
It links `simd` and includes the generated `sfmm.hpp` from
`build/generated_code/include/`.

It solves the Brill-wave Hamiltonian constraint by fixed-point iteration on the
conformal factor `chi`, running the same iteration twice — once with the FMM
tree solver and once with direct summation — and reports the difference.

Everything hangs off the class template
`tree<T, V, FT, FV, ORDER, FLAGS>`: `T`/`V` are the scalar and SIMD real types,
`FT`/`FV` the scalar and SIMD fixed-point position types. Particle state
(`parts`, `forces`, `masses`, `chi`, `sourceV`, `volume`, `forest`) lives in
static members, so it is shared per template instantiation rather than per
object.

### Call tree

```
main()                                                                brille.cpp:1084
├─ feenableexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID)                    :1085
├─ sfmm::distance(simd_fixed32, simd_fixed32)   10x10 warm-up loop            :1088
├─ run_tests<float,  simd_f32, fixed32, simd_fixed32,
│            PMIN, sfmmWithBestOptimization>()                                :1113
└─ run_tests<double, simd_f64, fixed64, simd_fixed64,
             PMIN, sfmmWithDoubleRotationOptimization>()                      :1114

run_tests<...,ORDER,...>::operator()                                          :960
├─ srand(1)
├─ tree::initialize()                                                         :723
│  └─ tree::brill_source()          per TEST_SIZE^3 grid cell                 :706
├─ tree::particle_count()                                                     :877
├─ tree::sort_grid()                recursive spatial split into `forest`     :114
│  └─ tree::partition_parts()                                                 :91
├─ tree::run_brill_comparison()                                               :862
│  ├─ tree::form_trees()            once, to fix the particle permutation     :865
│  ├─ tree::iterate_brill(gravity_method::fmm)                                :868
│  ├─ tree::iterate_brill(gravity_method::bruteforce)                         :871
│  └─ tree::compare_final_chi()     relative L2, max abs/rel deviation        :826
├─ tree::show_counters()                                                      :881
├─ tree::reset_counters()                                                     :893
└─ run_tests<...,ORDER+1,...>()     recurses; empty specialization at PMAX+1  :975

tree::iterate_brill()               `iterations` sweeps, default 40           :783
├─ tree::normalize_chi()            rescale chi to hit target_mass            :761
├─ tree::update_masses()            masses = volume*sourceV*(1+chi)/4pi       :775
├─ force_type::init()               per particle
├─ FMM path                                                                   :797
│  ├─ tree::reset_counters()
│  ├─ tree::form_trees()
│  └─ tree::compute_gravity()
├─ brute path                                                                 :802
│  └─ tree::compute_gravity_bruteforce()
├─ chi[i] = forces[i].potential  ->  tree::normalize_chi()                    :807
└─ damped update chi = (1-omega)*chi_old + omega*chi_next, print residual      :814

tree::form_trees()                  std::async per root in `forest`           :178
└─ tree::form_tree()                recursive, depth-cycled splitting dim     :193
   ├─ multipole.init(scale)
   ├─ tree::partition_parts()                                                 :216
   ├─ internal node  (nparts > BUCKET_SIZE)
   │  ├─ form_tree(LEFT)            via std::async while threads_avail >= 0   :225
   │  ├─ form_tree(RIGHT)           always on the calling thread              :230
   │  ├─ sfmm::M2M()                shift each child multipole to the center  :247
   │  └─ sfmm::reduce_sum()
   └─ leaf node
      ├─ sfmm::load / apply_padding / create_mask   per SIMD lane group
      ├─ sfmm::P2M()                                                          :280
      └─ sfmm::reduce_sum()

tree::compute_gravity()             std::async per root, seeded with the      :294
│                                   full root checklist
└─ tree::compute_cell_gravity()     recursive dual-tree walk                  :367
   ├─ sfmm::L2L()                   inherit parent expansion (if parent)      :376
   ├─ sfmm::P2P()                   leaf self-potential subtraction           :382
   ├─ walk loop                     SIMD-batched acceptance test              :401
   │  ├─ sfmm::distance()                                                     :411
   │  └─ classify into M2Llist / P2Llist / M2Plist / P2Plist,
   │     else open the node onto `nextlist`
   ├─ sfmm::M2L()                   -> local expansion                        :489
   ├─ sfmm::P2L()                   -> local expansion                        :517
   ├─ leaf only
   │  ├─ sfmm::M2P()                -> forces, direct from multipoles         :544
   │  └─ sfmm::P2P()                -> forces, particle-particle              :586
   ├─ internal node
   │  ├─ compute_cell_gravity(LEFT)   via std::async when threads allow       :603
   │  └─ compute_cell_gravity(RIGHT)                                          :608
   └─ leaf node
      └─ sfmm::L2P()                expansion -> per-particle force           :627

tree::compute_gravity_bruteforce()  O(N^2) reference solver                   :319
└─ std::async x min(2*hardware_concurrency, nparts), split over sinks
   ├─ sfmm::P2P()                   every source against the sink             :349
   └─ sfmm::P2P()                   self-interaction, subtracted off          :354
```

### Compile-time configuration

| Knob | Value | Where |
| --- | --- | --- |
| `PMIN` / `PMAX` | 3 / 3 | `CMakeLists.txt` -> `-DPMIN`/`-DPMAX` |
| `TEST_SIZE` | 32 | source grid is 32^3, keeping cells above a 1e-10 threshold |
| `BUCKET_SIZE` | 64 | max particles in a leaf |
| `theta_max` | 0.7 | opening angle; `c0 = 1/theta_max` |
| `hsoft` | 1e-6 | force softening / minimum node radius |
| `Ngrid` | 1 | `sort_grid` subdivision count |

Because `PMIN == PMAX == 3`, the `run_tests` recursion terminates after a single
expansion order, so each precision runs order 3 only.

### Things worth knowing before you read the code

- **The Ewald branches never execute.** The loop at
  [`brille.cpp:394`](src/brille.cpp#L394) is `for (int ewald = 0; ewald <= 0; ewald++)`,
  so `ewald` is always 0. Every `sfmm::*_ewald` call, the `echecklist`, and the
  `fmax(D, V(0.5) - rsum)` periodic correction at line 425 are unreachable as
  written, as are the `*_ewald` counters printed by `show_counters`.
- **`sort_grid` produces exactly one root.** With `Ngrid = 1` the initial cell
  span is 1 in every dimension, so the `largest == 1` branch fires immediately
  and `partition_parts` is never reached from `sort_grid` — the recursion and
  the `forest` vector only matter if `Ngrid` is raised.
- **Threading is cooperative, not pooled.** `form_tree` and
  `compute_cell_gravity` each decrement the shared `threads_avail` counter and
  fall back to serial recursion once it goes negative. The counter starts at
  `2 * hardware_concurrency() - 1`.
- **Dead code.** `compare_analytic` (:636), `prefetch` (:34), `random_unit`
  (:999), `test2<P>` (:1007) and `ewald` (:980) are never called — the last two
  have their bodies commented out entirely. `rand1` (:30) is only reachable from
  them.
- **The program traps on FP exceptions** by design (`main` unmasks
  `FE_DIVBYZERO`, `FE_OVERFLOW`, `FE_INVALID`). A `SIGFPE` is a real numerical
  event, not a crash in the usual sense.
