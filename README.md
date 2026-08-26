# Symmetry-reduced universal-parent SOS hierarchy

This repository contains the Julia code used for the numerical SOS hierarchy in the article. The implementation builds the
symmetry-reduced SDP for `k` projective measurements with `n` outcomes at hierarchy level `t`, exports it in sparse SDPA
`.dat-s` format, and can solve either a newly generated instance or a previously saved one.

## Files

- `universal_parent_sos_symmetrized.jl` — main module. It builds and solves the reduced hierarchy, exports `.dat-s`
  instances, and provides `solve_dat_s` for saved instances.
- `article_values.jl` — computes the polynomial `μ_{k,n}`, its largest zero `λ_{k,n}`, and the reference value
  `λ_{k,n}/k` used in the article.
- `reproduce_numerics.jl` — reproduces the hierarchy values reported in the numerical tables.
- `reproduce_k3_n4_high_precision.jl` — reproduces the high-precision `(k,n,t) = (3,4,3)` computation.
- `Project.toml` — Julia environment for the repository.

The generated numerical output is written to `results/`; sparse SDPA instances are stored by default in `results/dat-s/`.

## Setup

From the repository directory, instantiate the Julia environment once:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

The main solver is Hypatia. The high-precision example uses `MultiFloats.jl`, while `article_values.jl` uses
`PolynomialRoots.jl` only for the analytical reference values.

## Reproducing the article values

The routine numerical script accepts the hierarchy levels to run as command-line arguments. For example,

```bash
julia --project reproduce_numerics.jl 1 2 3
```

runs the cases used in the article at levels 1-3. Level 4 can be run separately with

```bash
julia --project reproduce_numerics.jl 4
```

since these instances are substantially larger. Results are written as CSV files `results/hierarchy_level_t.csv`.

If the corresponding `.dat-s` file already exists, the script solves it directly instead of rebuilding the hierarchy. Set

```bash
REBUILD=1 julia --project reproduce_numerics.jl 3
```

to force regeneration. To keep saved instances elsewhere, set `DAT_S_DIR`, for example

```bash
DAT_S_DIR=/path/to/dat-s julia --project reproduce_numerics.jl 1 2 3
```

## Solving a saved `.dat-s` file

Saved instances can be solved directly from the main module:

```julia
include("universal_parent_sos_symmetrized.jl")
using .SymmetrizedParentSOS
using JuMP

model = solve_dat_s("parent_k3_n4_t3.dat-s", Float64)
println(objective_value(model))
```

This path does not reconstruct the SOS hierarchy or its symmetry reduction; it only reads and solves the saved sparse-SDPA
problem. A different floating-point type can be supplied as the second argument.

## High-precision `(3,4,3)` computation

Run

```bash
julia --project reproduce_k3_n4_high_precision.jl
```

to solve `results/dat-s/parent_k3_n4_t3_x4.dat-s` with `Float64x4` and compare the numerical value with
`(1 + cos(π/9))/3`. If the file is absent, the script first generates it. An explicit path may instead be supplied as the
first argument:

```bash
julia --project reproduce_k3_n4_high_precision.jl /path/to/parent_k3_n4_t3_x4.dat-s
```

## Using the module directly

A hierarchy instance can be generated and solved with

```julia
include("universal_parent_sos_symmetrized.jl")
using .SymmetrizedParentSOS

problem = solve_symmetrized_parent_sdp(
    Float64;
    k = 3,
    n = 4,
    t = 3,
    dat_s_path = "parent_k3_n4_t3.dat-s",
)

println(problem.result.eta)
```

`build_symmetrized_parent_sdp` performs the same construction and export without solving. The main module also exposes the
reconstruction helpers needed to lift reduced Gram blocks back to the original word bases.

## Practical notes

The symmetry reduction is numerical, while the marginal word sieve is exact. Large level-4 instances can require substantial
memory during construction; when a `.dat-s` file has already been generated, `solve_dat_s` is therefore the preferred route.
For multiprecision calculations, coefficients should be generated and read using the same scalar type whenever possible.
