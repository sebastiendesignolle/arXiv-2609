include(joinpath(@__DIR__, "universal_parent_sos_symmetrized.jl"))
include(joinpath(@__DIR__, "article_values.jl"))
using .SymmetrizedParentSOS
using .ArticleValues

using JuMP

# Parameter pairs computed at each level in the article. Level four contains the three large runs discussed explicitly
# in the main text. Pass the desired levels on the command line, e.g. `julia --project reproduce_numerics.jl 1 2 3`.
const FULL_GRID = [(k, n) for k in 2:8 for n in 2:8]
const LEVEL_THREE = [
    (k, n) for k in 2:8 for n in 2:8 if
    k <= 4 || (k == 5 && n <= 6) || (k == 6 && n <= 5) || (k >= 7 && n == 2)
]
const ARTICLE_CASES = Dict(
    1 => FULL_GRID,
    2 => FULL_GRID,
    3 => LEVEL_THREE,
    4 => [(3, 5), (3, 6), (4, 4)],
)

const LEVELS = isempty(ARGS) ? [1] : parse.(Int, ARGS)
all(haskey(ARTICLE_CASES, t) for t in LEVELS) || error("levels must lie in 1:4")

const RESULTS_DIR = joinpath(@__DIR__, "results")
const DAT_S_DIR = get(ENV, "DAT_S_DIR", joinpath(RESULTS_DIR, "dat-s"))
const REBUILD = get(ENV, "REBUILD", "0") == "1"
mkpath(RESULTS_DIR)
mkpath(DAT_S_DIR)

function solve_case(k::Int, n::Int, t::Int)
    path = joinpath(DAT_S_DIR, "parent_k$(k)_n$(n)_t$(t).dat-s")
    if isfile(path) && !REBUILD
        return solve_dat_s(path, Float64; silent = true, verbose = false)
    end

    problem = solve_symmetrized_parent_sdp(
        Float64;
        k = k,
        n = n,
        t = t,
        dat_s_path = path,
        silent = true,
        verbose = false,
    )
    return problem.model
end

for t in LEVELS
    csv_path = joinpath(RESULTS_DIR, "hierarchy_level_$t.csv")
    open(csv_path, "w") do io
        println(io, "k,n,t,bound,candidate,candidate_minus_bound,termination_status")
        for (k, n) in ARTICLE_CASES[t]
            model = solve_case(k, n, t)
            candidate = candidate_eta(k, n)
            status = termination_status(model)

            if !has_values(model)
                println(io, "$k,$n,$t,NaN,$candidate,NaN,$status")
                println("(k,n,t)=($k,$n,$t): no numerical objective, status=$status")
                continue
            end

            bound = objective_value(model)
            gap = Float64(candidate - bound)
            println(io, "$k,$n,$t,$bound,$candidate,$gap,$status")
            println("(k,n,t)=($k,$n,$t): bound=$bound, candidate-bound=$gap")
        end
    end
    println("wrote ", csv_path)
end
