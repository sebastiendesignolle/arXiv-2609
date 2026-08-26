include(joinpath(@__DIR__, "universal_parent_sos_symmetrized.jl"))
using .SymmetrizedParentSOS

using JuMP
using MultiFloats

const T = Float64x4
const DAT_S_DIR = get(ENV, "DAT_S_DIR", joinpath(@__DIR__, "results", "dat-s"))
const DAT_S_PATH = isempty(ARGS) ? joinpath(DAT_S_DIR, "parent_k3_n4_t3_x4.dat-s") : ARGS[1]

mkpath(dirname(DAT_S_PATH))

# The distributed high-precision .dat-s file can be solved directly. If it is absent, regenerate the reduced SDP first.
if !isfile(DAT_S_PATH)
    println("No saved high-precision instance found; generating ", DAT_S_PATH)
    build_symmetrized_parent_sdp(
        T;
        k = 3,
        n = 4,
        t = 3,
        dat_s_path = DAT_S_PATH,
        silent = true,
        verbose = true,
    )
end

model = solve_dat_s(DAT_S_PATH, T; silent = true, verbose = false)
reference_big = setprecision(BigFloat, 256) do
    (one(BigFloat) + cos(big(pi) / 9)) / 3
end
reference = T(reference_big)
bound = objective_value(model)

println("reference             = ", reference)
println("hierarchy bound       = ", bound)
println("reference - bound     = ", reference - bound)
println("termination status    = ", termination_status(model))
