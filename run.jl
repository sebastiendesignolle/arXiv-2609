include("universal_parent_sos_symmetrized.jl")
using .SymmetrizedParentSOS

using JuMP
using Hypatia
using MultiFloats
using PolynomialRoots

function μ(k, n)
    c = zeros(Rational{BigInt}, n+1)
    for m in 0:min(k, n)
        c[n-m+1] = big(-1)^m * binomial(big(k), big(m)) * factorial(big(n)) // (factorial(big(n-m)) * big(n)^big(m))
    end
    return c
end

function λ(k, n)
    sp = roots(μ(k ,n))
    @assert isreal(sp)
    return maximum(real.(sp))
end

# number of measurements
k = 3
# number of outcomes
n = 4
# level of the hierarchy
t = 3
# conjectured value
reference_eta = λ(k, n) / k
# build and solve the sdp (by default with Hypatia, not very precise in this case)
problem = solve_symmetrized_parent_sdp(Float64; k, n, t, reference_eta)

# solve the accompanying high-precision sdp
model = read_from_file("parent_k3_n4_t3_x4.dat-s"; coefficient_type = Float64x4);
set_optimizer(model, Hypatia.Optimizer{Float64x4})
optimize!(model)
println(reference_eta - objective_value(model))
