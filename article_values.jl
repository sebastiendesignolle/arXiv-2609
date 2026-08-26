module ArticleValues

using PolynomialRoots: roots

export mu_coefficients, lambda_kn, candidate_eta

"""Return the coefficients of μ_{k,n}, ordered by increasing powers of the variable."""
function mu_coefficients(k::Int, n::Int)
    k >= 1 || throw(ArgumentError("k must be positive"))
    n >= 1 || throw(ArgumentError("n must be positive"))

    coefficients = zeros(Rational{BigInt}, n + 1)
    for m in 0:min(k, n)
        numerator = (-big(1))^m * binomial(big(k), big(m)) * factorial(big(n))
        denominator = factorial(big(n - m)) * big(n)^m
        coefficients[n - m + 1] = numerator // denominator
    end
    return coefficients
end

"""Return the largest zero λ_{k,n} of μ_{k,n}, evaluated in standard precision."""
function lambda_kn(k::Int, n::Int)
    spectrum = roots(mu_coefficients(k, n))
    scale = max(1.0, maximum(abs(real(z)) for z in spectrum))
    imaginary_error = maximum(abs(imag(z)) for z in spectrum)
    imaginary_error <= 1e-8 * scale || error("μ_{$k,$n} produced non-real numerical roots")
    return maximum(real(z) for z in spectrum)
end

"""Return the candidate universal value λ_{k,n}/k used in the article."""
candidate_eta(k::Int, n::Int) = lambda_kn(k, n) / k

end # module ArticleValues
