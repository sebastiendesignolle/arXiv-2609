"""
Symmetry-reduced universal-parent SOS hierarchy.

This single module constructs and solves only the symmetry-reduced SDP for
`k` projective measurements with `n` outcomes at SOS level `1 <= t`.
The last outcome of every measurement is eliminated, and the exact marginal
word sieve is always applied.  The full reconstruction isometry for every
Wedderburn component is stored, so reduced numerical blocks can be lifted back
to the original Gram matrices with `reconstruct_gram_matrices`.

The implementation uses a length-parametric `Word{L}` type.  Operations that
preserve or predict the word length use `ntuple(..., Val(L))`, allowing the
compiler to infer concrete word lengths whenever the calling context permits.

The reduced model is always exported to sparse SDPA `.dat-s` format and then
read back as the geometric-form solver model.  This avoids the numerically less
stable direct standard-form construction with PSD variables and many equality
constraints.  Coefficients are printed in the active scalar type, including
arbitrary-precision `BigFloat` decimals.

Required packages:

    import Pkg
    Pkg.activate(".")
    Pkg.up()

Typical use:

    include("universal_parent_sos_symmetrized.jl")
    using .SymmetrizedParentSOS

    setprecision(BigFloat, 256) do
        problem = solve_symmetrized_parent_sdp(
            BigFloat;
            n = 4,
            k = 3,
            t = 3,
            save_path = "parent_k3_n4_t3.dat",
        )

        Q, K = reconstruct_gram_matrices(
            problem.reduction,
            problem.result.Q_blocks,
            problem.result.K_blocks,
        )
    end
"""
module SymmetrizedParentSOS

using GenericLinearAlgebra
import Hypatia
using JuMP
using LinearAlgebra
using PolynomialRoots
using Random
using Serialization
using SparseArrays

export Word,
    setting_of,
    outcome_of,
    concrete_letter,
    reduce_abstract,
    reduce_concrete,
    abstract_words,
    concrete_words,
    marginal_allowed_letters,
    marginal_residual_basis,
    build_abstract_gram_map,
    build_coefficient_data,
    SymmetryComponent,
    SymmetryDecomposition,
    ParentSymmetryReduction,
    CompactCoefficientData,
    compact_reduction,
    ReducedNormalisationEquation,
    ReducedMarginalEquation,
    block_signature,
    full_isometry,
    setting_generators,
    normalisation_generators,
    marginal_generators,
    basis_permutations,
    orbit_partition,
    word_orbit_partition,
    symmetric_pair_orbits,
    stable_svd,
    decompose_permutation_representation,
    decomposition_error_scale,
    recommended_hypatia_rank_tolerance,
    zero_visibility_feasibility_diagnostics,
    reconstruct_matrix,
    compress_matrix,
    reconstruct_gram_matrices,
    save_reduction,
    load_reduction,
    build_parent_symmetry_reduction,
    reduced_row_blocks,
    dat_s_block_layout,
    write_dat_s,
    default_dat_s_path,
    extract_sdpa_standard_solution,
    build_symmetrized_parent_sdp,
    solve_symmetrized_parent_sdp,
    self_test

"""
    Word{L}

A reduced noncommutative word with `L` integer letters.  Encoding the length as
a type parameter makes length-preserving operations type-stable.  Mixed-degree
bases are still stored as `Vector{Word}`, but once a concrete `Word{L}` reaches
a specialised method its length is known to the compiler.
"""
struct Word{L}
    letters::NTuple{L,Int}
end

Word() = Word{0}(())
Word(letters::NTuple{L,Int}) where {L} = Word{L}(letters)
Word(letters::Vararg{Int,L}) where {L} = Word(letters)

@inline function _word_from_vector(
    letters::AbstractVector{<:Integer},
    ::Val{L},
) where {L}
    return Word(ntuple(i -> Int(letters[i]), Val(L)))
end

Word(letters::AbstractVector{<:Integer}) =
    _word_from_vector(letters, Val(length(letters)))

Base.convert(::Type{Word}, word::Word) = word
Base.convert(::Type{Word}, letters::NTuple{L,Int}) where {L} = Word(letters)
Base.eltype(::Type{<:Word}) = Int
Base.IteratorEltype(::Type{<:Word}) = Base.HasEltype()
Base.IteratorSize(::Type{<:Word}) = Base.HasLength()
Base.length(::Word{L}) where {L} = L
Base.size(::Word{L}) where {L} = (L,)
Base.axes(::Word{L}) where {L} = (Base.OneTo(L),)
Base.firstindex(::Word) = 1
Base.lastindex(::Word{L}) where {L} = L
Base.getindex(word::Word, index::Int) = word.letters[index]
Base.iterate(word::Word, state...) = iterate(word.letters, state...)
Base.isempty(::Word{0}) = true
Base.isempty(::Word) = false
Base.Tuple(word::Word) = word.letters
Base.reverse(word::Word{L}) where {L} =
    Word(ntuple(i -> word[L - i + 1], Val(L)))
Base.:(==)(left::Word, right::Word) = left.letters == right.letters
Base.isequal(left::Word, right::Word) = isequal(left.letters, right.letters)
Base.hash(word::Word, seed::UInt) = hash(word.letters, seed)
Base.isless(left::Word, right::Word) = isless(left.letters, right.letters)
Base.show(io::IO, word::Word) = show(io, word.letters)

"""Append one letter while preserving the resulting length in the type."""
@inline append_letter(word::Word{L}, letter::Int) where {L} =
    Word(ntuple(i -> i <= L ? word[i] : letter, Val(L + 1)))

"""Concatenate two words with compile-time length `L + M`."""
@inline concatenate(left::Word{L}, right::Word{M}) where {L,M} =
    Word(ntuple(i -> i <= L ? left[i] : right[i - L], Val(L + M)))

"""
    setting_of(letter, n) -> Int

Return the measurement setting of a concrete generator when each measurement
has `n` outcomes and its last outcome has been eliminated.
"""
@inline setting_of(letter::Int, n::Int) = (letter - 1) ÷ (n - 1) + 1

"""Return the explicit outcome label in `1:n-1` represented by `letter`."""
@inline outcome_of(letter::Int, n::Int) = (letter - 1) % (n - 1) + 1

"""Encode the explicit projector `P[outcome|setting]` as one integer letter."""
@inline concrete_letter(setting::Int, outcome::Int, n::Int) =
    (setting - 1) * (n - 1) + outcome


"""
    reduce_abstract(word) -> Word

Collapse adjacent equal setting labels, implementing abstract idempotence.
"""
function reduce_abstract(word::Word{L})::Word where {L}
    output = Int[]
    sizehint!(output, length(word))
    for letter in word
        if isempty(output) || output[end] != letter
            push!(output, letter)
        end
    end
    return Word(output)
end

"""
    reduce_concrete(word, n) -> Union{Nothing,Word}

Reduce a concrete projector word.  Adjacent equal projectors collapse;
adjacent distinct outcomes of the same measurement make the word zero and
return `nothing`.
"""
function reduce_concrete(word::Word{L}, n::Int)::Union{Nothing,Word} where {L}
    n >= 2 || throw(ArgumentError("n must be at least 2"))
    output = Int[]
    sizehint!(output, length(word))
    for letter in word
        if !isempty(output) &&
           setting_of(output[end], n) == setting_of(letter, n)
            output[end] == letter && continue
            return nothing
        end
        push!(output, letter)
    end
    return Word(output)
end

"""
    abstract_words(k, t) -> Vector{Word}

Enumerate reduced words of length at most `t` in one abstract idempotent
variable for each of the `k` measurements.
"""
function abstract_words(k::Int, t::Int)::Vector{Word}
    k >= 1 || throw(ArgumentError("k must be positive"))
    t >= 0 || throw(ArgumentError("t must be nonnegative"))

    words = Word[Word()]
    frontier = Word[Word()]
    for _ in 1:t
        next_frontier = Word[]
        for word in frontier, setting in 1:k
            if isempty(word) || word[end] != setting
                push!(next_frontier, append_letter(word, setting))
            end
        end
        append!(words, next_frontier)
        frontier = next_frontier
    end
    return words
end

"""
    concrete_words(n, k, t; allowed_letters)

Enumerate reduced concrete words of length at most `t` over the supplied
projector alphabet.  The keyword is required internally so that the exact
marginal retraction is always used.
"""
function concrete_words(
    n::Int,
    k::Int,
    t::Int;
    allowed_letters::AbstractVector{Int},
)::Vector{Word}
    n >= 2 || throw(ArgumentError("n must be at least 2"))
    k >= 1 || throw(ArgumentError("k must be positive"))
    t >= 0 || throw(ArgumentError("t must be nonnegative"))

    nind = n - 1
    letters = collect(allowed_letters)
    all(1 <= letter <= k * nind for letter in letters) ||
        throw(ArgumentError("allowed_letters contains an invalid generator"))
    allunique(letters) ||
        throw(ArgumentError("allowed_letters must not contain duplicates"))

    words = Word[Word()]
    frontier = Word[Word()]
    for _ in 1:t
        next_frontier = Word[]
        for word in frontier, letter in letters
            if isempty(word) ||
               setting_of(word[end], n) != setting_of(letter, n)
                push!(next_frontier, append_letter(word, letter))
            end
        end
        append!(words, next_frontier)
        frontier = next_frontier
    end
    return words
end

"""
    marginal_allowed_letters(n, k; selected_setting=1, selected_outcome=1)

Return the projector alphabet retained by the exact marginal sieve.  In the
selected setting only the selected projector remains; every explicit projector
of the other settings remains available.
"""
function marginal_allowed_letters(
    n::Int,
    k::Int;
    selected_setting::Int = 1,
    selected_outcome::Int = 1,
)::Vector{Int}
    n >= 2 || throw(ArgumentError("n must be at least 2"))
    nind = n - 1
    1 <= selected_setting <= k ||
        throw(ArgumentError("selected_setting must lie in 1:k"))
    1 <= selected_outcome <= nind || throw(ArgumentError(
        "selected_outcome must lie in 1:n-1; outcome 1 is WLOG by symmetry",
    ))

    letters = Int[]
    sizehint!(letters, 1 + (k - 1) * nind)
    for setting in 1:k
        if setting == selected_setting
            push!(letters, concrete_letter(setting, selected_outcome, n))
        else
            for outcome in 1:nind
                push!(letters, concrete_letter(setting, outcome, n))
            end
        end
    end
    return letters
end

"""
    marginal_residual_basis(n, k, t; selected_setting=1, selected_outcome=1)

Construct the exact word basis of the marginal SOS Gram matrix `K`.

Let `A` be the universal *-algebra of the `k` PVMs and let `B` be generated by
the selected projector of `selected_setting` and by every projector of the
other settings.  The map fixing `B` and sending the other explicit projectors
of the selected setting to zero is a unital *-homomorphic retraction `A -> B`.
Because the selected marginal residual lies in `B`, retracting any degree-`2t`
SOS certificate gives a certificate on the following smaller basis without
changing the hierarchy.
"""
function marginal_residual_basis(
    n::Int,
    k::Int,
    t::Int;
    selected_setting::Int = 1,
    selected_outcome::Int = 1,
)::Vector{Word}
    n >= 2 || throw(ArgumentError("n must be at least 2"))
    allowed_letters = marginal_allowed_letters(
        n,
        k;
        selected_setting = selected_setting,
        selected_outcome = selected_outcome,
    )
    return concrete_words(
        n,
        k,
        t;
        allowed_letters = allowed_letters,
    )
end

"""Return the falling factorial `n * (n-1) * ... * (n-r+1)` as a `BigInt`."""
function falling_factorial_big(n::Int, r::Int)::BigInt
    0 <= r <= n || throw(ArgumentError("need 0 <= r <= n"))
    value = big(1)
    for j in 0:(r - 1)
        value *= n - j
    end
    return value
end

"""
    foreach_distinct_relabelling(f, word, k)

Call `f(relabelled_word)` once for every distinct image of `word` under a
permutation of the measurement labels.  Only injections on the settings that
actually occur in `word` are enumerated, avoiding an unnecessary `k!` loop.
"""
function foreach_distinct_relabelling(
    f,
    word::Word{L},
    k::Int,
) where {L}
    used = unique(collect(word))
    isempty(used) && return f(Word())

    mapping = zeros(Int, k)
    taken = falses(k)

    function recurse(position::Int)
        if position > length(used)
            image = Word(ntuple(i -> mapping[word[i]], Val(L)))
            f(image)
            return
        end

        source = used[position]
        for target in 1:k
            taken[target] && continue
            mapping[source] = target
            taken[target] = true
            recurse(position + 1)
            taken[target] = false
        end
    end

    recurse(1)
    return nothing
end

"""
    build_abstract_gram_map(T, k, t)

Build the coefficient map of the setting-symmetrised parent Gram polynomial

    (1/k!) sum_{pi in S_k} v(pi(X))' Q v(pi(X)).

The dictionary maps each reduced abstract word to pairs
`q_column => coefficient`, where `q_column` is the column-major index in
`vec(Q)`.  Distinct orbit words are generated directly.  If a reduced word
uses `r` settings, each orbit element receives weight `1/(k)_r`.
"""
function build_abstract_gram_map(
    ::Type{T},
    k::Int,
    t::Int,
) where {T<:Real}
    T <: Integer && throw(ArgumentError("T must support exact or floating division"))

    words = abstract_words(k, t)
    m = length(words)
    accum = Dict{Word,Dict{Int,T}}()

    for (i, u) in enumerate(words), (j, v) in enumerate(words)
        base_word = reduce_abstract(concatenate(reverse(u), v))
        r = length(unique(base_word))
        orbit_size = falling_factorial_big(k, r)
        coefficient = one(T) / convert(T, orbit_size)
        column = i + (j - 1) * m

        foreach_distinct_relabelling(base_word, k) do relabelled
            coefficients = get!(accum, relabelled, Dict{Int,T}())
            coefficients[column] =
                get(coefficients, column, zero(T)) + coefficient
        end
    end

    gram_map = Dict{Word,Vector{Pair{Int,T}}}()
    for (word, coefficients) in accum
        filter!(pair -> !iszero(last(pair)), coefficients)
        gram_map[word] = collect(coefficients)
    end
    return words, gram_map
end

"""
    projector_expression(T, measurement, outcome, n)

Return the noncommutative polynomial representing one PVM effect after the
last outcome has been eliminated.  Explicit outcomes are single letters;
outcome `n` is represented as identity minus the sum of the first `n-1`
projectors of that setting.
"""
function projector_expression(
    ::Type{T},
    measurement::Int,
    outcome::Int,
    n::Int,
) where {T<:Real}
    nind = n - 1
    if 1 <= outcome <= nind
        letter = concrete_letter(measurement, outcome, n)
        return Dict{Word,T}(Word(letter) => one(T))
    elseif outcome == n
        expression = Dict{Word,T}(Word() => one(T))
        for a in 1:nind
            letter = concrete_letter(measurement, a, n)
            expression[Word(letter)] = -one(T)
        end
        return expression
    end
    throw(ArgumentError("outcome must lie in 1:n"))
end

"""
    polynomial_product(p, q, n)

Multiply two concrete noncommutative polynomials and reduce every monomial in
the universal PVM quotient.  Terms reduced to zero are discarded and equal
reduced words are combined.
"""
function polynomial_product(
    p::Dict{Word,T},
    q::Dict{Word,T},
    n::Int,
) where {T<:Real}
    result = Dict{Word,T}()
    for (u, a) in p, (v, b) in q
        word = reduce_concrete(concatenate(u, v), n)
        isnothing(word) && continue
        result[word] = get(result, word, zero(T)) + a * b
    end
    filter!(pair -> !iszero(last(pair)), result)
    return result
end

"""
    evaluate_abstract_word(T, word, outcomes, n)

Evaluate an abstract setting word at a concrete outcome assignment.  The
vector `outcomes` has one entry per measurement and may be reused by the
recursive summation routines.
"""
function evaluate_abstract_word(
    ::Type{T},
    word::Word,
    outcomes::AbstractVector{Int},
    n::Int,
) where {T<:Real}
    polynomial = Dict{Word,T}(Word() => one(T))
    for measurement in word
        factor = projector_expression(T, measurement, outcomes[measurement], n)
        polynomial = polynomial_product(polynomial, factor, n)
    end
    return polynomial
end

"""Add `scale * source` to `destination`, combining equal words."""
function add_scaled_polynomial!(
    destination::Dict{Word,T},
    source::Dict{Word,T},
    scale::T,
) where {T<:Real}
    for (word, coefficient) in source
        destination[word] =
            get(destination, word, zero(T)) + scale * coefficient
    end
    return destination
end

"""
    summed_abstract_evaluation(T, word, n, k;
                              fixed_setting=nothing,
                              fixed_outcome=nothing)

Sum the concrete evaluation of an abstract word over outcome assignments.

Only settings occurring in `word` are explicitly enumerated.  Every unused
free setting contributes a factor `n`.  If `fixed_setting` is supplied, its
outcome is fixed and the other settings are summed; this is the operation
needed for a marginal.  The result is a concrete polynomial dictionary.
"""
function summed_abstract_evaluation(
    ::Type{T},
    word::Word,
    n::Int,
    k::Int;
    fixed_setting::Union{Nothing,Int} = nothing,
    fixed_outcome::Union{Nothing,Int} = nothing,
) where {T<:Real}
    if xor(isnothing(fixed_setting), isnothing(fixed_outcome))
        throw(ArgumentError(
            "fixed_setting and fixed_outcome must be supplied together",
        ))
    end

    used_settings = unique(collect(word))
    outcomes = ones(Int, k)

    if !isnothing(fixed_setting)
        1 <= fixed_setting <= k || throw(ArgumentError(
            "fixed_setting must lie in 1:k",
        ))
        1 <= fixed_outcome <= n ||
            throw(ArgumentError("fixed_outcome must lie in 1:n"))
        outcomes[fixed_setting] = fixed_outcome
    end

    free_settings = isnothing(fixed_setting) ? used_settings :
        [x for x in used_settings if x != fixed_setting]
    number_unused_free_settings = isnothing(fixed_setting) ?
        k - length(free_settings) :
        k - 1 - length(free_settings)
    multiplicity = convert(T, big(n)^number_unused_free_settings)

    result = Dict{Word,T}()

    function recurse(position::Int)
        if position > length(free_settings)
            evaluated = evaluate_abstract_word(T, word, outcomes, n)
            add_scaled_polynomial!(result, evaluated, multiplicity)
            return
        end

        setting = free_settings[position]
        for outcome in 1:n
            outcomes[setting] = outcome
            recurse(position + 1)
        end
    end

    recurse(1)
    filter!(pair -> !iszero(last(pair)), result)
    return result
end

"""Strict ordering used for deterministic word lists: degree, then lexicographic."""
function word_isless(u::Word, v::Word)
    length(u) != length(v) && return length(u) < length(v)
    return isless(u, v)
end

"""
    build_coefficient_data(T; n=4, k=3, t=3, verbose=true)

Construct the sparse coefficient maps of the universal-parent SDP using a
low-peak-memory, multi-pass algorithm.

The returned maps encode

    An * vec(Q) = e_empty,
    Am * vec(Q) - eta * e_selected = Ah * vec(K).

The former implementation retained every evaluated abstract polynomial in two
nested dictionaries while simultaneously assembling all three sparse matrices.
For large outcome numbers this dominated memory.  The present routine instead:

  1. discovers the two equation supports and discards each evaluation;
  2. assembles `An` in a second pass and releases its triplets;
  3. assembles `Am` in a third pass and releases its triplets;
  4. scans the marginal Gram pairs once more to assemble `Ah` directly.

This repeats inexpensive degree-at-most-six polynomial evaluations, but it never
keeps the large evaluation dictionaries or more than one triplet buffer alive.
The exact marginal retraction basis is unchanged.
"""
function build_coefficient_data(
    ::Type{T};
    n::Int = 4,
    k::Int = 3,
    t::Int = 3,
    selected_setting::Int = 1,
    selected_outcome::Int = 1,
    verbose::Bool = true,
) where {T<:Real}
    n >= 2 || throw(ArgumentError("n must be at least 2"))
    k >= 1 || throw(ArgumentError("k must be positive"))
    t >= 1 || throw(ArgumentError("t must be positive"))
    1 <= selected_setting <= k ||
        throw(ArgumentError("selected_setting must lie in 1:k"))
    1 <= selected_outcome <= n - 1 || throw(ArgumentError(
        "selected_outcome must lie in 1:n-1; outcome 1 is WLOG by symmetry",
    ))

    nind = n - 1
    start_time = time()

    verbose && println(
        "Building setting-symmetrised abstract Gram map for ",
        k,
        " measurements...",
    )
    abstract_basis, gram_map = build_abstract_gram_map(T, k, t)
    qdim = length(abstract_basis)
    residual_basis = marginal_residual_basis(
        n,
        k,
        t;
        selected_setting = selected_setting,
        selected_outcome = selected_outcome,
    )
    kdim = length(residual_basis)

    verbose && println(
        "Discovering equation supports in a low-memory pass over ",
        length(gram_map),
        " abstract coefficient words...",
    )
    normalisation_word_set = Set{Word}([Word()])
    marginal_word_set = Set{Word}()

    for abstract_word in keys(gram_map)
        normalised = summed_abstract_evaluation(T, abstract_word, n, k)
        union!(normalisation_word_set, keys(normalised))
        normalised = nothing

        marginalised = summed_abstract_evaluation(
            T,
            abstract_word,
            n,
            k;
            fixed_setting = selected_setting,
            fixed_outcome = selected_outcome,
        )
        union!(marginal_word_set, keys(marginalised))
        marginalised = nothing
    end

    selected_letter = concrete_letter(
        selected_setting,
        selected_outcome,
        n,
    )
    push!(marginal_word_set, Word(selected_letter))

    verbose && println("Discovering the marginal-residual SOS support...")
    for u in residual_basis, v in residual_basis
        word = reduce_concrete(concatenate(reverse(u), v), n)
        isnothing(word) || push!(marginal_word_set, word)
    end

    normalisation_words = collect(normalisation_word_set)
    marginal_words = collect(marginal_word_set)
    normalisation_word_set = nothing
    marginal_word_set = nothing
    sort!(normalisation_words; lt = word_isless)
    sort!(marginal_words; lt = word_isless)

    normalisation_word_index =
        Dict{Word,Int}(word => i for (i, word) in enumerate(normalisation_words))
    marginal_word_index =
        Dict{Word,Int}(word => i for (i, word) in enumerate(marginal_words))
    GC.gc()

    function assemble_parent_map(
        word_index::Dict{Word,Int},
        number_of_rows::Int;
        fixed_setting::Union{Nothing,Int} = nothing,
        fixed_outcome::Union{Nothing,Int} = nothing,
        label::AbstractString,
    )
        verbose && println("Assembling $label one sparse map at a time...")
        rows = Int[]
        columns = Int[]
        values = T[]

        for (abstract_word, q_entries) in gram_map
            evaluation = if isnothing(fixed_setting)
                summed_abstract_evaluation(T, abstract_word, n, k)
            else
                summed_abstract_evaluation(
                    T,
                    abstract_word,
                    n,
                    k;
                    fixed_setting = fixed_setting,
                    fixed_outcome = fixed_outcome,
                )
            end
            for (concrete_word, evaluation_coefficient) in evaluation
                row = word_index[concrete_word]
                for entry in q_entries
                    push!(rows, row)
                    push!(columns, first(entry))
                    push!(values, evaluation_coefficient * last(entry))
                end
            end
            evaluation = nothing
        end

        result = sparse(rows, columns, values, number_of_rows, qdim * qdim)
        dropzeros!(result)
        rows = nothing
        columns = nothing
        values = nothing
        GC.gc()
        return result
    end

    An = assemble_parent_map(
        normalisation_word_index,
        length(normalisation_words);
        label = "normalisation",
    )
    Am = assemble_parent_map(
        marginal_word_index,
        length(marginal_words);
        fixed_setting = selected_setting,
        fixed_outcome = selected_outcome,
        label = "marginal",
    )

    verbose && println("Assembling the marginal-residual SOS map...")
    sos_rows = Int[]
    sos_columns = Int[]
    sos_values = T[]
    sizehint!(sos_rows, kdim * kdim)
    sizehint!(sos_columns, kdim * kdim)
    sizehint!(sos_values, kdim * kdim)
    for (i, u) in enumerate(residual_basis), (j, v) in enumerate(residual_basis)
        word = reduce_concrete(concatenate(reverse(u), v), n)
        isnothing(word) && continue
        push!(sos_rows, marginal_word_index[word])
        push!(sos_columns, i + (j - 1) * kdim)
        push!(sos_values, one(T))
    end
    Ah = sparse(
        sos_rows,
        sos_columns,
        sos_values,
        length(marginal_words),
        kdim * kdim,
    )
    sos_rows = nothing
    sos_columns = nothing
    sos_values = nothing
    GC.gc()

    elapsed = round(time() - start_time; digits = 2)
    verbose && println(
        "Coefficient data built in $(elapsed) s: Q = $(qdim)x$(qdim), " *
        "K = $(kdim)x$(kdim), normalisation words = " *
        "$(length(normalisation_words)), marginal words = " *
        "$(length(marginal_words)), nnz = " *
        "($(nnz(An)), $(nnz(Am)), $(nnz(Ah))).",
    )

    return (
        n = n,
        k = k,
        t = t,
        nind = nind,
        selected_setting = selected_setting,
        selected_outcome = selected_outcome,
        selected_letter = selected_letter,
        abstract_basis = abstract_basis,
        residual_basis = residual_basis,
        normalisation_words = normalisation_words,
        marginal_words = marginal_words,
        normalisation_word_index = normalisation_word_index,
        marginal_word_index = marginal_word_index,
        An = An,
        Am = Am,
        Ah = Ah,
    )
end


# ---------------------------------------------------------------------------
# Stored decomposition and reconstruction data
# ---------------------------------------------------------------------------

"""
    SymmetryComponent{T}

One real Wedderburn component of a permutation representation.

Fields:

  * `multiplicity`: size `m` of the reduced PSD block;
  * `irrep_dimension`: dimension `r` of the real irreducible representation;
  * `isometry`: an `N × (m*r)` matrix with orthonormal columns, ordered
    copy-by-copy.  Columns `(j-1)r+1:j*r` are the aligned basis of copy `j`;
  * `alignment_error`: maximum generator-intertwining residual produced while
    aligning equivalent copies.

If `B` is the reduced `m × m` block, this component contributes

    isometry * kron(B, I_r) * isometry'

to the original `N × N` invariant matrix.
"""
struct SymmetryComponent{T<:AbstractFloat}
    multiplicity::Int
    irrep_dimension::Int
    isometry::Matrix{T}
    alignment_error::T
end

"""
    SymmetryDecomposition{T}

Complete, explicitly invertible symmetry reduction of one Gram-matrix space.

`basis_permutations` stores the action of the chosen group generators on the
original word basis.  `components` stores the reconstruction isometries.
The diagnostic errors are evaluated in the original basis and should be small
relative to the requested `tolerance`.
"""
struct SymmetryDecomposition{T<:AbstractFloat}
    dimension::Int
    components::Vector{SymmetryComponent{T}}
    basis_permutations::Vector{Vector{Int}}
    symmetric_pair_orbit_count::Int
    splitter_seed::Int
    tolerance::T
    orthogonality_error::T
    representation_error::T
    invariance_test_error::T
end

"""
    ParentSymmetryReduction{T,D}

All symmetry information for one universal-parent hierarchy instance.

The field `data` is the coefficient data returned by
`build_coefficient_data`.  `Q_decomposition` and
`K_decomposition` are the stored reconstruction maps.  The remaining fields
record the generator actions and the representative rows of the normalisation
and marginal coefficient equations.
"""
struct ParentSymmetryReduction{T<:AbstractFloat,D}
    data::D
    Q_decomposition::SymmetryDecomposition{T}
    K_decomposition::SymmetryDecomposition{T}
    normalisation_letter_generators::Vector{Vector{Int}}
    marginal_letter_generators::Vector{Vector{Int}}
    normalisation_basis_permutations::Vector{Vector{Int}}
    marginal_basis_permutations::Vector{Vector{Int}}
    normalisation_orbit_representatives::Vector{Int}
    normalisation_orbit_id::Vector{Int}
    marginal_orbit_representatives::Vector{Int}
    marginal_orbit_id::Vector{Int}
end


"""
    CompactCoefficientData

Small metadata retained after the reduced equations have been built.  The large
unreduced word supports and sparse coefficient maps are deliberately omitted;
the reconstruction isometries, reduced equations, and hierarchy parameters are
sufficient for solving, SDPA export, diagnostics, and Gram reconstruction.
"""
struct CompactCoefficientData
    n::Int
    k::Int
    t::Int
    nind::Int
    selected_setting::Int
    selected_outcome::Int
    selected_letter::Int
    q_dimension::Int
    k_dimension::Int
end

"""
    compact_reduction(reduction)

Drop all unreduced coefficient maps, equation-basis permutations, and orbit-id
arrays from a completed parent reduction.  The returned object remains fully
usable by `reconstruct_gram_matrices`, `block_signature`, `write_dat_s`, and the
solver result.  It cannot be supplied back to `build_symmetrized_parent_sdp`,
because constructing the reduced equations requires the discarded maps.
"""
function compact_reduction(
    reduction::ParentSymmetryReduction{T},
) where {T<:AbstractFloat}
    data = reduction.data
    compact_data = CompactCoefficientData(
        data.n,
        data.k,
        data.t,
        data.nind,
        data.selected_setting,
        data.selected_outcome,
        data.selected_letter,
        reduction.Q_decomposition.dimension,
        reduction.K_decomposition.dimension,
    )
    return ParentSymmetryReduction(
        compact_data,
        reduction.Q_decomposition,
        reduction.K_decomposition,
        reduction.normalisation_letter_generators,
        reduction.marginal_letter_generators,
        Vector{Vector{Int}}(),
        Vector{Vector{Int}}(),
        Int[],
        Int[],
        Int[],
        Int[],
    )
end

is_compact_reduction(reduction::ParentSymmetryReduction) =
    reduction.data isa CompactCoefficientData

"""
    block_signature(decomposition)

Return `(multiplicity, irrep_dimension)` for every Wedderburn component, in the
same order as the reduced PSD blocks and stored isometries.
"""
block_signature(decomposition::SymmetryDecomposition) = [
    (multiplicity = component.multiplicity,
     irrep_dimension = component.irrep_dimension)
    for component in decomposition.components
]

"""
    full_isometry(decomposition)

Concatenate all component isometries.  The result is an orthogonal `N × N`
matrix, up to the numerical decomposition tolerance.
"""
function full_isometry(decomposition::SymmetryDecomposition{T}) where {T}
    isempty(decomposition.components) && return zeros(T, decomposition.dimension, 0)
    return hcat((component.isometry for component in decomposition.components)...)
end

# ---------------------------------------------------------------------------
# Permutations and group generators
# ---------------------------------------------------------------------------

"""Return the identity permutation of `1:n` as a vector of images."""
identity_permutation(n::Int) = collect(1:n)

"""
    transposition(n, i, j)

Return the permutation of `1:n` swapping `i` and `j`.
"""
function transposition(n::Int, i::Int, j::Int)::Vector{Int}
    1 <= i <= n || throw(BoundsError(1:n, i))
    1 <= j <= n || throw(BoundsError(1:n, j))
    permutation = identity_permutation(n)
    permutation[i], permutation[j] = permutation[j], permutation[i]
    return permutation
end

"""
    setting_generators(k)

Adjacent transpositions generating `S_k`, acting on abstract
measurement labels.
"""
function setting_generators(k::Int)::Vector{Vector{Int}}
    k >= 1 || throw(ArgumentError("k must be positive"))
    return [
        transposition(k, x, x + 1)
        for x in 1:(k - 1)
    ]
end

"""
    normalisation_generators(n, k)

Generators of `S_(n-1)^k ⋊ S_k` on the concrete projector letters remaining
after the last outcome has been eliminated.

The generators are adjacent outcome transpositions within each setting and
adjacent setting transpositions, the latter swapping whole blocks of `n-1`
letters.
"""
function normalisation_generators(
    n::Int,
    k::Int,
)::Vector{Vector{Int}}
    n >= 2 || throw(ArgumentError("n must be at least 2"))
    k >= 1 || throw(ArgumentError("k must be positive"))
    nind = n - 1
    nalphabet = k * nind
    generators = Vector{Vector{Int}}()

    for setting in 1:k
        base = (setting - 1) * nind
        for outcome in 1:(nind - 1)
            push!(generators, transposition(
                nalphabet,
                base + outcome,
                base + outcome + 1,
            ))
        end
    end

    for setting in 1:(k - 1)
        permutation = identity_permutation(nalphabet)
        for outcome in 1:nind
            left = (setting - 1) * nind + outcome
            right = setting * nind + outcome
            permutation[left], permutation[right] =
                permutation[right], permutation[left]
        end
        push!(generators, permutation)
    end
    return generators
end

"""
    marginal_generators(n, k; selected_setting=1,
                        selected_outcome=1)

Generators of the stabiliser of one selected marginal.

The selected concrete projector is fixed.  The other explicit outcomes of its
setting are permuted by `S_(n-2)`.  Every non-selected setting has an
independent `S_(n-1)` outcome action, and the non-selected settings are
permuted by `S_(k-1)`.
"""
function marginal_generators(
    n::Int,
    k::Int;
    selected_setting::Int = 1,
    selected_outcome::Int = 1,
)::Vector{Vector{Int}}
    n >= 2 || throw(ArgumentError("n must be at least 2"))
    1 <= selected_setting <= k || throw(ArgumentError(
        "selected_setting must lie in 1:k",
    ))
    nind = n - 1
    1 <= selected_outcome <= nind || throw(ArgumentError(
        "selected_outcome must be one of the explicitly retained outcomes",
    ))

    nalphabet = k * nind
    generators = Vector{Vector{Int}}()

    for setting in 1:k
        base = (setting - 1) * nind
        outcomes = setting == selected_setting ?
            [a for a in 1:nind if a != selected_outcome] : collect(1:nind)
        for position in 1:(length(outcomes) - 1)
            push!(generators, transposition(
                nalphabet,
                base + outcomes[position],
                base + outcomes[position + 1],
            ))
        end
    end

    other_settings = [
        setting for setting in 1:k
        if setting != selected_setting
    ]
    for position in 1:(length(other_settings) - 1)
        left_setting = other_settings[position]
        right_setting = other_settings[position + 1]
        permutation = identity_permutation(nalphabet)
        for outcome in 1:nind
            left = (left_setting - 1) * nind + outcome
            right = (right_setting - 1) * nind + outcome
            permutation[left], permutation[right] =
                permutation[right], permutation[left]
        end
        push!(generators, permutation)
    end
    return generators
end

"""
    act_on_word(word, letter_permutation)

Relabel every letter of `word`.  The permutation is represented by its vector
of images.
"""
act_on_word(word::Word{L}, letter_permutation::AbstractVector{Int}) where {L} =
    Word(ntuple(i -> letter_permutation[word[i]], Val(L)))

"""
    basis_permutations(words, letter_generators)

Convert permutations of the word alphabet into permutations of the indices of
`words`.  An error is raised if the supplied basis is not invariant under a
generator; this catches inconsistent sieves or equation supports early.
"""
function basis_permutations(
    words::AbstractVector{<:Word},
    letter_generators::AbstractVector{<:AbstractVector{Int}},
)::Vector{Vector{Int}}
    index = Dict{Word,Int}(word => i for (i, word) in enumerate(words))
    result = Vector{Vector{Int}}()
    sizehint!(result, length(letter_generators))
    for generator in letter_generators
        permutation = Vector{Int}(undef, length(words))
        for (i, word) in enumerate(words)
            image = act_on_word(word, generator)
            haskey(index, image) || throw(ArgumentError(
                "word basis is not invariant: $(word) maps to missing word $(image)",
            ))
            permutation[i] = index[image]
        end
        push!(result, permutation)
    end
    return result
end

# ---------------------------------------------------------------------------
# Union--find orbit computations
# ---------------------------------------------------------------------------

mutable struct DisjointSet
    parent::Vector{Int}
    rank::Vector{UInt8}
end

DisjointSet(n::Int) = DisjointSet(collect(1:n), zeros(UInt8, n))

function find_root!(sets::DisjointSet, i::Int)::Int
    root = i
    while sets.parent[root] != root
        root = sets.parent[root]
    end
    while sets.parent[i] != i
        next = sets.parent[i]
        sets.parent[i] = root
        i = next
    end
    return root
end

function union_sets!(sets::DisjointSet, i::Int, j::Int)
    root_i = find_root!(sets, i)
    root_j = find_root!(sets, j)
    root_i == root_j && return sets
    if sets.rank[root_i] < sets.rank[root_j]
        root_i, root_j = root_j, root_i
    end
    sets.parent[root_j] = root_i
    if sets.rank[root_i] == sets.rank[root_j]
        sets.rank[root_i] += 0x01
    end
    return sets
end

"""
    orbit_partition(n, generator_permutations)

Return `(representatives, orbit_id)` for the action generated by permutations
of `1:n`.  `orbit_id[i]` is a one-based orbit number.  The representative is
the smallest index in each orbit.
"""
function orbit_partition(
    n::Int,
    generator_permutations::AbstractVector{<:AbstractVector{Int}},
)
    sets = DisjointSet(n)
    for permutation in generator_permutations
        length(permutation) == n || throw(DimensionMismatch(
            "generator has length $(length(permutation)), expected $n",
        ))
        for i in 1:n
            union_sets!(sets, i, permutation[i])
        end
    end

    root_to_members = Dict{Int,Vector{Int}}()
    for i in 1:n
        root = find_root!(sets, i)
        push!(get!(root_to_members, root, Int[]), i)
    end
    orbits = collect(values(root_to_members))
    foreach(sort!, orbits)
    sort!(orbits; by = first)

    representatives = [first(orbit) for orbit in orbits]
    orbit_id = zeros(Int, n)
    for (id, orbit) in enumerate(orbits), i in orbit
        orbit_id[i] = id
    end
    return representatives, orbit_id
end

"""
    word_orbit_partition(words, word_index, letter_generators)

Compute equation-word orbits without materialising one full basis permutation
per generator.  This streaming variant is important for the millions of
normalisation words occurring at `n=8`: it retains only the union--find arrays
and reuses the already available word-to-row dictionary.
"""
function word_orbit_partition(
    words::AbstractVector{<:Word},
    word_index::AbstractDict{Word,Int},
    letter_generators::AbstractVector{<:AbstractVector{Int}},
)
    sets = DisjointSet(length(words))
    for generator in letter_generators
        for (i, word) in enumerate(words)
            image = act_on_word(word, generator)
            j = get(word_index, image, 0)
            j == 0 && throw(ArgumentError(
                "word support is not invariant: $(word) maps to missing $(image)",
            ))
            union_sets!(sets, i, j)
        end
    end

    roots = Vector{Int}(undef, length(words))
    representative_by_root = Dict{Int,Int}()
    for i in eachindex(words)
        root = find_root!(sets, i)
        roots[i] = root
        representative_by_root[root] = min(
            get(representative_by_root, root, i),
            i,
        )
    end
    ordered_roots = sort!(
        collect(keys(representative_by_root));
        by = root -> representative_by_root[root],
    )
    root_to_orbit = Dict(
        root => orbit for (orbit, root) in enumerate(ordered_roots)
    )
    orbit_id = Vector{Int}(undef, length(words))
    for i in eachindex(words)
        orbit_id[i] = root_to_orbit[roots[i]]
    end
    representatives = [representative_by_root[root] for root in ordered_roots]
    return representatives, orbit_id
end

"""Column-major upper-triangular index of `(i,j)`, assuming `1 <= i <= j`."""
@inline symmetric_pair_index(i::Int, j::Int) = j * (j - 1) ÷ 2 + i

"""
    symmetric_pair_orbits(n, generator_permutations)

Find orbits of unordered pairs `{i,j}`.  Such orbits index a basis of the
symmetric commutant of a permutation representation.  Returns
`(orbit_id, number_of_orbits)`, with `orbit_id` stored in upper-triangular
column-major order.
"""
function symmetric_pair_orbits(
    n::Int,
    generator_permutations::AbstractVector{<:AbstractVector{Int}},
)
    npairs = n * (n + 1) ÷ 2
    sets = DisjointSet(npairs)
    for permutation in generator_permutations
        for j in 1:n, i in 1:j
            image_i = permutation[i]
            image_j = permutation[j]
            if image_i > image_j
                image_i, image_j = image_j, image_i
            end
            union_sets!(
                sets,
                symmetric_pair_index(i, j),
                symmetric_pair_index(image_i, image_j),
            )
        end
    end

    roots = [find_root!(sets, i) for i in 1:npairs]
    unique_roots = sort!(unique(roots))
    root_to_id = Dict(root => id for (id, root) in enumerate(unique_roots))
    orbit_id = [root_to_id[root] for root in roots]
    return orbit_id, length(unique_roots)
end

# ---------------------------------------------------------------------------
# Numerical real Wedderburn decomposition
# ---------------------------------------------------------------------------

"""Default relative decomposition tolerance for standard floating types."""
default_decomposition_tolerance(::Type{Float32}) = Float32(2e-4)
default_decomposition_tolerance(::Type{Float64}) = 1e-9
default_decomposition_tolerance(::Type{BigFloat}) = sqrt(sqrt(eps(BigFloat)))
default_decomposition_tolerance(::Type{T}) where {T<:AbstractFloat} =
    sqrt(eps(T))

"""
    decomposition_acceptance_tolerance(T, tolerance)

Tolerance used for intertwiner residuals and final representation checks.
It is deliberately proportional to the decomposition tolerance rather than
its square root, which would discard most of the precision of fixed
multiprecision types such as `Float64x4`.
"""
decomposition_acceptance_tolerance(
    ::Type{T},
    tolerance::T,
) where {T<:AbstractFloat} = max(
    convert(T, 100) * tolerance,
    convert(T, 1_000) * eps(T),
)

"""
    apply_basis_permutation(permutation, matrix)

Apply the permutation matrix `P` satisfying `P*e_i = e_permutation[i]` to the
rows of `matrix`, without explicitly constructing `P`.
"""
function apply_basis_permutation(
    permutation::AbstractVector{Int},
    matrix::AbstractMatrix,
)
    return matrix[invperm(permutation), :]
end

"""Restricted generator matrix `U' P U` on the columns of `U`."""
function restricted_representation(
    permutation::AbstractVector{Int},
    U::AbstractMatrix,
)
    return transpose(U) * apply_basis_permutation(permutation, U)
end

"""
    generic_symmetric_commutant(T, n, basis_permutations; seed)

Construct a deterministic random symmetric matrix in the commutant.  One
integer weight is assigned to each unordered-pair orbit, so invariance is exact
before the integer matrix is converted to `T`.
"""
function generic_symmetric_commutant(
    ::Type{T},
    n::Int,
    generator_permutations::AbstractVector{<:AbstractVector{Int}};
    seed::Int,
) where {T<:AbstractFloat}
    orbit_id, number_of_orbits = symmetric_pair_orbits(
        n,
        generator_permutations,
    )
    rng = MersenneTwister(seed)
    weights = rand(rng, -1_000_000:1_000_000, number_of_orbits)
    matrix = zeros(T, n, n)
    for j in 1:n, i in 1:j
        value = convert(T, weights[orbit_id[symmetric_pair_index(i, j)]])
        matrix[i, j] = value
        matrix[j, i] = value
    end
    return matrix, number_of_orbits
end

"""
    cluster_eigenvalues(values, tolerance)

Cluster sorted eigenvalues using a relative tolerance.  A generic commutant
element has one eigenvalue for every irreducible copy, repeated according to
the irreducible dimension.
"""
function cluster_eigenvalues(
    values::AbstractVector{T},
    tolerance::T,
) where {T<:AbstractFloat}
    isempty(values) && return Vector{Vector{Int}}()
    scale = max(one(T), maximum(abs, values))
    threshold = tolerance * scale
    clusters = Vector{Vector{Int}}()
    current = Int[1]
    reference = values[1]
    for i in 2:length(values)
        if abs(values[i] - reference) <= threshold
            push!(current, i)
        else
            push!(clusters, current)
            current = Int[i]
            reference = values[i]
        end
    end
    push!(clusters, current)
    return clusters
end

"""
    stable_svd(matrix; full=false)

Compute an SVD using an algorithm appropriate for the scalar type.

For BLAS/LAPACK scalars Julia's default divide-and-conquer driver is fast but
can occasionally fail to converge (`gesdd`, LAPACKException).  The
intertwining systems in this module are small, so the more robust QR-iteration
driver (`gesvd`) is preferable.  Generic scalar types continue to use
`GenericLinearAlgebra.svd`.
"""
function stable_svd(
    matrix::AbstractMatrix{T};
    full::Bool = false,
) where {T<:AbstractFloat}
    if T <: LinearAlgebra.BlasReal
        return svd(
            Matrix{T}(matrix);
            full = full,
            alg = LinearAlgebra.QRIteration(),
        )
    end
    return svd(Matrix{T}(matrix); full = full)
end

"""
    intertwiner_data(reference, candidate, basis_permutations, tolerance;
                     max_sweeps=400, seed=2609)

Find an orthogonal intertwiner between two irreducible copies without forming the
Kronecker nullspace system.  For every involutive symmetry generator, the map

    X -> (X + R_candidate(g) * X * R_reference(g)') / 2

is the orthogonal projector onto the corresponding intertwining equation.
Repeated cyclic projections therefore converge to the common intertwiner space.
The storage is `O(r^2)` for an irreducible dimension `r`, instead of `O(g*r^4)`
for the former stacked-SVD construction.

The returned named tuple keeps the historical `nullity` field for compatibility:
it is `1` when a validated real intertwiner is found and `0` otherwise.  The
symmetric-group representations used here are of real type; accidental splitter
collisions are detected later by the commutant-dimension and reconstruction
checks.
"""
function intertwiner_data(
    reference::AbstractMatrix{T},
    candidate::AbstractMatrix{T},
    generator_permutations::AbstractVector{<:AbstractVector{Int}},
    tolerance::T;
    max_sweeps::Int = 400,
    seed::Int = 2609,
) where {T<:AbstractFloat}
    dimension = size(reference, 2)
    size(candidate, 2) == dimension || return (
        nullity = 0,
        intertwiner = nothing,
        residual = convert(T, Inf),
        sweeps = 0,
    )
    max_sweeps >= 1 || throw(ArgumentError("max_sweeps must be positive"))

    if isempty(generator_permutations)
        return (
            nullity = dimension^2,
            intertwiner = Matrix{T}(I, dimension, dimension),
            residual = zero(T),
            sweeps = 0,
        )
    end

    reference_actions = [
        restricted_representation(permutation, reference)
        for permutation in generator_permutations
    ]
    candidate_actions = [
        restricted_representation(permutation, candidate)
        for permutation in generator_permutations
    ]

    # A copy is always intertwined with itself by the identity.  Handling this
    # case explicitly also removes all large self-nullspace calculations.
    if reference === candidate
        identity_d = Matrix{T}(I, dimension, dimension)
        residual = zero(T)
        for (reference_action, candidate_action) in
            zip(reference_actions, candidate_actions)
            residual = max(
                residual,
                opnorm(candidate_action * identity_d -
                       identity_d * reference_action, Inf),
            )
        end
        return (
            nullity = residual <= decomposition_acceptance_tolerance(T, tolerance) ? 1 : 0,
            intertwiner = residual <= decomposition_acceptance_tolerance(T, tolerance) ? identity_d : nothing,
            residual = residual,
            sweeps = 0,
        )
    end

    rng = MersenneTwister(seed)
    intertwiner = T.(rand(rng, -1000:1000, dimension, dimension))
    scale = norm(intertwiner)
    iszero(scale) && (intertwiner[1, 1] = one(T); scale = one(T))
    intertwiner ./= scale

    left_product = similar(intertwiner)
    transformed = similar(intertwiner)
    acceptance_tolerance = decomposition_acceptance_tolerance(T, tolerance)
    half = inv(convert(T, 2))
    residual = convert(T, Inf)
    previous_residual = residual
    performed_sweeps = 0

    for sweep in 1:max_sweeps
        for (reference_action, candidate_action) in
            zip(reference_actions, candidate_actions)
            mul!(left_product, candidate_action, intertwiner)
            mul!(transformed, left_product, transpose(reference_action))
            @. intertwiner = (intertwiner + transformed) * half
        end

        scale = norm(intertwiner)
        if !isfinite(scale) || scale <= eps(T)
            return (
                nullity = 0,
                intertwiner = nothing,
                residual = convert(T, Inf),
                sweeps = sweep,
            )
        end
        intertwiner ./= scale

        residual = zero(T)
        for (reference_action, candidate_action) in
            zip(reference_actions, candidate_actions)
            mul!(left_product, candidate_action, intertwiner)
            mul!(transformed, intertwiner, reference_action)
            @. transformed = left_product - transformed
            residual = max(residual, opnorm(transformed, Inf))
        end
        performed_sweeps = sweep
        residual <= acceptance_tolerance && break

        # Stop once the best approximate intertwiner has stabilised.  A
        # non-equivalent pair then retains a residual well above acceptance.
        if sweep >= 20 && isfinite(previous_residual)
            change = abs(previous_residual - residual)
            change <= tolerance * max(one(T), residual) && break
        end
        previous_residual = residual
    end

    residual <= acceptance_tolerance || return (
        nullity = 0,
        intertwiner = nothing,
        residual = residual,
        sweeps = performed_sweeps,
    )

    # Polar-normalise the projected map.  For equivalent real irreducibles,
    # Schur's lemma makes T'T a scalar multiple of the identity up to numerical
    # error, so this is a small r x r eigenproblem.
    gram = Symmetric(transpose(intertwiner) * intertwiner)
    gram_eigen = eigen(gram)
    minimum(gram_eigen.values) > tolerance^2 || return (
        nullity = 0,
        intertwiner = nothing,
        residual = convert(T, Inf),
        sweeps = performed_sweeps,
    )
    inverse_square_root = gram_eigen.vectors *
        Diagonal(inv.(sqrt.(gram_eigen.values))) *
        transpose(gram_eigen.vectors)
    intertwiner = intertwiner * inverse_square_root

    significant = findfirst(value -> abs(value) > tolerance, vec(intertwiner))
    if !isnothing(significant) && vec(intertwiner)[significant] < zero(T)
        intertwiner .*= -one(T)
    end

    residual = zero(T)
    for (reference_action, candidate_action) in
        zip(reference_actions, candidate_actions)
        mul!(left_product, candidate_action, intertwiner)
        mul!(transformed, intertwiner, reference_action)
        @. transformed = left_product - transformed
        residual = max(residual, opnorm(transformed, Inf))
    end
    return (
        nullity = residual <= acceptance_tolerance ? 1 : 0,
        intertwiner = residual <= acceptance_tolerance ? intertwiner : nothing,
        residual = residual,
        sweeps = performed_sweeps,
    )
end

"""
    validate_decomposition(components, basis_permutations)

Return orthogonality, representation-alignment, and random reconstruction
invariance errors.  This validates the precise convention used by
`reconstruct_matrix`.
"""
function validate_decomposition(
    components::AbstractVector{SymmetryComponent{T}},
    basis_permutations::AbstractVector{<:AbstractVector{Int}};
    seed::Int,
) where {T<:AbstractFloat}
    isempty(components) && return (zero(T), zero(T), zero(T))
    U = hcat((component.isometry for component in components)...)
    identity_n = Matrix{T}(I, size(U, 2), size(U, 2))
    orthogonality_error = opnorm(transpose(U) * U - identity_n, Inf)

    # Check complete transformed generators, including cross-component
    # leakage. Componentwise checks alone can miss a misclassification of two
    # equivalent copies as inequivalent.
    representation_error = zero(T)
    for permutation in basis_permutations
        transformed = transpose(U) * apply_basis_permutation(permutation, U)
        expected = zeros(T, size(transformed))
        offset = 0
        for component in components
            m = component.multiplicity
            r = component.irrep_dimension
            width = m * r
            indices = (offset + 1):(offset + width)
            local_action = transformed[indices, indices]
            reference_action = local_action[1:r, 1:r]
            expected[indices, indices] .=
                kron(Matrix{T}(I, m, m), reference_action)
            offset += width
        end
        representation_error = max(
            representation_error,
            opnorm(transformed - expected, Inf),
        )
    end

    rng = MersenneTwister(seed)
    blocks = Matrix{T}[]
    for component in components
        raw = T.(rand(rng, -20:20, component.multiplicity, component.multiplicity))
        push!(blocks, (raw + transpose(raw)) / convert(T, 2))
    end
    test_matrix = reconstruct_matrix(
        SymmetryDecomposition(
            size(U, 1),
            collect(components),
            [collect(permutation) for permutation in basis_permutations],
            0,
            seed,
            zero(T),
            zero(T),
            zero(T),
            zero(T),
        ),
        blocks,
    )
    invariance_error = zero(T)
    for permutation in basis_permutations
        invariance_error = max(
            invariance_error,
            opnorm(test_matrix[permutation, permutation] - test_matrix, Inf),
        )
    end
    return orthogonality_error, representation_error, invariance_error
end

"""
    decompose_permutation_representation(T, words, letter_generators; kwargs...)

Compute a full real Wedderburn reduction and store its reconstruction
isometries.

The procedure is:

  1. turn the alphabet generators into permutations of the word basis;
  2. form a generic symmetric commutant element from unordered-pair orbits;
  3. diagonalise it; every eigenspace should be one irreducible copy;
  4. identify equivalent copies by solving generator intertwining equations;
  5. orthogonally align equivalent copies and validate the resulting
     `kron(B,I)` convention.

No character table, central idempotent, or isotypic decomposition is used.
The routine retries with new splitters if an accidental eigenvalue collision is
detected.  The real-type assumption is certified by the symmetric-commutant
dimension and full reconstruction checks, without constructing self-intertwiner
nullspace matrices.
"""
function decompose_permutation_representation(
    ::Type{T},
    words::AbstractVector{<:Word},
    letter_generators::AbstractVector{<:AbstractVector{Int}};
    seed::Int = 2609,
    tolerance::Union{Nothing,Real} = nothing,
    max_attempts::Int = 8,
    intertwiner_max_sweeps::Int = 400,
    verbose::Bool = true,
)::SymmetryDecomposition{T} where {T<:AbstractFloat}
    n = length(words)
    n >= 1 || throw(ArgumentError("word basis must be nonempty"))
    tol = isnothing(tolerance) ?
        default_decomposition_tolerance(T) : convert(T, tolerance)
    tol > zero(T) || throw(ArgumentError("tolerance must be positive"))
    acceptance_tol = decomposition_acceptance_tolerance(T, tol)

    generator_permutations = basis_permutations(words, letter_generators)

    # A trivial action leaves the entire PSD cone unreduced but already has the
    # desired real Wedderburn form with one one-dimensional irrep of
    # multiplicity n.
    if isempty(generator_permutations)
        component = SymmetryComponent(
            n,
            1,
            Matrix{T}(I, n, n),
            zero(T),
        )
        return SymmetryDecomposition(
            n,
            [component],
            generator_permutations,
            n * (n + 1) ÷ 2,
            seed,
            tol,
            zero(T),
            zero(T),
            zero(T),
        )
    end

    last_failure = "unknown decomposition failure"
    for attempt in 1:max_attempts
        splitter_seed = seed + attempt - 1
        verbose && println(
            "  symmetry splitter attempt $attempt/$max_attempts, seed=$splitter_seed...",
        )
        splitter, pair_orbit_count = generic_symmetric_commutant(
            T,
            n,
            generator_permutations;
            seed = splitter_seed,
        )
        decomposition = eigen(Symmetric(splitter))
        clusters = cluster_eigenvalues(decomposition.values, tol)
        spaces = [decomposition.vectors[:, cluster] for cluster in clusters]

        # A generic symmetric commutant element splits each real irreducible
        # copy into a separate eigenspace.  We no longer certify this by an
        # r^4 self-intertwiner SVD: the commutant-dimension and full
        # reconstruction checks below detect accidental eigenvalue collisions.

        groups = Vector{Vector{Int}}()
        grouping_failed = false
        for candidate_index in eachindex(spaces)
            candidate = spaces[candidate_index]
            placed = false
            for group in groups
                reference = spaces[first(group)]
                size(reference, 2) == size(candidate, 2) || continue
                data = intertwiner_data(
                    reference,
                    candidate,
                    generator_permutations,
                    tol;
                    max_sweeps = intertwiner_max_sweeps,
                    seed = splitter_seed + 1000 * candidate_index + first(group),
                )
                if data.nullity == 1 && !isnothing(data.intertwiner) &&
                   data.residual <= acceptance_tol
                    push!(group, candidate_index)
                    placed = true
                    break
                elseif data.nullity > 1
                    grouping_failed = true
                    last_failure = "equivalent-copy test had nullity $(data.nullity)"
                    break
                end
            end
            grouping_failed && break
            placed || push!(groups, Int[candidate_index])
        end
        grouping_failed && continue

        components = SymmetryComponent{T}[]
        alignment_failed = false
        for group in groups
            reference = spaces[first(group)]
            copies = Matrix{T}[reference]
            maximum_alignment_error = zero(T)
            for candidate_index in group[2:end]
                candidate = spaces[candidate_index]
                data = intertwiner_data(
                    reference,
                    candidate,
                    generator_permutations,
                    tol;
                    max_sweeps = intertwiner_max_sweeps,
                    seed = splitter_seed + 1000 * candidate_index + first(group),
                )
                if data.nullity != 1 || isnothing(data.intertwiner)
                    alignment_failed = true
                    last_failure = "failed to recover a unique copy intertwiner"
                    break
                end
                push!(copies, candidate * data.intertwiner)
                maximum_alignment_error = max(
                    maximum_alignment_error,
                    data.residual,
                )
            end
            alignment_failed && break
            push!(components, SymmetryComponent(
                length(copies),
                size(reference, 2),
                hcat(copies...),
                maximum_alignment_error,
            ))
        end
        alignment_failed && continue

        sort!(
            components;
            by = component -> (-component.multiplicity,
                               -component.irrep_dimension),
        )
        total_dimension = sum(
            component.multiplicity * component.irrep_dimension
            for component in components
        )
        if total_dimension != n
            last_failure = "components span $total_dimension dimensions, expected $n"
            continue
        end

        # Completeness check for the symmetric commutant. For real-type
        # components R^m tensor W, the symmetric commutant has dimension
        # sum m(m+1)/2. This catches accidentally separated equivalent copies.
        predicted_pair_orbit_count = sum(
            component.multiplicity * (component.multiplicity + 1) ÷ 2
            for component in components
        )
        if predicted_pair_orbit_count != pair_orbit_count
            last_failure = "symmetric-commutant dimension mismatch: decomposition " *
                "predicts $predicted_pair_orbit_count, orbit algebra has " *
                "$pair_orbit_count"
            continue
        end

        orthogonality_error, representation_error, invariance_error =
            validate_decomposition(
                components,
                generator_permutations;
                seed = splitter_seed + 10_000,
            )
        validation_threshold = acceptance_tol
        if max(orthogonality_error, representation_error, invariance_error) >
           validation_threshold
            last_failure = "validation errors (orthogonality=$orthogonality_error, " *
                "representation=$representation_error, invariance=$invariance_error)"
            continue
        end

        verbose && println(
            "  decomposition blocks ",
            [component.multiplicity for component in components],
            " with irrep dimensions ",
            [component.irrep_dimension for component in components],
            ".",
        )
        return SymmetryDecomposition(
            n,
            components,
            generator_permutations,
            pair_orbit_count,
            splitter_seed,
            tol,
            orthogonality_error,
            representation_error,
            invariance_error,
        )
    end

    error(
        "Could not obtain a validated real Wedderburn decomposition after " *
        "$max_attempts attempts: $last_failure.  Increase precision, adjust " *
        "tolerance, or change the seed.",
    )
end

# ---------------------------------------------------------------------------
# Reconstruction and compression
# ---------------------------------------------------------------------------

"""
    reconstruct_matrix(decomposition, blocks)

Reconstruct the original invariant matrix from the reduced PSD blocks.

`blocks[a]` must be a square matrix of size equal to the multiplicity of
`decomposition.components[a]`.  The exact convention is

    X = sum_a U_a * kron(blocks[a], I_(r_a)) * U_a'.

This is the central reconstruction routine that was missing from the original
Python implementation.
"""
function reconstruct_matrix(
    decomposition::SymmetryDecomposition{T},
    blocks::AbstractVector{<:AbstractMatrix},
) where {T<:AbstractFloat}
    length(blocks) == length(decomposition.components) || throw(DimensionMismatch(
        "received $(length(blocks)) blocks for " *
        "$(length(decomposition.components)) components",
    ))
    matrix = zeros(T, decomposition.dimension, decomposition.dimension)
    for (component, block_raw) in zip(decomposition.components, blocks)
        m = component.multiplicity
        r = component.irrep_dimension
        size(block_raw) == (m, m) || throw(DimensionMismatch(
            "block has size $(size(block_raw)), expected ($m,$m)",
        ))
        block = Matrix{T}(block_raw)
        lifted = kron(block, Matrix{T}(I, r, r))
        matrix .+= component.isometry * lifted * transpose(component.isometry)
    end
    return (matrix + transpose(matrix)) / convert(T, 2)
end

"""
    compress_matrix(decomposition, matrix; check_invariant=true)

Project an original matrix back to the reduced blocks.  For component `a`,
the `(i,j)` block is the trace over the irreducible coordinate divided by the
irrep dimension.

The returned named tuple contains `blocks` and `structure_error`, the latter
measuring the distance of `U_a' * matrix * U_a` from
`kron(blocks[a], I_r)`.  For a symmetry-invariant matrix represented with the
stored convention, reconstruction followed by compression is the identity up
to numerical precision.
"""
function compress_matrix(
    decomposition::SymmetryDecomposition{T},
    matrix::AbstractMatrix;
    check_invariant::Bool = true,
) where {T<:AbstractFloat}
    size(matrix) == (decomposition.dimension, decomposition.dimension) ||
        throw(DimensionMismatch("matrix has incompatible size $(size(matrix))"))
    source = Matrix{T}(matrix)
    blocks = Matrix{T}[]
    structure_error = zero(T)

    for component in decomposition.components
        m = component.multiplicity
        r = component.irrep_dimension
        local_matrix = transpose(component.isometry) * source * component.isometry
        block = zeros(T, m, m)
        for j in 1:m, i in 1:m
            rows = ((i - 1) * r + 1):(i * r)
            columns = ((j - 1) * r + 1):(j * r)
            block[i, j] = tr(local_matrix[rows, columns]) / convert(T, r)
        end
        block = (block + transpose(block)) / convert(T, 2)
        push!(blocks, block)
    end
    if check_invariant
        # This also detects cross-component blocks.
        projected = reconstruct_matrix(decomposition, blocks)
        structure_error = opnorm(source - projected, Inf)
    end
    return (blocks = blocks, structure_error = structure_error)
end

"""
    reconstruct_gram_matrices(reduction, Q_blocks, K_blocks)

Return `(Q,K)` in the original word bases from reduced block values.
"""
function reconstruct_gram_matrices(
    reduction::ParentSymmetryReduction,
    Q_blocks::AbstractVector{<:AbstractMatrix},
    K_blocks::AbstractVector{<:AbstractMatrix},
)
    Q = reconstruct_matrix(reduction.Q_decomposition, Q_blocks)
    K = reconstruct_matrix(reduction.K_decomposition, K_blocks)
    return Q, K
end

"""Serialise a reconstruction map and all coefficient/orbit metadata."""
function save_reduction(
    path::AbstractString,
    reduction::ParentSymmetryReduction,
)
    open(path, "w") do io
        serialize(io, reduction)
    end
    return path
end

"""Load a reconstruction map previously written by `save_reduction`."""
function load_reduction(path::AbstractString)
    open(path, "r") do io
        return deserialize(io)
    end
end

# ---------------------------------------------------------------------------
# Parent hierarchy symmetry reduction
# ---------------------------------------------------------------------------

"""
    build_parent_symmetry_reduction(T=Float64; kwargs...)

Build coefficient maps, equation orbits, both Wedderburn decompositions, and
all reconstruction metadata, but do not create a JuMP model.

Important keywords:

  * `n`, `k`, and `t`: hierarchy parameters;
  * `decomposition_tolerance`: relative eigenvalue/SVD tolerance;
  * `seed`: deterministic seed for the generic commutant splitters;
  * `save_reduction_path`: optional path at which the complete reconstruction
    map is immediately serialised.
"""
function build_parent_symmetry_reduction(
    ::Type{T} = Float64;
    n::Int = 4,
    k::Int = 3,
    t::Int = 3,
    selected_setting::Int = 1,
    selected_outcome::Int = 1,
    decomposition_tolerance::Union{Nothing,Real} = nothing,
    seed::Int = 2609,
    max_splitter_attempts::Int = 8,
    intertwiner_max_sweeps::Int = 400,
    verbose::Bool = true,
    save_reduction_path::Union{Nothing,AbstractString} = nothing,
)::ParentSymmetryReduction{T} where {T<:AbstractFloat}
    1 <= t || throw(ArgumentError("this symmetry implementation intentionally supports only t >= 1"))
    k >= 1 || throw(ArgumentError("k must be positive"))

    # The Gram-word bases and symmetry generators are tiny compared with the
    # unreduced coefficient maps.  Decompose them first, so the decomposition
    # workspaces never coexist with the large sparse maps.
    abstract_basis = abstract_words(k, t)
    residual_basis = marginal_residual_basis(
        n,
        k,
        t;
        selected_setting = selected_setting,
        selected_outcome = selected_outcome,
    )
    q_letter_generators = setting_generators(k)
    normalisation_letter_generators = normalisation_generators(n, k)
    marginal_letter_generators = marginal_generators(
        n,
        k;
        selected_setting = selected_setting,
        selected_outcome = selected_outcome,
    )

    verbose && println("Reducing the Q Gram space before coefficient expansion...")
    Q_decomposition = decompose_permutation_representation(
        T,
        abstract_basis,
        q_letter_generators;
        seed = seed + 1,
        tolerance = decomposition_tolerance,
        max_attempts = max_splitter_attempts,
        intertwiner_max_sweeps = intertwiner_max_sweeps,
        verbose = verbose,
    )

    verbose && println("Reducing the K Gram space before coefficient expansion...")
    K_decomposition = decompose_permutation_representation(
        T,
        residual_basis,
        marginal_letter_generators;
        seed = seed + 2,
        tolerance = decomposition_tolerance,
        max_attempts = max_splitter_attempts,
        intertwiner_max_sweeps = intertwiner_max_sweeps,
        verbose = verbose,
    )
    GC.gc()

    verbose && println("Building unreduced universal-parent coefficient maps...")
    data = build_coefficient_data(
        T;
        n = n,
        k = k,
        t = t,
        selected_setting = selected_setting,
        selected_outcome = selected_outcome,
        verbose = verbose,
    )
    data.abstract_basis == abstract_basis || error(
        "internal abstract-basis mismatch after low-memory construction",
    )
    data.residual_basis == residual_basis || error(
        "internal marginal-basis mismatch after low-memory construction",
    )

    verbose && println("Computing coefficient-equation orbits in streaming form...")
    normalisation_representatives, normalisation_orbit_id = word_orbit_partition(
        data.normalisation_words,
        data.normalisation_word_index,
        normalisation_letter_generators,
    )
    marginal_representatives, marginal_orbit_id = word_orbit_partition(
        data.marginal_words,
        data.marginal_word_index,
        marginal_letter_generators,
    )
    # Full equation-basis permutations can require several gigabytes and are not
    # needed once the orbit partition is known.
    normalisation_basis_actions = Vector{Vector{Int}}()
    marginal_basis_actions = Vector{Vector{Int}}()

    reduction = ParentSymmetryReduction(
        data,
        Q_decomposition,
        K_decomposition,
        normalisation_letter_generators,
        marginal_letter_generators,
        normalisation_basis_actions,
        marginal_basis_actions,
        normalisation_representatives,
        normalisation_orbit_id,
        marginal_representatives,
        marginal_orbit_id,
    )

    verbose && println(
        "Symmetry reduction ready: Q blocks=",
        [component.multiplicity for component in Q_decomposition.components],
        ", K blocks=",
        [component.multiplicity for component in K_decomposition.components],
        ", equation orbits=",
        length(normalisation_representatives),
        " normalisation + ",
        length(marginal_representatives),
        " marginal.",
    )

    if !isnothing(save_reduction_path)
        save_reduction(save_reduction_path, reduction)
        verbose && println("Saved reconstruction map to ", save_reduction_path)
    end
    return reduction
end

# ---------------------------------------------------------------------------
# Reducing coefficient rows to Wedderburn blocks
# ---------------------------------------------------------------------------

"""
    reduced_row_blocks(transposed_map, row, decomposition; cleanup_tolerance)

Substitute the stored reconstruction formula into one sparse coefficient row.
The input is the CSC transpose of an original map acting on `vec(X)`.

For a component isometry written as copy blocks `U_1,...,U_m`, the reduced
coefficient is

    D[i,j] = sum_(a,b) C[a,b] * sum_(ell=1)^r
             U_i[a,ell] U_j[b,ell],

where `C` is the original coefficient matrix.  This direct sparse contraction
avoids forming an `N^2 × m^2` reconstruction matrix.
"""
function reduced_row_blocks(
    transposed_map::SparseMatrixCSC{T,Ti},
    row::Int,
    decomposition::SymmetryDecomposition{T};
    cleanup_tolerance::T = zero(T),
) where {T<:AbstractFloat,Ti<:Integer}
    n = decomposition.dimension
    size(transposed_map, 1) == n^2 || throw(DimensionMismatch(
        "coefficient map expects matrices of another dimension",
    ))
    1 <= row <= size(transposed_map, 2) || throw(BoundsError(
        axes(transposed_map, 2),
        row,
    ))

    blocks = [
        zeros(T, component.multiplicity, component.multiplicity)
        for component in decomposition.components
    ]
    column_indices = rowvals(transposed_map)
    coefficients = nonzeros(transposed_map)

    for pointer in nzrange(transposed_map, row)
        matrix_column = column_indices[pointer]
        coefficient = coefficients[pointer]
        i = (matrix_column - 1) % n + 1
        j = (matrix_column - 1) ÷ n + 1

        for (block, component) in zip(blocks, decomposition.components)
            m = component.multiplicity
            r = component.irrep_dimension
            U = component.isometry
            @inbounds for right_copy in 1:m, left_copy in 1:m
                left_offset = (left_copy - 1) * r
                right_offset = (right_copy - 1) * r
                contraction = zero(T)
                for coordinate in 1:r
                    contraction += U[i, left_offset + coordinate] *
                        U[j, right_offset + coordinate]
                end
                block[left_copy, right_copy] += coefficient * contraction
            end
        end
    end

    for block in blocks
        block .= (block + transpose(block)) / convert(T, 2)
        if cleanup_tolerance > zero(T)
            scale = max(one(T), opnorm(block, Inf))
            threshold = cleanup_tolerance * scale
            for index in eachindex(block)
                abs(block[index]) <= threshold && (block[index] = zero(T))
            end
        end
    end
    return blocks
end

"""Add `dot(coefficient, symmetric_variable)` to a JuMP affine expression."""
function add_symmetric_block_dot!(
    expression,
    coefficient::AbstractMatrix{T},
    variable,
) where {T<:Real}
    n = size(coefficient, 1)
    size(coefficient, 2) == n || throw(DimensionMismatch("coefficient is not square"))
    for j in 1:n, i in 1:j
        value = i == j ? coefficient[i, j] : convert(T, 2) * coefficient[i, j]
        iszero(value) || add_to_expression!(expression, value, variable[i, j])
    end
    return expression
end

"""Return true when every numerical entry in a vector of matrices is zero."""
all_zero_blocks(blocks) = all(block -> all(iszero, block), blocks)


"""
    ReducedNormalisationEquation{T}

One retained normalisation equation after quotienting coefficient rows by
symmetry.  `Q_coefficients[b]` is the coefficient matrix paired with the
`b`-th reduced `Q` block, and `rhs` is zero except for the identity-word orbit.
The original representative row and its orbit number are retained for
inspection and reproducibility.
"""
struct ReducedNormalisationEquation{T<:AbstractFloat}
    Q_coefficients::Vector{Matrix{T}}
    rhs::T
    orbit::Int
    row::Int
end

"""
    ReducedMarginalEquation{T}

One retained marginal equation.  In the reduced primal it has the form

    sum_b dot(Q_coefficients[b], Q_blocks[b])
      - sum_b dot(K_coefficients[b], K_blocks[b])
      - eta_coefficient * eta == 0.

`eta_coefficient` is one on the selected-projector orbit and zero otherwise.
"""
struct ReducedMarginalEquation{T<:AbstractFloat}
    Q_coefficients::Vector{Matrix{T}}
    K_coefficients::Vector{Matrix{T}}
    eta_coefficient::T
    orbit::Int
    row::Int
end

"""Format one SDPA coefficient without silently converting its scalar type."""
function _sdpa_number(value::T, digits::Union{Nothing,Int}) where {T<:Real}
    if !isnothing(digits)
        digits >= 1 || throw(ArgumentError("dat_s_digits must be positive"))
        value = round(value; sigdigits = digits)
    end
    return iszero(value) ? "0" : string(value)
end

function _write_sdpa_block!(
    io::IO,
    matrix_number::Int,
    block_number::Int,
    matrix::AbstractMatrix{T},
    digits::Union{Nothing,Int},
) where {T<:Real}
    size(matrix, 1) == size(matrix, 2) ||
        throw(DimensionMismatch("SDPA blocks must be square"))
    for column in axes(matrix, 2), row in firstindex(matrix, 1):column
        value = matrix[row, column]
        iszero(value) && continue
        println(
            io,
            matrix_number,
            ' ',
            block_number,
            ' ',
            row,
            ' ',
            column,
            ' ',
            _sdpa_number(value, digits),
        )
    end
    return nothing
end

"""
    dat_s_block_layout(reduction)

Return the block order used by `write_dat_s`.  The first blocks are the reduced
`Q` blocks, followed by the reduced `K` blocks and two scalar blocks.  In the
SDPA dual matrix solution these final blocks are `eta` and `1-eta`.  The
returned ranges can be applied directly to the matrix blocks before calling
`reconstruct_gram_matrices`.
"""
function dat_s_block_layout(reduction::ParentSymmetryReduction)
    q_sizes = [
        component.multiplicity
        for component in reduction.Q_decomposition.components
    ]
    k_sizes = [
        component.multiplicity
        for component in reduction.K_decomposition.components
    ]
    number_q_blocks = length(q_sizes)
    number_k_blocks = length(k_sizes)
    q_range = 1:number_q_blocks
    k_range = (number_q_blocks + 1):(number_q_blocks + number_k_blocks)
    eta_block = number_q_blocks + number_k_blocks + 1
    eta_slack_block = eta_block + 1
    return (
        Q_blocks = q_range,
        K_blocks = k_range,
        eta_block = eta_block,
        eta_slack_block = eta_slack_block,
        block_sizes = vcat(q_sizes, k_sizes, -1, -1),
    )
end

"""
    write_dat_s(path, reduction, normalisation_equations, marginal_equations;
                digits=nothing)

Write the dual of the reduced primal SDP in sparse SDPA `.dat-s` format.
The SDPA scalar variables are ordered as all normalisation multipliers followed
by all marginal multipliers.  Its block-diagonal LMI is

    sum_i y_i N_i + sum_j z_j M_j >= 0,
    -sum_j z_j H_j >= 0,
    -sum_j tau_j z_j + mu - 1 >= 0,
    mu >= 0,

where the first line contains every reduced `Q` block, the second every reduced
`K` block, and the two scalar blocks are dual to `0 <= eta <= 1`.  The
objective is `min sum_i rhs_i y_i + mu`.

No Float64 file-format model is introduced: coefficients are printed directly
from `T`.  With `T == BigFloat`, omitting `digits` retains Julia's decimal
representation at the active precision; supplying `digits` rounds every token
to that many significant decimal digits.
"""
function write_dat_s(
    path::AbstractString,
    reduction::ParentSymmetryReduction{T},
    normalisation_equations::AbstractVector{<:ReducedNormalisationEquation{T}},
    marginal_equations::AbstractVector{<:ReducedMarginalEquation{T}};
    digits::Union{Nothing,Int} = nothing,
) where {T<:AbstractFloat}
    layout = dat_s_block_layout(reduction)
    q_sizes = [
        component.multiplicity
        for component in reduction.Q_decomposition.components
    ]
    k_sizes = [
        component.multiplicity
        for component in reduction.K_decomposition.components
    ]
    # The last two scalar blocks encode respectively
    #
    #     -sum_j tau_j z_j + mu - 1 >= 0,
    #     mu >= 0,
    #
    # where `mu` is the dual multiplier of the primal bound `eta <= 1`.
    block_sizes = layout.block_sizes
    eta_block = layout.eta_block
    eta_slack_block = layout.eta_slack_block
    number_equation_variables =
        length(normalisation_equations) + length(marginal_equations)
    upper_multiplier_variable = number_equation_variables + 1
    number_variables = upper_multiplier_variable
    objective = vcat(
        [equation.rhs for equation in normalisation_equations],
        zeros(T, length(marginal_equations)),
        one(T),
    )

    open(path, "w") do io
        println(io, "* Symmetry-reduced universal-parent SOS dual")
        println(io, "* n=$(reduction.data.n), k=$(reduction.data.k), t=$(reduction.data.t)")
        println(io, number_variables)
        println(io, length(block_sizes))
        println(io, "{", join(block_sizes, ", "), "}")
        println(
            io,
            "{",
            join((_sdpa_number(value, digits) for value in objective), ", "),
            "}",
        )

        # F_0 and the dual variable `mu` for the primal upper bound eta <= 1.
        println(
            io,
            "0 ",
            eta_block,
            " 1 1 ",
            _sdpa_number(one(T), digits),
        )
        println(
            io,
            upper_multiplier_variable,
            ' ',
            eta_block,
            " 1 1 ",
            _sdpa_number(one(T), digits),
        )
        println(
            io,
            upper_multiplier_variable,
            ' ',
            eta_slack_block,
            " 1 1 ",
            _sdpa_number(one(T), digits),
        )

        for (variable, equation) in enumerate(normalisation_equations)
            for (block, coefficient) in enumerate(equation.Q_coefficients)
                _write_sdpa_block!(io, variable, block, coefficient, digits)
            end
        end

        marginal_offset = length(normalisation_equations)
        k_block_offset = length(q_sizes)
        for (local_index, equation) in enumerate(marginal_equations)
            variable = marginal_offset + local_index
            for (block, coefficient) in enumerate(equation.Q_coefficients)
                _write_sdpa_block!(io, variable, block, coefficient, digits)
            end
            for (block, coefficient) in enumerate(equation.K_coefficients)
                _write_sdpa_block!(
                    io,
                    variable,
                    k_block_offset + block,
                    -coefficient,
                    digits,
                )
            end
            if !iszero(equation.eta_coefficient)
                println(
                    io,
                    variable,
                    ' ',
                    eta_block,
                    " 1 1 ",
                    _sdpa_number(-equation.eta_coefficient, digits),
                )
            end
        end
    end
    return path
end

"""Convenience method exporting the coefficient data stored by a built model."""
function write_dat_s(
    path::AbstractString,
    problem;
    digits::Union{Nothing,Int} = nothing,
)
    return write_dat_s(
        path,
        problem.reduction,
        problem.normalisation_equations,
        problem.marginal_equations;
        digits = digits,
    )
end

"""Largest recorded numerical error in a stored symmetry reduction."""
function decomposition_error_scale(
    reduction::ParentSymmetryReduction{T},
) where {T<:AbstractFloat}
    decompositions = (
        reduction.Q_decomposition,
        reduction.K_decomposition,
    )
    value = zero(T)
    for decomposition in decompositions
        value = max(
            value,
            decomposition.tolerance,
            decomposition.orthogonality_error,
            decomposition.representation_error,
            decomposition.invariance_test_error,
        )
        for component in decomposition.components
            value = max(value, component.alignment_error)
        end
    end
    return value
end

"""
    recommended_hypatia_rank_tolerance(reduction)

Return a QR rank tolerance commensurate with the numerical symmetry reduction.

Hypatia's default `init_tol_qr = 1000eps(T)` assumes that the affine data were
formed directly at precision `T`.  Here they are obtained after numerical
eigenvector and intertwiner calculations, so fixed multiprecision types can
contain harmless equation noise many orders of magnitude above `eps(T)`.
"""
function recommended_hypatia_rank_tolerance(
    reduction::ParentSymmetryReduction{T},
) where {T<:AbstractFloat}
    return max(
        convert(T, 100) * decomposition_error_scale(reduction),
        convert(T, 1_000) * eps(T),
    )
end

function _has_raw_optimizer_attribute(attributes, name::AbstractString)
    return any(attributes) do attribute
        key = first(attribute)
        key isa AbstractString && key == name
    end
end

"""
    default_dat_s_path(n, k, t)

Return the default sparse-SDPA filename used by `build_symmetrized_parent_sdp` when `dat_s_path` is omitted.
"""
default_dat_s_path(n::Int, k::Int, t::Int) = "parent_k$(k)_n$(n)_t$(t).dat-s"

"""
    build_symmetrized_parent_sdp(T=Float64; reduction=nothing, kwargs...)

Build the reduced coefficient equations, export the geometric dual to sparse SDPA format, and read that file back into JuMP.

If `dat_s_path` is omitted, the file is written to `default_dat_s_path(n, k, t)`. The file is always written. The keyword
`hypatia_rank_tolerance` is retained for explicit experiments, but no precision-dependent value is selected automatically.
"""
function build_symmetrized_parent_sdp(
    ::Type{T} = Float64;
    reduction::Union{Nothing,ParentSymmetryReduction{T}} = nothing,
    n::Int = 4,
    k::Int = 3,
    t::Int = 3,
    selected_setting::Int = 1,
    selected_outcome::Int = 1,
    decomposition_tolerance::Union{Nothing,Real} = nothing,
    coefficient_cleanup_tolerance::Union{Nothing,Real} = nothing,
    enforce_all_equation_rows::Bool = false,
    hypatia_rank_tolerance::Union{Nothing,Real} = nothing,
    seed::Int = 2609,
    max_splitter_attempts::Int = 8,
    intertwiner_max_sweeps::Int = 400,
    save_reduction_path::Union{Nothing,AbstractString} = nothing,
    optimizer = Hypatia.Optimizer{T},
    optimizer_attributes = Pair[],
    silent::Bool = false,
    verbose::Bool = true,
    dat_s_path::Union{Nothing,AbstractString} = nothing,
    dat_s_digits::Union{Nothing,Int} = nothing,
    retain_unreduced_data::Bool = false,
) where {T<:AbstractFloat}
    !isnothing(reduction) && is_compact_reduction(reduction) && throw(ArgumentError(
        "a compact reconstruction map cannot rebuild the reduced equations; " *
        "supply a full reduction or omit the reduction keyword",
    ))

    actual_reduction = isnothing(reduction) ? build_parent_symmetry_reduction(
        T;
        n = n,
        k = k,
        t = t,
        selected_setting = selected_setting,
        selected_outcome = selected_outcome,
        decomposition_tolerance = decomposition_tolerance,
        seed = seed,
        max_splitter_attempts = max_splitter_attempts,
        intertwiner_max_sweeps = intertwiner_max_sweeps,
        verbose = verbose,
        save_reduction_path = nothing,
    ) : reduction

    data = actual_reduction.data
    cleanup = isnothing(coefficient_cleanup_tolerance) ? zero(T) : convert(T, coefficient_cleanup_tolerance)
    using_hypatia = optimizer === Hypatia.Optimizer{T}
    if !using_hypatia && !isnothing(hypatia_rank_tolerance)
        throw(ArgumentError("hypatia_rank_tolerance is only valid with Hypatia.Optimizer{T}"))
    end
    rank_tolerance = isnothing(hypatia_rank_tolerance) ? nothing : convert(T, hypatia_rank_tolerance)
    if !isnothing(rank_tolerance)
        rank_tolerance > zero(T) || throw(ArgumentError("hypatia_rank_tolerance must be positive"))
    end

    normalisation_equations = ReducedNormalisationEquation{T}[]
    marginal_equations = ReducedMarginalEquation{T}[]

    empty_row = data.normalisation_word_index[Word()]
    empty_orbit = actual_reduction.normalisation_orbit_id[empty_row]
    normalisation_rows = enforce_all_equation_rows ? collect(eachindex(data.normalisation_words)) :
        actual_reduction.normalisation_orbit_representatives

    verbose && println("Reducing normalisation equations...")
    An_transpose = sparse(transpose(data.An))
    for row in normalisation_rows
        orbit = actual_reduction.normalisation_orbit_id[row]
        coefficients = reduced_row_blocks(
            An_transpose,
            row,
            actual_reduction.Q_decomposition;
            cleanup_tolerance = cleanup,
        )
        rhs = orbit == empty_orbit ? one(T) : zero(T)
        all_zero_blocks(coefficients) && iszero(rhs) && continue
        push!(normalisation_equations, ReducedNormalisationEquation(coefficients, rhs, orbit, row))
    end
    An_transpose = nothing
    GC.gc()

    selected_row = data.marginal_word_index[Word(data.selected_letter)]
    selected_orbit = actual_reduction.marginal_orbit_id[selected_row]
    marginal_rows = enforce_all_equation_rows ? collect(eachindex(data.marginal_words)) :
        actual_reduction.marginal_orbit_representatives

    verbose && println("Reducing the Q side of the marginal equations...")
    marginal_Q_coefficients = Vector{Vector{Matrix{T}}}(undef, length(marginal_rows))
    Am_transpose = sparse(transpose(data.Am))
    for (index, row) in enumerate(marginal_rows)
        marginal_Q_coefficients[index] = reduced_row_blocks(
            Am_transpose,
            row,
            actual_reduction.Q_decomposition;
            cleanup_tolerance = cleanup,
        )
    end
    Am_transpose = nothing
    GC.gc()

    verbose && println("Reducing the K side of the marginal equations...")
    Ah_transpose = sparse(transpose(data.Ah))
    for (index, row) in enumerate(marginal_rows)
        orbit = actual_reduction.marginal_orbit_id[row]
        Q_coefficients = marginal_Q_coefficients[index]
        K_coefficients = reduced_row_blocks(
            Ah_transpose,
            row,
            actual_reduction.K_decomposition;
            cleanup_tolerance = cleanup,
        )
        is_target = orbit == selected_orbit
        all_zero_blocks(Q_coefficients) && all_zero_blocks(K_coefficients) && !is_target && continue
        push!(
            marginal_equations,
            ReducedMarginalEquation(Q_coefficients, K_coefficients, is_target ? one(T) : zero(T), orbit, row),
        )
    end
    Ah_transpose = nothing
    marginal_Q_coefficients = nothing
    GC.gc()

    returned_reduction = retain_unreduced_data ? actual_reduction : compact_reduction(actual_reduction)
    resolved_dat_s_path = isnothing(dat_s_path) ? default_dat_s_path(data.n, data.k, data.t) : String(dat_s_path)
    write_dat_s(
        resolved_dat_s_path,
        returned_reduction,
        normalisation_equations,
        marginal_equations;
        digits = dat_s_digits,
    )
    verbose && println("Exported reduced SDP to ", resolved_dat_s_path)

    if !isnothing(save_reduction_path)
        save_reduction(save_reduction_path, returned_reduction)
        verbose && println(
            "Saved ",
            retain_unreduced_data ? "full reduction" : "compact reconstruction map",
            " to ",
            save_reduction_path,
        )
    end

    if !retain_unreduced_data
        actual_reduction = returned_reduction
        data = nothing
        GC.gc()
    end

    model = read_from_file(resolved_dat_s_path; coefficient_type = T)
    set_optimizer(model, optimizer)
    silent && set_silent(model)

    if !isnothing(rank_tolerance)
        if !_has_raw_optimizer_attribute(optimizer_attributes, "init_tol_qr")
            set_optimizer_attribute(model, "init_tol_qr", rank_tolerance)
        end
        if !_has_raw_optimizer_attribute(optimizer_attributes, "tol_inconsistent")
            set_optimizer_attribute(model, "tol_inconsistent", rank_tolerance)
        end
        verbose && println("Using explicit Hypatia equality-rank tolerance ", rank_tolerance, ".")
    end
    for attribute in optimizer_attributes
        set_optimizer_attribute(model, first(attribute), last(attribute))
    end

    verbose && println(
        "SDPA model ready: ",
        length(normalisation_equations),
        " normalisation equations, ",
        length(marginal_equations),
        " marginal equations, Q blocks ",
        [component.multiplicity for component in returned_reduction.Q_decomposition.components],
        ", K blocks ",
        [component.multiplicity for component in returned_reduction.K_decomposition.components],
        ".",
    )

    return (
        model = model,
        reduction = returned_reduction,
        coefficient_cleanup_tolerance = cleanup,
        enforce_all_equation_rows = enforce_all_equation_rows,
        hypatia_rank_tolerance = rank_tolerance,
        number_normalisation_constraints = length(normalisation_equations),
        number_marginal_constraints = length(marginal_equations),
        normalisation_equations = normalisation_equations,
        marginal_equations = marginal_equations,
        dat_s_path = resolved_dat_s_path,
        dat_s_layout = dat_s_block_layout(returned_reduction),
        save_reduction_path = save_reduction_path,
        retain_unreduced_data = retain_unreduced_data,
        solver_formulation = :sdpa_geometric_dual,
    )
end

"""Convert an upper-triangular MOI vector into a dense symmetric matrix."""
function symmetric_matrix_from_triangle(values::AbstractVector, side_dimension::Int, ::Type{T}) where {T<:AbstractFloat}
    expected_length = side_dimension * (side_dimension + 1) ÷ 2
    length(values) == expected_length || throw(DimensionMismatch(
        "triangle vector has length $(length(values)), expected $expected_length for side dimension $side_dimension",
    ))
    matrix = zeros(T, side_dimension, side_dimension)
    index = 0
    for column in 1:side_dimension, row in 1:column
        index += 1
        value = convert(T, values[index])
        matrix[row, column] = value
        matrix[column, row] = value
    end
    return matrix
end

"""
    extract_sdpa_standard_solution(problem)

Extract the original standard-form variables `(Q_blocks, K_blocks, eta, eta_slack)` from the dual solution of the
geometric SDPA model.
"""
function extract_sdpa_standard_solution(problem)
    model = problem.model
    T = value_type(typeof(model))
    moi_model = backend(model)
    psd_indices = MOI.get(
        moi_model,
        MOI.ListOfConstraintIndices{MOI.VectorAffineFunction{T},MOI.PositiveSemidefiniteConeTriangle}(),
    )
    nonnegative_indices = MOI.get(
        moi_model,
        MOI.ListOfConstraintIndices{MOI.VectorAffineFunction{T},MOI.Nonnegatives}(),
    )

    q_sizes = [component.multiplicity for component in problem.reduction.Q_decomposition.components]
    k_sizes = [component.multiplicity for component in problem.reduction.K_decomposition.components]
    expected_psd_blocks = length(q_sizes) + length(k_sizes)
    length(psd_indices) == expected_psd_blocks || error(
        "the SDPA reader returned $(length(psd_indices)) PSD blocks, expected $expected_psd_blocks",
    )
    length(nonnegative_indices) == 2 || error(
        "the SDPA reader returned $(length(nonnegative_indices)) nonnegative blocks, expected the eta and 1-eta blocks",
    )

    psd_duals = [MOI.get(moi_model, MOI.ConstraintDual(), index) for index in psd_indices]
    q_count = length(q_sizes)
    Q_blocks = [symmetric_matrix_from_triangle(psd_duals[index], q_sizes[index], T) for index in eachindex(q_sizes)]
    K_blocks = [
        symmetric_matrix_from_triangle(psd_duals[q_count + index], k_sizes[index], T) for index in eachindex(k_sizes)
    ]

    eta_dual = MOI.get(moi_model, MOI.ConstraintDual(), nonnegative_indices[1])
    eta_slack_dual = MOI.get(moi_model, MOI.ConstraintDual(), nonnegative_indices[2])
    length(eta_dual) == 1 || error("the eta SDPA block is not scalar")
    length(eta_slack_dual) == 1 || error("the eta-slack SDPA block is not scalar")

    return (
        Q_blocks = Q_blocks,
        K_blocks = K_blocks,
        eta = convert(T, eta_dual[1]),
        eta_slack = convert(T, eta_slack_dual[1]),
    )
end

"""Evaluate one stored reduced coefficient row on numerical block values."""
function reduced_block_pairing(
    coefficients::AbstractVector{<:AbstractMatrix{T}},
    blocks::AbstractVector{<:AbstractMatrix},
) where {T<:AbstractFloat}
    length(coefficients) == length(blocks) || throw(DimensionMismatch(
        "coefficient and value block counts differ",
    ))
    value = zero(T)
    for (coefficient, block) in zip(coefficients, blocks)
        size(coefficient) == size(block) || throw(DimensionMismatch(
            "coefficient and value block sizes differ",
        ))
        @inbounds for index in eachindex(coefficient, block)
            value += coefficient[index] * convert(T, block[index])
        end
    end
    return value
end

"""Residuals of the retained symmetry-representative equations."""
function reduced_equation_residuals(
    normalisation_equations::AbstractVector{<:ReducedNormalisationEquation{T}},
    marginal_equations::AbstractVector{<:ReducedMarginalEquation{T}},
    Q_blocks::AbstractVector{<:AbstractMatrix},
    K_blocks::AbstractVector{<:AbstractMatrix},
    eta::T,
) where {T<:AbstractFloat}
    normalisation = T[
        reduced_block_pairing(equation.Q_coefficients, Q_blocks) - equation.rhs
        for equation in normalisation_equations
    ]
    marginal = T[
        reduced_block_pairing(equation.Q_coefficients, Q_blocks) -
        reduced_block_pairing(equation.K_coefficients, K_blocks) -
        equation.eta_coefficient * eta
        for equation in marginal_equations
    ]
    return (normalisation = normalisation, marginal = marginal)
end

"""Compress `scale * e_index * e_index'` without allocating the full matrix."""
function compress_basis_projector(
    decomposition::SymmetryDecomposition{T},
    index::Int,
    scale::T,
) where {T<:AbstractFloat}
    1 <= index <= decomposition.dimension || throw(BoundsError(
        1:decomposition.dimension,
        index,
    ))
    blocks = Matrix{T}[]
    for component in decomposition.components
        m = component.multiplicity
        r = component.irrep_dimension
        U = component.isometry
        block = zeros(T, m, m)
        for right_copy in 1:m, left_copy in 1:m
            left_offset = (left_copy - 1) * r
            right_offset = (right_copy - 1) * r
            entry = zero(T)
            @inbounds for coordinate in 1:r
                entry += U[index, left_offset + coordinate] *
                    U[index, right_offset + coordinate]
            end
            block[left_copy, right_copy] = scale * entry / convert(T, r)
        end
        push!(blocks, (block + transpose(block)) / convert(T, 2))
    end
    return blocks
end

"""
    zero_visibility_feasibility_diagnostics(reduction,
        normalisation_equations, marginal_equations)

Low-memory variant operating only on the compact reconstruction map and retained
reduced equations.  It is the method used by the solver path after the large
unreduced coefficient maps have been released.
"""
function zero_visibility_feasibility_diagnostics(
    reduction::ParentSymmetryReduction{T},
    normalisation_equations::AbstractVector{<:ReducedNormalisationEquation{T}},
    marginal_equations::AbstractVector{<:ReducedMarginalEquation{T}},
) where {T<:AbstractFloat}
    data = reduction.data
    Q_blocks = compress_basis_projector(
        reduction.Q_decomposition,
        1,
        inv(convert(T, data.n)^data.k),
    )
    K_blocks = compress_basis_projector(
        reduction.K_decomposition,
        1,
        inv(convert(T, data.n)),
    )
    residuals = reduced_equation_residuals(
        normalisation_equations,
        marginal_equations,
        Q_blocks,
        K_blocks,
        zero(T),
    )
    max_norm = isempty(residuals.normalisation) ? zero(T) :
        maximum(abs, residuals.normalisation)
    max_marg = isempty(residuals.marginal) ? zero(T) :
        maximum(abs, residuals.marginal)
    return (
        eta = zero(T),
        Q_blocks = Q_blocks,
        K_blocks = K_blocks,
        exact_maximum_normalisation_residual = zero(T),
        exact_maximum_marginal_residual = zero(T),
        projected_maximum_normalisation_residual = max_norm,
        projected_maximum_marginal_residual = max_marg,
        Q_projection_error = convert(T, NaN),
        K_projection_error = convert(T, NaN),
    )
end

"""
    zero_visibility_feasibility_diagnostics(reduction)

Construct the universal feasible point at `eta = 0` and measure how accurately
it survives the numerical Wedderburn coordinates.

The unreduced point is

    Q[empty, empty] = n^(-k),
    K[empty, empty] = n^(-1),

with every other entry zero. It represents the constant parent
`G_a = I/n^k`; every marginal is `I/n`, so the residual at `eta = 0` is the
constant SOS `I/n`. Consequently, an `INFEASIBLE` status is necessarily
numerical rather than mathematical.

The returned diagnostics distinguish residuals of the exact unreduced point
from residuals after compression and reconstruction through the stored
isometries.
"""
function zero_visibility_feasibility_diagnostics(
    reduction::ParentSymmetryReduction{T},
) where {T<:AbstractFloat}
    data = reduction.data
    qdim = length(data.abstract_basis)
    kdim = length(data.residual_basis)

    Q_exact = zeros(T, qdim, qdim)
    K_exact = zeros(T, kdim, kdim)
    q_empty = findfirst(==(Word()), data.abstract_basis)
    k_empty = findfirst(==(Word()), data.residual_basis)
    isnothing(q_empty) && error("abstract basis does not contain the empty word")
    isnothing(k_empty) && error("marginal basis does not contain the empty word")
    Q_exact[q_empty, q_empty] = inv(convert(T, data.n)^data.k)
    K_exact[k_empty, k_empty] = inv(convert(T, data.n))

    normalisation_rhs = zeros(T, length(data.normalisation_words))
    normalisation_rhs[data.normalisation_word_index[Word()]] = one(T)

    exact_normalisation_residual =
        data.An * vec(Q_exact) - normalisation_rhs
    exact_marginal_residual =
        data.Am * vec(Q_exact) - data.Ah * vec(K_exact)

    Q_compressed = compress_matrix(
        reduction.Q_decomposition,
        Q_exact;
        check_invariant = true,
    )
    K_compressed = compress_matrix(
        reduction.K_decomposition,
        K_exact;
        check_invariant = true,
    )
    Q_projected, K_projected = reconstruct_gram_matrices(
        reduction,
        Q_compressed.blocks,
        K_compressed.blocks,
    )
    projected_normalisation_residual =
        data.An * vec(Q_projected) - normalisation_rhs
    projected_marginal_residual =
        data.Am * vec(Q_projected) - data.Ah * vec(K_projected)

    return (
        eta = zero(T),
        Q_blocks = Q_compressed.blocks,
        K_blocks = K_compressed.blocks,
        exact_maximum_normalisation_residual =
            maximum(abs, exact_normalisation_residual),
        exact_maximum_marginal_residual =
            maximum(abs, exact_marginal_residual),
        projected_maximum_normalisation_residual =
            maximum(abs, projected_normalisation_residual),
        projected_maximum_marginal_residual =
            maximum(abs, projected_marginal_residual),
        Q_projection_error = Q_compressed.structure_error,
        K_projection_error = K_compressed.structure_error,
    )
end

"""
    solve_symmetrized_parent_sdp(T=Float64; kwargs...)

Build, export, reload, and solve the symmetry-reduced hierarchy. The solver model is the geometric-form dual read from
the generated `.dat-s` file. The original reduced Gram blocks and `eta` are recovered from the conic dual solution.
"""
function solve_symmetrized_parent_sdp(
    ::Type{T} = Float64;
    save_path::Union{Nothing,AbstractString} = nothing,
    reference_eta = nothing,
    kwargs...,
) where {T<:AbstractFloat}
    problem = build_symmetrized_parent_sdp(T; kwargs...)
    GC.gc()
    optimize!(problem.model)
    println(solution_summary(problem.model))

    standard_status = dual_status(problem.model)
    if !has_duals(problem.model) || !(standard_status in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT))
        witness = zero_visibility_feasibility_diagnostics(
            problem.reduction,
            problem.normalisation_equations,
            problem.marginal_equations,
        )
        @warn(
            "The geometric SDPA model did not return a feasible dual point, so the original Gram matrices " *
            "cannot be recovered. " *
            "The eta=0 analytic point confirms feasibility of the original standard-form SDP.",
            termination_status = termination_status(problem.model),
            geometric_primal_status = primal_status(problem.model),
            standard_primal_status = standard_status,
            projected_normalisation_residual = witness.projected_maximum_normalisation_residual,
            projected_marginal_residual = witness.projected_maximum_marginal_residual,
        )
        return merge(problem, (result = nothing, zero_visibility_witness = witness))
    end

    extracted = extract_sdpa_standard_solution(problem)
    Q_block_values = [(block + transpose(block)) / convert(T, 2) for block in extracted.Q_blocks]
    K_block_values = [(block + transpose(block)) / convert(T, 2) for block in extracted.K_blocks]
    eta_value = extracted.eta
    eta_slack_value = extracted.eta_slack

    Q, K = reconstruct_gram_matrices(problem.reduction, Q_block_values, K_block_values)
    data = problem.reduction.data
    reduced_residuals = reduced_equation_residuals(
        problem.normalisation_equations,
        problem.marginal_equations,
        Q_block_values,
        K_block_values,
        eta_value,
    )
    normalisation_residual = reduced_residuals.normalisation
    marginal_residual = reduced_residuals.marginal

    Q_compression = compress_matrix(problem.reduction.Q_decomposition, Q)
    K_compression = compress_matrix(problem.reduction.K_decomposition, K)
    Q_roundtrip = maximum(
        isempty(Q_block_values) ? T[zero(T)] : [
            opnorm(a - b, Inf) for (a, b) in zip(Q_block_values, Q_compression.blocks)
        ],
    )
    K_roundtrip = maximum(
        isempty(K_block_values) ? T[zero(T)] : [
            opnorm(a - b, Inf) for (a, b) in zip(K_block_values, K_compression.blocks)
        ],
    )

    minimum_eigenvalue_Q_blocks = [eigmin(block) for block in Q_block_values]
    minimum_eigenvalue_K_blocks = [eigmin(block) for block in K_block_values]
    minimum_eigenvalue_Q = minimum(minimum_eigenvalue_Q_blocks)
    minimum_eigenvalue_K = minimum(minimum_eigenvalue_K_blocks)
    geometric_objective = convert(T, objective_value(problem.model))

    diagnostics = (
        eta = eta_value,
        eta_slack = eta_slack_value,
        eta_plus_slack_error = eta_value + eta_slack_value - one(T),
        geometric_objective = geometric_objective,
        geometric_objective_minus_eta = geometric_objective - eta_value,
        minimum_eigenvalue_Q = minimum_eigenvalue_Q,
        minimum_eigenvalue_K = minimum_eigenvalue_K,
        minimum_eigenvalue_Q_blocks = minimum_eigenvalue_Q_blocks,
        minimum_eigenvalue_K_blocks = minimum_eigenvalue_K_blocks,
        maximum_normalisation_residual = isempty(normalisation_residual) ? zero(T) : maximum(abs, normalisation_residual),
        maximum_marginal_residual = isempty(marginal_residual) ? zero(T) : maximum(abs, marginal_residual),
        Q_structure_error = Q_compression.structure_error,
        K_structure_error = K_compression.structure_error,
        Q_block_roundtrip_error = Q_roundtrip,
        K_block_roundtrip_error = K_roundtrip,
        termination_status = termination_status(problem.model),
        geometric_primal_status = primal_status(problem.model),
        standard_primal_status = dual_status(problem.model),
    )

    println("eta                         = ", diagnostics.eta)
    println("eta slack                   = ", diagnostics.eta_slack)
    println("eta + slack - 1             = ", diagnostics.eta_plus_slack_error)
    println("SDPA objective - eta        = ", diagnostics.geometric_objective_minus_eta)
    if !isnothing(reference_eta)
        reference = convert(T, reference_eta)
        println("reference eta               = ", reference)
        println("reference - eta             = ", reference - eta_value)
    end
    println("minimum eigenvalue of Q     = ", diagnostics.minimum_eigenvalue_Q)
    println("minimum eigenvalue of K     = ", diagnostics.minimum_eigenvalue_K)
    println("max normalisation residual  = ", diagnostics.maximum_normalisation_residual)
    println("max marginal residual       = ", diagnostics.maximum_marginal_residual)
    println("Q reconstruction roundtrip  = ", diagnostics.Q_block_roundtrip_error)
    println("K reconstruction roundtrip  = ", diagnostics.K_block_roundtrip_error)

    result = (
        eta = eta_value,
        eta_slack = eta_slack_value,
        Q_blocks = Q_block_values,
        K_blocks = K_block_values,
        Q = Q,
        K = K,
        diagnostics = diagnostics,
        Q_block_signature = block_signature(problem.reduction.Q_decomposition),
        K_block_signature = block_signature(problem.reduction.K_decomposition),
        n = data.n,
        k = data.k,
        t = data.t,
    )

    if !isnothing(save_path)
        payload = (reduction = problem.reduction, result = result)
        open(save_path, "w") do io
            serialize(io, payload)
        end
        println("Saved reduced blocks and reconstruction map to ", save_path)
    end
    return merge(problem, (result = result,))
end

"""
    self_test(; exhaustive=false, verbose=true)

Run combinatorial, decomposition, and reconstruction checks without invoking an
SDP solver.  The default test verifies the length-parametric `Word` operations,
the exact marginal-basis counts, and a small reconstruction round trip.

With `exhaustive=true`, the test additionally reproduces the exact
`(n,k,t)=(4,3,3)` block structures

    Q: 7, 5, 3,
    K: 20, 14, 12, 7, 7, 2, 1.

The former unsieved 388-word basis is deliberately not constructed anywhere in
this module.
"""
function self_test(; exhaustive::Bool = false, verbose::Bool = true)
    @assert Word() isa Word{0}
    @assert Word(1, 2, 3) isa Word{3}
    @assert append_letter(Word(1, 2), 3) isa Word{3}
    @assert concatenate(Word(1, 2), Word(3, 4)) isa Word{4}
    @assert reverse(Word(1, 2, 3)) == Word(3, 2, 1)
    @assert Dict(Word(1, 2) => 7)[Word(1, 2)] == 7

    # Exact sieve sizes quoted in the numerical experiments.
    @assert length(marginal_residual_basis(3, 3, 2)) == 22
    @assert length(marginal_residual_basis(4, 3, 2)) == 38
    @assert length(marginal_residual_basis(3, 3, 3)) == 74
    @assert length(marginal_residual_basis(4, 3, 3)) == 170

    T = Float64
    q_words = abstract_words(3, 2)
    q_decomposition = decompose_permutation_representation(
        T,
        q_words,
        setting_generators(3);
        seed = 31415,
        verbose = verbose,
    )
    @assert sum(
        component.multiplicity * component.irrep_dimension
        for component in q_decomposition.components
    ) == length(q_words)

    rng = MersenneTwister(2718)
    blocks = Matrix{T}[]
    for component in q_decomposition.components
        raw = randn(rng, component.multiplicity, component.multiplicity)
        push!(blocks, raw * transpose(raw))
    end
    matrix = reconstruct_matrix(q_decomposition, blocks)
    compressed = compress_matrix(q_decomposition, matrix)
    @assert compressed.structure_error < 1e-6
    @assert maximum(opnorm(a - b, Inf) for (a, b) in zip(blocks, compressed.blocks)) < 1e-6

    if exhaustive
        q_words = abstract_words(3, 3)
        q = decompose_permutation_representation(
            T,
            q_words,
            setting_generators(3);
            seed = 1234,
            verbose = verbose,
        )
        @assert [component.multiplicity for component in q.components] == [7, 5, 3]
        @assert [component.irrep_dimension for component in q.components] == [2, 1, 1]

        k_words = marginal_residual_basis(4, 3, 3)
        @assert length(k_words) == 170
        k_decomposition = decompose_permutation_representation(
            T,
            k_words,
            marginal_generators(4, 3);
            seed = 9012,
            verbose = verbose,
        )
        @assert [component.multiplicity for component in k_decomposition.components] ==
            [20, 14, 12, 7, 7, 2, 1]
        @assert [component.irrep_dimension for component in k_decomposition.components] ==
            [4, 1, 1, 4, 4, 2, 4]
    end

    println("All SymmetrizedParentSOS self-tests passed.")
    return true
end


end # module SymmetrizedParentSOS
