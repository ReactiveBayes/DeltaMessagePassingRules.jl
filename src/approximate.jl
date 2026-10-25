"""
    approximate_normal(method, f, distributions::Tuple)

The normal approximation of `f(x₁, …, xₙ)` for independent normal inputs `distributions`, by
`method`, one of the Gaussian methods: a normal with the pushforward's mean and covariance,
univariate or multivariate as the output of `f` is. The message rules towards `out`, and towards
an input through a known inverse, are this call.
"""
function approximate_normal end

# The Gaussian methods: their inputs are normal messages, given by their means and covariances.
const GaussianMethod = Union{Unscented, Linearization, GaussHermiteCubature, GenUnscented, GenConstrainedUnscented}

function approximate_normal(method::GaussianMethod, f::F, distributions::NTuple{N, NormalDistributionsFamily}) where {F, N}
    statistics = mean_cov.(distributions)
    μ, Σ = MessagePassingRulesApproximations.approximate(method, f, first.(statistics), last.(statistics))
    return convert(promote_variate_type(typeof(μ), NormalMeanVariance), μ, Σ)
end

"""
    forward_statistics(method, f, μs, Σs, μ_in, Σ_in) -> (μ, Σ, C)

The mean `μ` and covariance `Σ` of `f` of the inputs, and the cross-covariance `C` of the inputs
with it, by `method`, for inputs with means `μs` and covariances `Σs` (tuples, one entry per
input) whose joint has mean `μ_in` and covariance `Σ_in`. The rule for the joint over the inputs
smooths these with the message from `out`, by
[`smoothRTS`](@extref MessagePassingRulesApproximations.smoothRTS).
"""
function forward_statistics end

forward_statistics(method::GaussianMethod, f::F, μs, Σs, μ_in, Σ_in) where {F} =
    MessagePassingRulesApproximations.forward_statistics(method, f, μs, Σs)

## The scratch of the Gaussian rules

# A Gaussian rule's working memory: the method's workspace for the inputs' moments, and the outputs
# of `forward_statistics!`, `nothing` where they are scalars. Built once per stream, it calls `f`
# once, at the inputs' means, for the size of the output; the rule's message copies the outputs.
struct GaussianScratch{W, M, V, C}
    workspace::W
    m::M
    V::V
    C::C
end

function input_moments(distributions::Tuple)
    statistics = map(mean_cov, distributions)
    return map(first, statistics), map(last, statistics)
end

function gaussian_scratch(method, f::F, distributions::Tuple) where {F}
    μs, Σs = input_moments(distributions)
    y = f(μs...)
    # One scalar input and a scalar output: the allocating forms allocate nothing already, their
    # few points on the stack, so the scratch holds nothing.
    (y isa Real && length(μs) == 1 && first(μs) isa Real) && return GaussianScratch(nothing, nothing, nothing, nothing)
    ws = MessagePassingRulesApproximations.workspace(method, f, μs, Σs)
    d_in = sum(length, μs)
    T = float(promote_type(eltype(y), map(eltype, μs)..., map(eltype, Σs)...))
    if y isa Real
        cross = (length(μs) == 1 && first(μs) isa Real) ? nothing : Vector{T}(undef, d_in)
        return GaussianScratch(ws, nothing, nothing, cross)
    else
        n = length(y)
        return GaussianScratch(ws, Vector{T}(undef, n), Matrix{T}(undef, n, n), Matrix{T}(undef, d_in, n))
    end
end

# `(μ, Σ, C)` into the scratch, in place; the arrays are the scratch's. A scratch that holds
# nothing, for scalars, leaves them to the allocating form, which allocates nothing there. The
# branches are on types, so each call compiles to one of them.
const MPRA = MessagePassingRulesApproximations

function scratch_statistics!(s::GaussianScratch, method, f::F, μs, Σs) where {F}
    s.workspace === nothing && return MPRA.forward_statistics(method, f, μs, Σs)
    s.m === nothing || return MPRA.forward_statistics!(s.m, s.V, s.C, s.workspace, method, f, μs, Σs)
    s.C === nothing && return MPRA.forward_statistics!(s.workspace, method, f, μs, Σs)
    return MPRA.forward_statistics!(s.C, s.workspace, method, f, μs, Σs)
end

# The mean and covariance of `f` of the inputs, without the cross-covariance, which a message
# towards `out` does not need; a linearisation computes them with it.
function scratch_moments!(s::GaussianScratch, method, f::F, μs, Σs) where {F}
    s.workspace === nothing && return MPRA.approximate(method, f, μs, Σs)
    method isa Linearization && return Base.front(scratch_statistics!(s, method, f, μs, Σs))
    s.m === nothing && return MPRA.approximate!(s.workspace, method, f, μs, Σs)
    return MPRA.approximate!(s.m, s.V, s.workspace, method, f, μs, Σs)
end

# The normal approximation of `f` of the inputs, as `approximate_normal` gives it, through the
# scratch: no allocation but the message's own arrays.
function approximate_normal!(scratch::GaussianScratch, method, f::F, distributions::Tuple) where {F}
    μs, Σs = input_moments(distributions)
    μ, Σ = scratch_moments!(scratch, method, f, μs, Σs)
    return normal_message(μ, Σ)
end

normal_message(μ::Real, Σ::Real) = NormalMeanVariance(μ, Σ)
normal_message(μ::AbstractVector, Σ::AbstractMatrix) = MvNormalMeanCovariance(copy(μ), copy(Σ))
