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
    ws = MessagePassingRulesApproximations.workspace(method, f, μs, Σs)
    y = f(μs...)
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

# `(μ, Σ, C)` into the scratch, in place; the arrays are the scratch's.
scratch_statistics!(s::GaussianScratch{W, Nothing, Nothing, Nothing}, method, f, μs, Σs) where {W} =
    MessagePassingRulesApproximations.forward_statistics!(s.workspace, method, f, μs, Σs)
scratch_statistics!(s::GaussianScratch{W, Nothing, Nothing}, method, f, μs, Σs) where {W} =
    MessagePassingRulesApproximations.forward_statistics!(s.C, s.workspace, method, f, μs, Σs)
scratch_statistics!(s::GaussianScratch, method, f, μs, Σs) =
    MessagePassingRulesApproximations.forward_statistics!(s.m, s.V, s.C, s.workspace, method, f, μs, Σs)

# The normal approximation of `f` of the inputs, as `approximate_normal` gives it, through the
# scratch: no allocation but the message's own arrays.
function approximate_normal!(scratch::GaussianScratch, method, f::F, distributions::Tuple) where {F}
    μs, Σs = input_moments(distributions)
    μ, Σ, _ = scratch_statistics!(scratch, method, f, μs, Σs)
    return normal_message(μ, Σ)
end

normal_message(μ::Real, Σ::Real) = NormalMeanVariance(μ, Σ)
normal_message(μ::AbstractVector, Σ::AbstractMatrix) = MvNormalMeanCovariance(copy(μ), copy(Σ))
