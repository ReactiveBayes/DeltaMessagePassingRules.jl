# The generalized unscented transform towards `out`, and towards an input through a known inverse:
# its sigma points match the third and fourth central moments of each input, so the inputs'
# messages may be of any univariate family with four moments, or normal. The result is the normal
# with the pushforward's mean and covariance.

const MomentApproximation = Union{GenUnscented, GenConstrainedUnscented}
const MomentInput = Union{UnivariateDistribution, MultivariateNormalDistributionsFamily}

# The mean, covariance and components' third and fourth central moments of a message:
# Distributions' skewness and kurtosis are standardised.
function central_moments(d::UnivariateDistribution)
    m, v = mean(d), var(d)
    return (m, v, skewness(d) * v^(3 / 2), kurtosis(d, false) * v^2)
end

function central_moments(d::UnivariateNormalDistributionsFamily)
    m, v = mean_var(d)
    return (m, v, zero(v), 3 * v^2)
end

function central_moments(d::MultivariateNormalDistributionsFamily)
    m, Σ = mean_cov(d)
    return (m, Σ, zeros(eltype(Σ), length(m)), 3 .* diag(Σ) .^ 2)
end

function approximate_moments(method::MomentApproximation, f::F, distributions::Tuple) where {F}
    moments = map(central_moments, distributions)
    μ, Σ = MessagePassingRulesApproximations.approximate(method, f, map(m -> m[1], moments), map(m -> m[2], moments), map(m -> m[3], moments), map(m -> m[4], moments))
    return convert(promote_variate_type(typeof(μ), NormalMeanVariance), μ, Σ)
end

@define_message_update_rule(
    node = DeltaFn, target = :out, algorithm = DeltaApproximation{<:MomentApproximation}, ctx = (:node,),
    args = (m[:in...]::MomentInput,),
    body = (algo, ctx, args) -> approximate_moments(algo.method, getnodefn(ctx.node, Target(:out)), args.m[:in]),
)

@define_message_update_rule(
    node = DeltaFn, target = (:in, k), algorithm = DeltaApproximation{<:MomentApproximation, <:Union{Function, Tuple{Vararg{Function}}}},
    args = (m[:out]::MomentInput, m[:in][!k]::MomentInput),
    body = (algo, args) -> approximate_moments(algo.method, inverse_towards(algo, k), (args.m[:out], filter(!isnothing, args.m[:in])...)),
)
