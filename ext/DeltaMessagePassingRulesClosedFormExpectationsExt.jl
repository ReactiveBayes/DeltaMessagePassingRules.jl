# The natural-gradient targets in closed form: loaded with ClosedFormExpectations, whose Williams
# product `E_q[log p(x) ∇ log q(x)]` is `∇ E_q[log p]`, from which the target follows.
module DeltaMessagePassingRulesClosedFormExpectationsExt

using DeltaMessagePassingRules, ExponentialFamily, BayesBase, Distributions
using DeltaMessagePassingRules: PulledBack
using MessagePassingRulesBase: NodeFunctionLogPdf
using ClosedFormExpectations: ClosedFormExpectations, ClosedWilliamsProduct, Logpdf, LogGamma, has_closed_form

# A Gamma message of shape `a` and rate `b` pulled back through `λ = eᶻ`, `(a - 1) z - b eᶻ`, is the
# log-density of a `LogGamma(1 / b, a - 1)` up to a constant, which need not be normalisable
function DeltaMessagePassingRules.closed_form_integrand(
        message::NodeFunctionLogPdf{<:PulledBack{typeof(exp), <:GammaDistributionsFamily}}
    )
    outbound = message.logpdf.outbound
    return LogGamma(scale(outbound), shape(outbound) - 1; check_args = false)
end

# At a Gaussian, by Bonnet's and Price's theorems: with `(∂μ, ∂σ)` the Williams product in the mean
# and standard deviation, `E[ℓ''] = ∂σ / σ` and `E[ℓ'] = ∂μ`, so that `Λ⋆ = -∂σ / σ` and
# `ξ⋆ = ∂μ + Λ⋆ m`
function DeltaMessagePassingRules.closed_form_target(p::Distribution, q::UnivariateNormalDistributionsFamily)
    m, v = mean_var(q)
    normal = Normal(float(m), sqrt(float(v)))
    has_closed_form(ClosedWilliamsProduct(), Logpdf(p), normal) || return nothing
    ∂μ, ∂σ = ClosedFormExpectations.mean(ClosedWilliamsProduct(), Logpdf(p), normal)
    Λ = -∂σ / std(normal)
    return [∂μ + Λ * mean(normal), -Λ / 2]
end

# At a Gamma, the definition: the Williams product in natural parameters is `Cov_q[T, ℓ]`
function DeltaMessagePassingRules.closed_form_target(p::Distribution, q::GammaDistributionsFamily)
    ef = convert(ExponentialFamilyDistribution, q)
    has_closed_form(ClosedWilliamsProduct(), Logpdf(p), ef) || return nothing
    return fisherinformation(ef) \ collect(ClosedFormExpectations.mean(ClosedWilliamsProduct(), Logpdf(p), ef))
end

end
