# The natural-gradient targets in closed form: loaded with ClosedFormExpectations, whose gradient
# of an expectation, `∇ E_q[ℓ] = E_q[ℓ(x) ∇ log q(x)]` (the Williams product), gives the target,
# for an integrand `ℓ` from `closed_form_integrand`.
module DeltaMessagePassingRulesClosedFormExpectationsExt

using DeltaMessagePassingRules, ExponentialFamily, BayesBase, Distributions
using DeltaMessagePassingRules: PulledBack
using MessagePassingRulesBase: NodeFunctionLogPdf
using ClosedFormExpectations: ClosedFormExpectation, LogGammaKernel

# A Gamma message of shape `a` and rate `b` pulled back through `λ = eᶻ` is `(a - 1) z - b eᶻ`, the
# kernel `LogGammaKernel(1 / b, a - 1)`, for any shape: a Poisson count of zero gives `a = 1`
function DeltaMessagePassingRules.closed_form_integrand(
        message::NodeFunctionLogPdf{<:PulledBack{typeof(exp), <:GammaDistributionsFamily}}
    )
    outbound = message.logpdf.outbound
    return LogGammaKernel(scale(outbound), shape(outbound) - 1)
end

# At a Gaussian, by Bonnet's and Price's theorems: with `(∂μ, ∂σ)` the gradient in the mean and
# standard deviation, `E[ℓ''] = ∂σ / σ` and `E[ℓ'] = ∂μ`, so that `Λ⋆ = -∂σ / σ` and
# `ξ⋆ = ∂μ + Λ⋆ m`
function DeltaMessagePassingRules.closed_form_target(ℓ, q::UnivariateNormalDistributionsFamily)
    m, v = mean_var(q)
    normal = Normal(float(m), sqrt(float(v)))
    BayesBase.supports_expectation_gradient(ClosedFormExpectation(), ℓ, normal) || return nothing
    ∂μ, ∂σ = BayesBase.expectation_gradient(ClosedFormExpectation(), ℓ, normal)
    Λ = -∂σ / std(normal)
    return [∂μ + Λ * mean(normal), -Λ / 2]
end

# At a Gamma, the definition: the gradient in natural parameters is `Cov_q[T, ℓ]`
function DeltaMessagePassingRules.closed_form_target(ℓ, q::GammaDistributionsFamily)
    ef = convert(ExponentialFamilyDistribution, q)
    BayesBase.supports_expectation_gradient(ClosedFormExpectation(), ℓ, ef) || return nothing
    return fisherinformation(ef) \ collect(BayesBase.expectation_gradient(ClosedFormExpectation(), ℓ, ef))
end

end
