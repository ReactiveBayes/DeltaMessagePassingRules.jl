# Gauss–Laguerre quadrature, for a node with one input whose messages are gamma or exponential.
# Towards `out`, the normal with the pushforward's mean and covariance; the joint over the input,
# the input's message reweighted by the likelihood of the message from `out`, in log space, and
# moment-matched to a gamma; towards the input, that gamma divided by the input's message.

const GammaInput = Union{GammaDistributionsFamily, Exponential}

gamma_parameters(d::GammaDistributionsFamily) = (shape(d), scale(d))
gamma_parameters(d::Exponential) = (one(scale(d)), scale(d))

# Gauss–Laguerre is univariate: the node must have one input.
function only_input(inputs::Tuple)
    length(inputs) == 1 || throw(ArgumentError(lazy"`GaussLaguerreQuadrature` takes a Delta node with one input, not $(length(inputs))"))
    return only(inputs)
end

@define_message_update_rule(
    node = DeltaFn, target = :out, algorithm = DeltaApproximation{<:GaussLaguerreQuadrature}, ctx = (:node,),
    args = (m[:in...]::GammaInput,),
    body = (algo, ctx, args) -> begin
        f = getnodefn(ctx.node, Target(:out))
        μ, Σ = MessagePassingRulesApproximations.approximate(algo.method, f, gamma_parameters(only_input(args.m[:in]))...)
        convert(promote_variate_type(typeof(μ), NormalMeanVariance), μ, Σ)
    end,
)

@define_marginal_update_rule(
    node = DeltaFn, target = (:in,), algorithm = DeltaApproximation{<:GaussLaguerreQuadrature}, ctx = (:node,),
    args = (m[:out]::Any, m[:in...]::GammaInput),
    body = (algo, ctx, args) -> begin
        f, m_out = getnodefn(ctx.node, Target(:out)), args.m[:out]
        mean, var = approximate_meancov(algo.method, LogIntegrand(x -> logpdf(m_out, f(x))), gamma_parameters(only_input(args.m[:in]))...)
        FactorizedJoint((GammaShapeScale(mean^2 / var, var / mean),))
    end,
)

# A gamma divided by another: shapes subtract, less one, and so do rates. The result may be
# improper.
@define_message_update_rule(
    node = DeltaFn, target = (:in, k), algorithm = DeltaApproximation{<:GaussLaguerreQuadrature, Nothing},
    args = (m[:in][k]::GammaInput, q[(:in,)]::FactorizedJoint),
    body = (args) -> begin
        q, m = component(args.q[(:in,)], k), args.m[:in][k]
        GammaShapeRate(shape(q) - shape(m) + 1, rate(q) - rate(m))
    end,
)
