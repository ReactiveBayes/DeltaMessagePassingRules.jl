# `ExpectationRuleFallback` on a deterministic node, the Delta node `out = f(in...)` without an
# approximation method: towards `out` the moment-matched pushforward of the inputs' messages, and
# towards an input the message on `out` pulled back through `f`.

@testmodule DeltaContext begin
    using MessagePassingRulesBase: MessagePassingRulesBase, Target, RuleContext, call_fallback
    import ExpectationApproximations as EA
    import DeltaMessagePassingRules

    struct FunctionNode{F}
        f::F
    end
    MessagePassingRulesBase.getnodefn(node::FunctionNode, ::Target{:out}) = node.f

    context(f) = RuleContext(node = FunctionNode(f), expectation = EA.EADefault())
    message(f, target, args) = call_fallback(DeltaMessagePassingRules.ExpectationRuleFallback(), DeltaMessagePassingRules.DeltaFn{typeof(f)}, target, nothing, context(f), args)
end

@testitem "ExpectationRuleFallback: a Delta node towards out" tags = [:universal] setup = [DeltaContext] begin
    using MessagePassingRulesBase, BayesBase, ExponentialFamily
    using MessagePassingRulesBase: RuleArgs, Target, unwrap_fallback
    import ExpectationApproximations as EA
    D = DeltaContext

    # A linear function: the pushforward of Gaussians is Gaussian, and its moments exact.
    linear(x, y) = 2x + 3y + 1
    message, logscale = unwrap_fallback(D.message(linear, Target(:out), RuleArgs(m = (in = (NormalMeanVariance(1.0, 0.5), NormalMeanVariance(-1.0, 2.0)),))))
    @test message isa NormalMeanVariance
    @test mean(message) ≈ 0.0 atol = 1.0e-12
    @test var(message) ≈ 4 * 0.5 + 9 * 2.0
    @test logscale isa UndefinedLogScale
    # A nonlinear one: the moments of the pushforward, as the backend computes them.
    message, _ = unwrap_fallback(D.message(tanh, Target(:out), RuleArgs(m = (in = (NormalMeanVariance(0.3, 0.2),),))))
    m, v = EA.mean_cov(EA.EADefault(), tanh, NormalMeanVariance(0.3, 0.2))
    @test mean(message) ≈ m && var(message) ≈ v
    # Point masses in, a point mass out: exact, normalised.
    message, logscale = unwrap_fallback(D.message(linear, Target(:out), RuleArgs(m = (in = (PointMass(1.0), PointMass(2.0)),))))
    @test message === PointMass(9.0) && logscale === 0.0
end

@testitem "ExpectationRuleFallback: a Delta node towards an input" tags = [:universal] setup = [DeltaContext] begin
    using MessagePassingRulesBase, BayesBase, ExponentialFamily
    using MessagePassingRulesBase: RuleArgs, IndexedTarget, unwrap_fallback
    D = DeltaContext

    # A single input: the message on out pulled back, exact, its log scale that of the message
    # on out up to the normaliser.
    outbound = NormalMeanVariance(0.5, 0.1)
    args = RuleArgs(m = (out = outbound, in = (nothing,)), logscale = (out = -0.7,))
    message, logscale = unwrap_fallback(D.message(tanh, IndexedTarget(:in, 1), args))
    @test logpdf(message, 0.4) ≈ logpdf(outbound, tanh(0.4))
    @test logscale isa UndefinedLogScale && logscale.cause === :unnormalised_product && logscale.detail == -0.7
    # Where log scales are not tracked, none.
    message, logscale = unwrap_fallback(D.message(tanh, IndexedTarget(:in, 1), RuleArgs(m = (out = outbound, in = (nothing,)))))
    @test logscale.cause === :fallback

    # Two inputs: in expectation over the other one. For a linear function,
    # E_y[log N(2x + 3y + 1; 0, 1)] over y ~ N(-1, 2) is -((2x - 2)² + 18)/2 - log(2π)/2.
    linear(x, y) = 2x + 3y + 1
    args = RuleArgs(m = (out = NormalMeanVariance(0.0, 1.0), in = (nothing, NormalMeanVariance(-1.0, 2.0))))
    message, logscale = unwrap_fallback(D.message(linear, IndexedTarget(:in, 1), args))
    for x in (-1.0, 0.5, 2.0)
        @test logpdf(message, x) ≈ -((2x - 2)^2 + 18) / 2 - log(2π) / 2
    end
    @test logscale.cause === :fallback
end

@testitem "ExpectationRuleFallback: a deterministic node without its node instance" tags = [:universal] setup = [DeltaContext] begin
    using MessagePassingRulesBase, BayesBase, ExponentialFamily
    using MessagePassingRulesBase: RuleArgs, RuleContext, Target, call_fallback
    import ExpectationApproximations as EA
    import DeltaMessagePassingRules
    # The function belongs to the engine's node object; without it there is no message.
    @test call_fallback(DeltaMessagePassingRules.ExpectationRuleFallback(), DeltaMessagePassingRules.DeltaFn{typeof(tanh)}, Target(:out), nothing, RuleContext(expectation = EA.EADefault()), RuleArgs(m = (in = (NormalMeanVariance(0.0, 1.0),),))) === nothing
end
