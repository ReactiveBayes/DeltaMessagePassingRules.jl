# `ExpectationRuleFallback`: where no rule fits, the message of a stochastic node is its expected
# log-density, `x ↦ E_q[log f(x, ·)]` over every other input, through the context's expectation
# backend.

@testmodule UniversalNodes begin
    using MessagePassingRulesBase, BayesBase, Distributions
    using MessagePassingRulesBase: RuleArgs, RuleContext, Target, IndexedTarget, ClusterTarget, call_fallback
    import ExpectationApproximations as EA

    # `out ~ Toy(a, b)`: a Gaussian log-density in `out` with mean `a` and variance `b`,
    # unnormalised, with no rules
    struct Toy{A, B}
        a::A
        b::B
    end
    BayesBase.logpdf(d::Toy, x) = -(x - d.a)^2 / (2 * d.b)
    @define_factor_node(node = Toy, type = Stochastic, interfaces = [:out, :a, :b])

    # `out ~ Shape(α, θ)`: a Gamma distribution, a node whose `out` has a support
    struct Shape{A, T} <: ContinuousUnivariateDistribution
        α::A
        θ::T
    end
    Distributions.logpdf(d::Shape, x::Real) = logpdf(Gamma(d.α, d.θ), x)
    Base.minimum(::Shape) = 0.0
    Base.maximum(::Shape) = Inf
    @define_factor_node(node = Shape, type = Stochastic, interfaces = [:out, :α, :θ])

    # `out ~ Unbounded(m)`: a distribution that does not say its support
    struct Unbounded{M} <: ContinuousUnivariateDistribution
        m::M
    end
    Distributions.logpdf(d::Unbounded, x::Real) = -(x - d.m)^2 / 2
    @define_factor_node(node = Unbounded, type = Stochastic, interfaces = [:out, :m])

    # `out ~ Mix(z, means...)`: a unit-variance Gaussian around the `z`-th mean; a node with
    # groups opts in to the fallbacks by defining its node function
    struct Mix{Z, M}
        z::Z
        means::M
    end
    BayesBase.logpdf(d::Mix, x) = -(x - d.means[d.z])^2 / 2
    @define_factor_node(node = Mix, type = Stochastic, interfaces = [:out, :z, :means...])
    MessagePassingRulesBase.nodefunction(::Type{<:Mix}) = (; out, z, means) -> logpdf(Mix(z, means), out)

    context() = RuleContext(expectation = EA.EADefault())
    message(node, target, args; ctx = context()) = call_fallback(DeltaMessagePassingRules.ExpectationRuleFallback(), node, target, nothing, ctx, args)

    import DeltaMessagePassingRules
end

@testitem "ExpectationRuleFallback: towards a parameter, point masses plugged in" tags = [:universal] setup = [UniversalNodes] begin
    using MessagePassingRulesBase, BayesBase
    using MessagePassingRulesBase: RuleArgs, Target, unwrap_fallback
    N = UniversalNodes

    # With every other input a point mass the message is the node's log-density itself, exact:
    # an unnormalised density whose known log scale is zero.
    answer = N.message(N.Toy, Target(:a), RuleArgs(q = (out = PointMass(2.0), b = PointMass(0.5))))
    message, logscale = unwrap_fallback(answer)
    @test logpdf(message, 1.0) ≈ logpdf(N.Toy(1.0, 0.5), 2.0)
    @test logscale isa UndefinedLogScale && logscale.cause === :unnormalised_product && logscale.detail == 0.0
end

@testitem "ExpectationRuleFallback: towards a parameter, the expectation over the other inputs" tags = [:universal] setup = [UniversalNodes] begin
    using MessagePassingRulesBase, BayesBase, ExponentialFamily
    using MessagePassingRulesBase: RuleArgs, Target, unwrap_fallback
    N = UniversalNodes

    # E[-(out - a)² / 2b] over out ~ N(2, 3) is -((2 - a)² + 3) / 2b: quadrature is exact for it,
    # and the variance of out, which the plug-in fallback loses, is in the message.
    message, logscale = unwrap_fallback(N.message(N.Toy, Target(:a), RuleArgs(m = (out = NormalMeanVariance(2.0, 3.0),), q = (b = PointMass(0.5),))))
    for a in (-1.0, 1.0, 2.5)
        @test logpdf(message, a) ≈ -((2 - a)^2 + 3) / (2 * 0.5)
    end
    # The message is the expected log-density, not the exact message: its log scale is unknown.
    @test logscale isa UndefinedLogScale && logscale.cause === :fallback
    # Evaluating it allocates nothing: the expectation is prepared once, with the message. On
    # Julia 1.10 the node's keyword function itself allocates 16 bytes a call.
    logpdf(message, 1.0)
    if VERSION >= v"1.11"
        @test (@allocated logpdf(message, 1.0)) == 0
    end
end

@testitem "ExpectationRuleFallback: towards out, the support of the node's distribution" tags = [:universal] setup = [UniversalNodes] begin
    using MessagePassingRulesBase, BayesBase, Distributions
    using MessagePassingRulesBase: RuleArgs, Target, unwrap_fallback
    import ExpectationApproximations as EA
    N = UniversalNodes

    message, _ = unwrap_fallback(N.message(N.Shape, Target(:out), RuleArgs(q = (α = PointMass(2.0), θ = PointMass(1.5)))))
    @test logpdf(message, 0.7) ≈ logpdf(Gamma(2.0, 1.5), 0.7)
    # A projection reads the domain from it, with no prior on `out`.
    @test EA.domain(message) === EA.Positive()
    # A node that is no distribution has no support to give: the domain is left to the product.
    toy, _ = unwrap_fallback(N.message(N.Toy, Target(:out), RuleArgs(q = (a = PointMass(2.0), b = PointMass(0.5)))))
    @test EA.domain(toy) === nothing
    @test logpdf(toy, 1.0) ≈ logpdf(N.Toy(2.0, 0.5), 1.0)
    # Nor does a distribution that does not say its support.
    unbounded, _ = unwrap_fallback(N.message(N.Unbounded, Target(:out), RuleArgs(q = (m = PointMass(1.0),))))
    @test EA.domain(unbounded) === nothing && logpdf(unbounded, 0.5) ≈ -0.125
end

@testitem "ExpectationRuleFallback: a categorical input is summed exactly" tags = [:universal] setup = [UniversalNodes] begin
    using MessagePassingRulesBase, BayesBase, Distributions
    using MessagePassingRulesBase: RuleArgs, Target, IndexedTarget, unwrap_fallback
    N = UniversalNodes

    # Towards out: Σ_k p_k log N(x; m_k, 1), the expectation over z of the log-density.
    p = [0.3, 0.7]
    means = (PointMass(-1.0), PointMass(2.0))
    message, _ = unwrap_fallback(N.message(N.Mix, Target(:out), RuleArgs(q = (z = Categorical(p), means = means))))
    @test logpdf(message, 0.5) ≈ p[1] * logpdf(N.Mix(1, (-1.0, 2.0)), 0.5) + p[2] * logpdf(N.Mix(2, (-1.0, 2.0)), 0.5)
    # Towards a member of the group: the other members and z as they are.
    member, _ = unwrap_fallback(N.message(N.Mix, IndexedTarget(:means, 2), RuleArgs(q = (out = PointMass(0.5), z = Categorical(p), means = (PointMass(-1.0), nothing)))))
    @test logpdf(member, 2.0) ≈ p[1] * logpdf(N.Mix(1, (-1.0, 2.0)), 0.5) + p[2] * logpdf(N.Mix(2, (-1.0, 2.0)), 0.5)
end

@testitem "ExpectationRuleFallback: where it has nothing to say, or no backend" tags = [:universal] setup = [UniversalNodes] begin
    using MessagePassingRulesBase, BayesBase, ExponentialFamily
    using MessagePassingRulesBase: RuleArgs, RuleContext, Target
    N = UniversalNodes

    # An interface with no input: the log-density cannot be evaluated.
    @test N.message(N.Toy, Target(:a), RuleArgs(q = (out = PointMass(2.0),))) === nothing
    # A joint input: not split into its members yet.
    joint = MessagePassingRulesBase.Marginals(NamedTuple(), Val(((:out, :b),)), (PointMass([2.0, 0.5]),))
    @test N.message(N.Toy, Target(:a), RuleArgs(MessagePassingRulesBase.Messages(NamedTuple()), joint)) === nothing
    # Without an expectation backend it says where one is given.
    @test_throws "expectation" N.message(N.Toy, Target(:a), RuleArgs(q = (out = PointMass(2.0), b = PointMass(0.5))); ctx = RuleContext())
end
