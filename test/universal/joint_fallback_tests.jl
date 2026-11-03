# `ExpectationRuleFallback` for joint marginals and average energies: a cluster's joint is the
# joint Laplace approximation of the node's log-density times its members' messages, in
# expectation over the other clusters; an average energy is -E_q[log f] under the clusters'
# marginals, joints included.

@testitem "ExpectationRuleFallback: the joint of a stochastic node's cluster" tags = [:universal] setup = [UniversalNodes] begin
    using MessagePassingRulesBase, BayesBase, Distributions, ExponentialFamily
    using MessagePassingRulesBase: RuleArgs, ClusterTarget
    import ExpectationApproximations as EA
    N = UniversalNodes

    # N(out; a, b) N(out; 1, 0.5) N(a; 0, 2) with b = 0.5 is Gaussian, so Laplace is exact.
    args = RuleArgs(m = (out = NormalMeanVariance(1.0, 0.5), a = NormalMeanVariance(0.0, 2.0)), q = (b = PointMass(0.5),))
    joint, outcome = MessagePassingRulesBase.unwrap_outcome(N.message(N.Toy, ClusterTarget((:out, :a)), args))
    b = 0.5
    Λ = [1 / b + 2 -1 / b; -1 / b 1 / b + 1 / 2]
    @test joint isa MvNormalMeanCovariance
    @test mean(joint) ≈ Λ \ [2.0, 0.0]
    @test cov(joint) ≈ inv(Λ)
    # how the joint Laplace went, for the engine's events
    @test outcome.method === :joint_laplace && outcome.converged && outcome.iterations > 0

    # Members on the half-line: the Laplace approximation in log coordinates, as by hand.
    args = RuleArgs(m = (out = Gamma(3.0, 1.0), θ = Gamma(2.0, 2.0)), q = (α = PointMass(2.0),))
    joint, outcome = MessagePassingRulesBase.unwrap_outcome(N.message(N.Shape, ClusterTarget((:out, :θ)), args))
    logp(x) = logpdf(Gamma(2.0, x[2]), x[1]) + logpdf(Gamma(3.0, 1.0), x[1]) + logpdf(Gamma(2.0, 2.0), x[2])
    reference = EA.laplace(logp, [3.0, 4.0], EA.Componentwise(EA.Positive(), EA.Positive()))
    @test joint isa EA.TransformedGaussian && outcome.converged
    @test mean(joint.gaussian) ≈ mean(reference.distribution.gaussian) && cov(joint.gaussian) ≈ cov(reference.distribution.gaussian)
    # Its entropy is defined, for the free energy.
    @test isfinite(entropy(joint))

    # Members that are vectors are not joined yet.
    vector = RuleArgs(m = (out = MvNormalMeanCovariance(zeros(2), [1.0 0.0; 0.0 1.0]), a = NormalMeanVariance(0.0, 1.0)), q = (b = PointMass(0.5),))
    @test N.message(N.Toy, ClusterTarget((:out, :a)), vector) === nothing
end

@testitem "ExpectationRuleFallback: the joint over a Delta node's inputs" tags = [:universal] setup = [DeltaContext] begin
    using MessagePassingRulesBase, BayesBase, ExponentialFamily
    using MessagePassingRulesBase: RuleArgs, ClusterTarget
    D = DeltaContext

    # N(x + y; 1, 0.5) N(x; 0, 1) N(y; 0, 2) is Gaussian: precision [3 2; 2 2.5], linear term [2, 2].
    sum2(x, y) = x + y
    args = RuleArgs(m = (out = NormalMeanVariance(1.0, 0.5), in = (NormalMeanVariance(0.0, 1.0), NormalMeanVariance(0.0, 2.0))))
    joint, outcome = MessagePassingRulesBase.unwrap_outcome(D.message(sum2, ClusterTarget((:in,)), args))
    @test joint isa MvNormalMeanCovariance && outcome.method === :joint_laplace && outcome.converged
    Λ = [3.0 2.0; 2.0 2.5]
    @test mean(joint) ≈ Λ \ [2.0, 2.0]
    @test cov(joint) ≈ inv(Λ)
end

@testitem "ExpectationRuleFallback: average energies, mean-field and joint" tags = [:universal] setup = [UniversalNodes] begin
    using MessagePassingRulesBase, BayesBase, ExponentialFamily
    using MessagePassingRulesBase: RuleArgs, RuleContext, call_energy_fallback
    import ExpectationApproximations as EA
    import DeltaMessagePassingRules: ExpectationRuleFallback
    N = UniversalNodes
    energy(args) = call_energy_fallback(ExpectationRuleFallback(), N.Toy, nothing, N.context(), args)

    # -E[-(out - a)²/2b] = (E[out - a]² + var(out - a)) / 2b.
    @test energy(RuleArgs(q = (out = NormalMeanVariance(1.0, 0.5), a = NormalMeanVariance(0.0, 2.0), b = PointMass(0.5)))) ≈ (1 + 0.5 + 2) / (2 * 0.5)
    # A joint of out and a, with their covariance.
    m, Σ = [1.0, 0.2], [0.5 0.3; 0.3 2.0]
    q = MessagePassingRulesBase.Marginals((b = PointMass(0.5),), Val(((:out, :a),)), (MvNormalMeanCovariance(m, Σ),))
    args = RuleArgs(MessagePassingRulesBase.Messages(NamedTuple()), q)
    @test energy(args) ≈ ((m[1] - m[2])^2 + Σ[1, 1] + Σ[2, 2] - 2 * Σ[1, 2]) / (2 * 0.5)
    # A node with groups, under mean field.
    mix = RuleArgs(q = (out = PointMass(0.5), z = BayesBase.PointMass(2), means = (NormalMeanVariance(-1.0, 1.0), NormalMeanVariance(2.0, 0.5))))
    @test call_energy_fallback(ExpectationRuleFallback(), N.Mix, nothing, N.context(), mix) ≈ ((0.5 - 2.0)^2 + 0.5) / 2
    # An interface without a marginal: nothing to say.
    @test energy(RuleArgs(q = (out = PointMass(1.0), a = PointMass(0.0)))) === nothing
end
