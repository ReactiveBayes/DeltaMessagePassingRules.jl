# Cavity propagation: `MomentForm`, the sites `proj(m · c) / c` against a cavity, proper or not,
# the pull-back of a message in weighted-mean form, and `cavity_energy`, EP's term of a node that
# sends a raw message.

@testitem "moment_form: closed forms pass, and products are moment-matched" tags = [:universal] begin
    using DeltaMessagePassingRules, BayesBase, Distributions, ExponentialFamily
    using MessagePassingRulesBase: NodeFunctionLogPdf
    import DeltaMessagePassingRules: MomentForm, moment_form
    import ExpectationApproximations as EA
    for d in (NormalMeanVariance(1.0, 2.0), Gamma(2.0, 1.0), PointMass(1.0))
        result = moment_form(MomentForm(), d)
        @test result.distribution === d && result.method === :closed_form
    end
    # a Gaussian prior and a Poisson likelihood of the log-rate: against `moment_matching`
    product = ProductOf(NormalMeanVariance(0.0, 1.0), NodeFunctionLogPdf(x -> 3x - exp(x)))
    result = moment_form(MomentForm(points = 30), product)
    reference = EA.moment_matching(x -> logpdf(product, x), 0.0; backend = EA.GaussHermite(30))
    @test result.distribution isa NormalMeanVariance && result.method === :moment_matching
    @test mean(result.distribution) ≈ mean(reference.distribution) && var(result.distribution) ≈ var(reference.distribution)
    @test result.logZ ≈ reference.logZ && result.mode ≈ reference.mean
    @test_throws ArgumentError MomentForm(nomode = :drop)
end

@testitem "site: closed messages are their own, raw ones are projections less the cavity" tags = [:universal] begin
    using DeltaMessagePassingRules, BayesBase, Distributions, ExponentialFamily, ForwardDiff
    using MessagePassingRulesBase: NodeFunctionLogPdf
    import DeltaMessagePassingRules: LaplaceForm, MomentForm, site
    import ExpectationApproximations as EA

    cavity = NormalWeightedMeanPrecision(0.5, 2.0)
    gaussian = NormalMeanVariance(1.0, 3.0)
    @test site(LaplaceForm(), gaussian, cavity).site === gaussian
    @test site(LaplaceForm(), PointMass(1.0), cavity).outcome.method === :closed_form
    raw = NodeFunctionLogPdf(x -> 2x - exp(x))
    # against a cavity that is not a Gaussian, the message passes
    passed = site(LaplaceForm(), raw, Gamma(2.0, 1.0))
    @test passed.site === raw && passed.outcome.method === :passed

    # Laplace: the site is the second-order expansion of log f at the tilted mode
    result = site(LaplaceForm(), raw, cavity)
    x̂ = result.outcome.mode
    @test result.site isa NormalWeightedMeanPrecision && result.outcome.method === :laplace
    g, h = ForwardDiff.derivative(x -> 2x - exp(x), x̂), ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> 2y - exp(y), x), x̂)
    ξs, ws = weightedmean_precision(result.site)
    @test ws ≈ -h && ξs ≈ g - h * x̂
    # the mode of the tilted density: log f + the cavity's kernel
    @test 2 - exp(x̂) + 0.5 - 2.0 * x̂ ≈ 0 atol = 1.0e-8

    # moment matching: site · cavity has the tilted density's moments
    result = site(MomentForm(points = 30), raw, cavity)
    q = prod(PreserveTypeProd(Distribution), result.site, cavity)
    reference = EA.moment_matching(x -> 2x - exp(x) + 0.5x - x^2, 0.0; backend = EA.GaussHermite(30))
    @test mean(q) ≈ mean(reference.distribution) && var(q) ≈ var(reference.distribution)
end

@testitem "site: against an improper cavity, in natural parameters" tags = [:universal] begin
    using DeltaMessagePassingRules, BayesBase, Distributions, ExponentialFamily, LinearAlgebra
    using MessagePassingRulesBase: NodeFunctionLogPdf
    import DeltaMessagePassingRules: LaplaceForm, MomentForm, site

    # A Gaussian likelihood of x₁ alone against a cavity singular in x₂ and indefinite overall:
    # the tilted density is proper in x₁ only through the likelihood, and the site, being exact,
    # is the likelihood's natural parameters, whatever the cavity.
    y, r = 0.8, 0.1
    raw = NodeFunctionLogPdf(x -> -(y - x[1])^2 / (2r))
    for W in ([0.5 0.0; 0.0 2.0], [-0.3 0.0; 0.0 2.0])
        cavity = MvNormalWeightedMeanPrecision([0.1, -0.4], W)
        for form in (LaplaceForm(), MomentForm())
            result = site(form, raw, cavity)
            ξ, Λ = weightedmean_precision(result.site)
            @test Λ ≈ [1 / r 0.0; 0.0 0.0] atol = 1.0e-6
            @test ξ ≈ [y / r, 0.0] atol = 1.0e-6
        end
    end
end

@testitem "ExpectationRuleFallback: a weighted-mean message on out pulls back as its kernel" tags = [:universal] begin
    using DeltaMessagePassingRules, BayesBase, ExponentialFamily, MessagePassingRulesBase
    using MessagePassingRulesBase: RuleArgs, RuleContext, IndexedTarget, Target, call_fallback, unwrap_fallback
    import ExpectationApproximations as EA
    struct Fn{F}
        f::F
    end
    MessagePassingRulesBase.getnodefn(node::Fn, ::Target{:out}) = node.f
    ctx = RuleContext(node = Fn(sin), expectation = EA.EADefault())
    # an indefinite message, which has no log-density
    outbound = MvNormalWeightedMeanPrecision([0.3, 0.1], [1.0 0.0; 0.0 -0.5])
    f(x) = [sin(x), x]
    ctx = RuleContext(node = Fn(f), expectation = EA.EADefault())
    message, _ = unwrap_fallback(call_fallback(DeltaMessagePassingRules.ExpectationRuleFallback(), DeltaFn{typeof(f)}, IndexedTarget(:in, 1), nothing, ctx, RuleArgs(m = (out = outbound, in = (nothing,)))))
    for x in (-0.5, 0.2, 1.0)
        z = f(x)
        @test logpdf(message, x) ≈ 0.3 * z[1] + 0.1 * z[2] - (z[1]^2 - 0.5 * z[2]^2) / 2
    end
end

@testitem "cavity_energy: EP's term, against Gaussian integrals" tags = [:universal] begin
    using DeltaMessagePassingRules, BayesBase, Distributions, ExponentialFamily, MessagePassingRulesBase
    using MessagePassingRulesBase: RuleArgs, RuleContext, Target, unwrap_outcome, call_energy_fallback
    import DeltaMessagePassingRules: LaplaceForm, MomentForm, cavity_energy
    import ExpectationApproximations as EA

    # `out ~ Lik(x)`: a Gaussian log-density of the observation `out` with mean `x`, variance 0.5
    struct Lik{X}
        x::X
    end
    BayesBase.logpdf(d::Lik, y) = -(y - d.x)^2 / (2 * 0.5) - log(2π * 0.5) / 2
    @define_factor_node(node = Lik, type = Stochastic, interfaces = [:out, :x])

    # With an observed out and a Gaussian cavity N(m, v), log ∫ f c = log N(y | m, v + 0.5) for the
    # normalised cavity; and the marginal q is the exact posterior.
    y, m, v = 1.2, 0.3, 2.0
    cavity = NormalMeanVariance(m, v)
    q = NormalMeanVariance((m * 0.5 + y * v) / (v + 0.5), v * 0.5 / (v + 0.5))
    ctx = RuleContext(expectation = EA.EADefault())
    args = RuleArgs(m = (out = PointMass(y), x = cavity), q = (out = PointMass(y), x = q))
    # the cavity enters as its kernel: its normaliser cancels between the two parts
    ξ, w = weightedmean_precision(cavity)
    kernel_normaliser = (log(2π / w) + ξ^2 / w) / 2
    expected = -(logpdf(Normal(m, sqrt(v + 0.5)), y) + kernel_normaliser) + (ξ * mean(q) - w * (var(q) + mean(q)^2) / 2)
    for form in (LaplaceForm(), MomentForm())
        F, outcome = unwrap_outcome(cavity_energy(form, Lik, ctx, args))
        @test F ≈ expected rtol = 1.0e-8
        @test outcome.converged
        # through the fallback, given messages
        energy, _ = unwrap_outcome(call_energy_fallback(DeltaMessagePassingRules.ExpectationRuleFallback(form), Lik, nothing, ctx, args))
        @test energy ≈ F
    end
    # without messages, the average energy as before
    @test call_energy_fallback(DeltaMessagePassingRules.ExpectationRuleFallback(LaplaceForm()), Lik, nothing, ctx, RuleArgs(q = (out = PointMass(y), x = q))) isa Real

    # A deterministic node, out = 2x: ∫ c_out(2x) c_in(x) dx with both cavities as kernels
    struct Fn{F}
        f::F
    end
    MessagePassingRulesBase.getnodefn(node::Fn, ::Target{:out}) = node.f
    double(x) = 2x
    c_out, c_in = NormalWeightedMeanPrecision(0.4, 1.5), NormalWeightedMeanPrecision(-0.2, 0.8)
    q_out, q_in = NormalMeanVariance(0.1, 0.3), NormalMeanVariance(0.05, 0.1)
    ctx = RuleContext(node = Fn(double), expectation = EA.EADefault())
    args = RuleArgs(m = (out = c_out, in = (c_in,)), q = (out = q_out, in = (q_in,)))
    # exponent (2·0.4 - 0.2) x - (4·1.5 + 0.8) x²/2: a Gaussian integral
    a, b = 2 * 0.4 - 0.2, 4 * 1.5 + 0.8
    integral = (log(2π / b) + a^2 / b) / 2
    cross(q, c) = ((ξ, w) = weightedmean_precision(c); ξ * mean(q) - w * (var(q) + mean(q)^2) / 2)
    F, _ = unwrap_outcome(cavity_energy(MomentForm(), DeltaFn{typeof(double)}, ctx, args))
    @test F ≈ -integral + cross(q_out, c_out) + cross(q_in, c_in) rtol = 1.0e-8
end
