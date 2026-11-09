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

@testitem "MomentForm: its quadrature, built once, or any backend with a point set" tags = [:universal] begin
    using DeltaMessagePassingRules, BayesBase, Distributions, ExponentialFamily
    using MessagePassingRulesBase: NodeFunctionLogPdf
    import DeltaMessagePassingRules: MomentForm, site
    import ExpectationApproximations as EA
    @test MomentForm(points = 7).backend isa EA.GaussHermite
    # a Gaussian likelihood of x₁ against a cavity indefinite in x₁: the tilted density is a
    # Gaussian, whose moments a rule exact to degree 3 gets, so the site is the likelihood's
    y, r = 0.8, 0.1
    raw = NodeFunctionLogPdf(x -> -(y - x[1])^2 / (2r))
    cavity = MvNormalWeightedMeanPrecision([0.1, -0.4], [-0.3 0.2; 0.2 2.0])
    for backend in (EA.SphericalRadialCubature(), EA.GenUnscented())
        result = site(MomentForm(backend = backend), raw, cavity)
        ξ, Λ = weightedmean_precision(result.site)
        @test Λ ≈ [1 / r 0.0; 0.0 0.0] atol = 1.0e-8
        @test ξ ≈ [y / r, 0.0] atol = 1.0e-8
    end
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

@testitem "site: a pushforward against its output's cavity" tags = [:universal] begin
    using DeltaMessagePassingRules, BayesBase, Distributions, ExponentialFamily, LinearAlgebra
    import DeltaMessagePassingRules: LaplaceForm, MomentForm, PushforwardMessage, site, same_message, same_density

    # z = 2x, linear: the site of the pushforward is belief propagation's message, N(2m, 4v),
    # whatever the output's cavity; against none yet, the same
    double(x) = 2x
    input = NormalMeanVariance(0.3, 0.5)
    message = PushforwardMessage(double, (input,))
    expected = NormalMeanVariance(0.6, 2.0)
    for form in (LaplaceForm(), MomentForm()), cavity in (NormalWeightedMeanPrecision(0.2, 1.5), missing)
        result = site(form, message, cavity)
        ξ, w = weightedmean_precision(result.site)
        @test w ≈ 1 / var(expected) rtol = 1.0e-8
        @test ξ ≈ mean(expected) / var(expected) rtol = 1.0e-8
    end
    # a function of two coordinates, against an output cavity that is improper: the tilted
    # density is proper through the input
    first_plus(x) = x[1] + x[2]
    input = MvNormalMeanCovariance([0.1, -0.2], [0.4 0.1; 0.1 0.3])
    result = site(MomentForm(), PushforwardMessage(first_plus, (input,)), NormalWeightedMeanPrecision(0.1, -0.2))
    ξ, w = weightedmean_precision(result.site)
    @test w ≈ 1 / (0.4 + 0.3 + 2 * 0.1) rtol = 1.0e-6
    @test ξ / w ≈ -0.1 atol = 1.0e-6

    # the same message to a tolerance, for a site that need not be projected again
    @test same_message(message, PushforwardMessage(double, (NormalMeanVariance(0.3, 0.5),)))
    @test !same_message(message, PushforwardMessage(double, (NormalMeanVariance(0.3 + 1.0e-6, 0.5),)), 1.0e-10)
    @test same_message(message, PushforwardMessage(double, (NormalMeanVariance(0.3 + 1.0e-12, 0.5),)), 1.0e-10)
    @test same_density(NormalWeightedMeanPrecision(1.0, -0.5), NormalWeightedMeanPrecision(1.0, -0.5))
    @test !same_density(NormalWeightedMeanPrecision(1.0, -0.5), NormalWeightedMeanPrecision(1.0, 0.5), 1.0e-10)
end

@testmodule JointNodes begin
    using BayesBase, Distributions, MessagePassingRulesBase

    # `out ~ LogPrecisionNormal(μ, z)`: a Gaussian with mean `μ` and precision `exp(z)`, with no
    # rules, a node with two random inputs
    struct LogPrecisionNormal{M, Z} <: ContinuousUnivariateDistribution
        μ::M
        z::Z
    end
    BayesBase.logpdf(d::LogPrecisionNormal, y::Real) = (d.z - log(2π) - exp(d.z) * (y - d.μ)^2) / 2
    @define_factor_node(node = LogPrecisionNormal, type = Stochastic, interfaces = [:out, :μ, :z])
end

@testitem "TiltedJointMessage: a node with two random inputs, against a grid" tags = [:universal] setup = [JointNodes] begin
    using DeltaMessagePassingRules, BayesBase, Distributions, ExponentialFamily, MessagePassingRulesBase
    using MessagePassingRulesBase: RuleArgs, RuleContext, Target, call_fallback, unwrap_fallback
    import DeltaMessagePassingRules: ExpectationRuleFallback, LaplaceForm, MomentForm, TiltedJointMessage, site, same_message
    import ExpectationApproximations as EA
    J = JointNodes

    y = 1.3
    cμ, cz = NormalWeightedMeanPrecision(0.2, 1.0), NormalWeightedMeanPrecision(0.0, 2.0)
    ctx = RuleContext(expectation = EA.EADefault())
    message(propagation, target, args) = first(unwrap_fallback(call_fallback(ExpectationRuleFallback(propagation), J.LogPrecisionNormal, target, nothing, ctx, args)))
    toμ = message(MomentForm(), Target(:μ), RuleArgs(m = (out = PointMass(y), z = cz)))
    toz = message(MomentForm(), Target(:z), RuleArgs(m = (out = PointMass(y), μ = cμ)))
    # under propagation, belief propagation's message; without, the expected log-density as before
    @test toμ isa TiltedJointMessage && toz isa TiltedJointMessage
    @test !(message(nothing, Target(:μ), RuleArgs(m = (out = PointMass(y), z = cz))) isa TiltedJointMessage)
    # data that comes as a marginal, as RxInfer gives an observation, is plugged in as well
    @test message(MomentForm(), Target(:μ), RuleArgs(m = (z = cz,), q = (out = PointMass(y),))) isa TiltedJointMessage
    # an input that comes as a marginal, mean field, keeps the expected log-density
    @test !(message(MomentForm(), Target(:μ), RuleArgs(m = (out = PointMass(y),), q = (z = cz,))) isa TiltedJointMessage)

    # the tilted density over (μ, z) on a fine grid
    μs, zs = range(-5, 5; length = 801), range(-6, 5; length = 801)
    logt = [logpdf(J.LogPrecisionNormal(μ, z), y) + logpdf(cμ, μ) + logpdf(cz, z) for μ in μs, z in zs]
    w = exp.(logt .- maximum(logt))
    w ./= sum(w)
    mμ, mz = sum(w .* μs), sum(w .* zs')
    vμ, vz = sum(w .* (μs .- mμ) .^ 2), sum(w .* (zs' .- mz) .^ 2)

    # each site times its cavity is the tilted density's marginal of its variable
    qμ = prod(PreserveTypeProd(Distribution), site(MomentForm(points = 30), toμ, cμ).site, cμ)
    qz = prod(PreserveTypeProd(Distribution), site(MomentForm(points = 30), toz, cz).site, cz)
    @test mean(qμ) ≈ mμ atol = 1.0e-4
    @test var(qμ) ≈ vμ rtol = 1.0e-3
    @test mean(qz) ≈ mz atol = 1.0e-4
    @test var(qz) ≈ vz rtol = 1.0e-3

    # Laplace: the variable's block of the joint Laplace approximation
    joint = EA.laplace(v -> logpdf(J.LogPrecisionNormal(v[1], v[2]), y) + logpdf(cμ, v[1]) + logpdf(cz, v[2]), [0.0, 0.0])
    qμ = prod(PreserveTypeProd(Distribution), site(LaplaceForm(), toμ, cμ).site, cμ)
    @test mean(qμ) ≈ mean(joint.distribution)[1] atol = 1.0e-8
    @test var(qμ) ≈ cov(joint.distribution)[1, 1] rtol = 1.0e-6

    # its log-density, where a variable projects it as a product: the factor integrated against
    # the other input's message
    for μ in (-0.5, 0.4)
        reference = log(sum(z -> exp(logpdf(J.LogPrecisionNormal(μ, z), y) + logpdf(cz, z)), zs) * step(zs))
        @test logpdf(toμ, μ) ≈ reference rtol = 1.0e-3
    end
    # the same message again is the same, and a site against a missing cavity waits
    @test same_message(toμ, message(MomentForm(), Target(:μ), RuleArgs(m = (out = PointMass(y), z = cz))), 1.0e-10)
    @test !same_message(toμ, toz)
    @test ismissing(site(MomentForm(), toμ, missing).site)
end

@testitem "TiltedJointMessage: a deterministic node's pull-back to one of two inputs" tags = [:universal] begin
    using DeltaMessagePassingRules, BayesBase, Distributions, ExponentialFamily, MessagePassingRulesBase
    using MessagePassingRulesBase: RuleArgs, RuleContext, Target, IndexedTarget, call_fallback, unwrap_fallback
    import DeltaMessagePassingRules: ExpectationRuleFallback, MomentForm, TiltedJointMessage, site
    import ExpectationApproximations as EA
    struct Fn{F}
        f::F
    end
    MessagePassingRulesBase.getnodefn(node::Fn, ::Target{:out}) = node.f
    product(a, b) = a * b
    ctx = RuleContext(node = Fn(product), expectation = EA.EADefault())
    # out = a·b observed through a Gaussian message on out, b's message a Gaussian about 1 (a broad
    # one would leave a ridge along a·b = 1.5, which no Gaussian resolves)
    outbound, ca, cb = NormalMeanVariance(1.5, 0.2), NormalWeightedMeanPrecision(0.5, 1.0), NormalWeightedMeanPrecision(25.0, 25.0)
    args = RuleArgs(m = (out = outbound, in = (nothing, cb)))
    toa, _ = unwrap_fallback(call_fallback(ExpectationRuleFallback(MomentForm()), DeltaFn{typeof(product)}, IndexedTarget(:in, 1), nothing, ctx, args))
    @test toa isa TiltedJointMessage
    as, bs = range(-6, 6; length = 801), range(-6, 6; length = 801)
    logt = [logpdf(outbound, a * b) + logpdf(ca, a) + logpdf(cb, b) for a in as, b in bs]
    w = exp.(logt .- maximum(logt))
    w ./= sum(w)
    ma = sum(w .* as)
    va = sum(w .* (as .- ma) .^ 2)
    qa = prod(PreserveTypeProd(Distribution), site(MomentForm(points = 30), toa, ca).site, ca)
    @test mean(qa) ≈ ma atol = 1.0e-4
    @test var(qa) ≈ va rtol = 1.0e-3
end
