# Natural-gradient propagation: `natural_gradient_target`, `G(q)⁻¹ Cov_q[T, ℓ]`, against closed
# forms in a Gaussian and a Gamma family, and `site(::NaturalGradientForm, …)`, the damped step
# towards it, in the family of the cavity.

@testmodule NaturalGradientClosedForms begin
    using BayesBase, Distributions, SpecialFunctions

    # The target at N(m, v) of `ℓ(z) = (a - 1) z - b eᶻ`, a Gamma(a, b) message pulled back
    # through `λ = eᶻ`: with ρ = E[eᶻ], Λ⋆ = bρ and ξ⋆ = (a - 1) - bρ + bρ m
    function gaussian_site_of_gamma(a, b, q)
        m, v = mean_var(q)
        Λ = b * exp(m + v / 2)
        return ((a - 1) - Λ + Λ * m, Λ)
    end

    # The target at Gamma(a, b) of `ℓ(λ) = log LogNormal(λ | m, v)`, a normal message pushed
    # through `λ = eᶻ`, as its shape and rate
    function gamma_site_of_lognormal(m, v, q)
        a, b = shape(q), rate(q)
        μ₁ = digamma(a) - log(b)
        κ = polygamma(2, a) / (2 * v * (a * trigamma(a) - 1))
        return ((m - μ₁) / v - a * κ, -b * κ)
    end

    export gaussian_site_of_gamma, gamma_site_of_lognormal
end

@testitem "natural_gradient_target: Gaussian beliefs, by Stein's lemma" tags = [:universal] setup = [NaturalGradientClosedForms] begin
    using DeltaMessagePassingRules, BayesBase, Distributions, ExponentialFamily, LinearAlgebra
    using MessagePassingRulesBase: NodeFunctionLogPdf
    import DeltaMessagePassingRules: natural_gradient_target
    import ExpectationApproximations as EA

    # one dimension, against the closed form, in ExponentialFamily's natural parameters (ξ, -Λ/2)
    for (a, b, m, v) in ((4.0, 1.0, 0.0, 1.0), (2.0, 3.0, 0.5, 0.5), (1.0, 0.5, -1.0, 0.1), (13.0, 2.0, 1.0, 2.0))
        q = NormalMeanVariance(m, v)
        η = natural_gradient_target(NodeFunctionLogPdf(z -> (a - 1) * z - b * exp(z)), q, nothing)
        ξ, Λ = gaussian_site_of_gamma(a, b, q)
        @test η ≈ [ξ, -Λ / 2] rtol = 1.0e-10
        @test η ≈ getnaturalparameters(convert(ExponentialFamilyDistribution, NormalWeightedMeanPrecision(ξ, Λ))) rtol = 1.0e-10
    end

    # two dimensions: `ℓ(x) = aᵀx - exp(bᵀx)`, with ρ = E[exp(bᵀx)], E[∇ℓ] = a - ρb and
    # E[∇²ℓ] = -ρbbᵀ, so Λ⋆ = ρbbᵀ, singular, and ξ⋆ = a - ρb + Λ⋆m
    a, b = [2.0, -1.0], [0.4, 0.3]
    q = MvNormalMeanCovariance([0.3, -0.2], [1.0 0.3; 0.3 0.5])
    m, Σ = mean_cov(q)
    ρ = exp(dot(b, m) + dot(b, Σ * b) / 2)
    Λ = ρ * b * b'
    η = natural_gradient_target(NodeFunctionLogPdf(x -> dot(a, x) - exp(dot(b, x))), q, nothing)
    @test η ≈ vcat(a - ρ * b + Λ * m, vec(-Λ / 2)) atol = 1.0e-10
    # in the layout of ExponentialFamily's natural parameters of the belief's family
    @test length(η) == length(getnaturalparameters(convert(ExponentialFamilyDistribution, q)))
    # a quadratic log-density is its own target, whatever the belief, with any backend exact to
    # degree 4, as Stein's lemma for the Hessian takes the fourth moments
    A, c = [1.0 0.5; -0.2 2.0], [0.4, -1.0]
    S = (A + A') / 2
    quadratic = NodeFunctionLogPdf(x -> -dot(x - c, A * (x - c)) / 2)
    for backend in (nothing, EA.GaussHermite(3))
        @test natural_gradient_target(quadratic, q, backend) ≈ vcat(S * c, vec(-S / 2)) atol = 1.0e-10
    end
end

@testitem "natural_gradient_target: any other family, against closed forms" tags = [:universal] setup = [NaturalGradientClosedForms] begin
    using DeltaMessagePassingRules, BayesBase, Distributions, ExponentialFamily
    using MessagePassingRulesBase: NodeFunctionLogPdf
    import DeltaMessagePassingRules: natural_gradient_target, PushforwardMessage
    import ExpectationApproximations as EA

    # A Gamma belief and a log-normal message, as a distribution and as the pushforward of a
    # normal through exp: the closed form, in natural parameters (a⋆ - 1, -b⋆), to 1e-9, small
    # shapes included, where Gauss–Laguerre points are 10% off
    for (m, v, a, b) in ((0.0, 1.0, 2.0, 1.0), (1.5, 0.3, 5.0, 2.0), (-1.0, 2.0, 1.5, 0.5), (2.0, 0.05, 30.0, 3.0), (0.3, 4.0, 0.7, 1.2))
        q = GammaShapeRate(a, b)
        as, bs = gamma_site_of_lognormal(m, v, q)
        for message in (LogNormal(m, sqrt(v)), PushforwardMessage(exp, (NormalMeanVariance(m, v),)))
            @test natural_gradient_target(message, q, nothing) ≈ [as - 1, -bs] rtol = 1.0e-9
        end
        # the parametrisation of the belief does not matter
        @test natural_gradient_target(LogNormal(m, sqrt(v)), Gamma(a, 1 / b), nothing) ≈ [as - 1, -bs] rtol = 1.0e-9
    end
    # a given backend is used as it is: Gauss–Laguerre, whose error in `E[log λ]` falls as a power of
    # its points, set by the shape, is close at a large shape and far at a small one
    rate_only = NodeFunctionLogPdf(λ -> -2λ)
    @test natural_gradient_target(rate_only, GammaShapeRate(30.0, 2.0), EA.GaussLaguerre(64)) ≈ [0.0, -2.0] atol = 1.0e-8
    @test !isapprox(natural_gradient_target(rate_only, GammaShapeRate(0.7, 2.0), EA.GaussLaguerre(16)), [0.0, -2.0]; atol = 1.0e-3)
    @test natural_gradient_target(rate_only, GammaShapeRate(0.7, 2.0), nothing) ≈ [0.0, -2.0] atol = 1.0e-9
    # a log-density of the belief's family is its own target: a Beta's (a - 1, b - 1)
    @test natural_gradient_target(NodeFunctionLogPdf(x -> 3log(x) + 2log(1 - x)), Beta(2.0, 5.0), nothing) ≈ [3.0, 2.0] rtol = 1.0e-6
    # where the message's log-density underflows on a negligible mass, those points are left out:
    # tanh(x) is 1 in floating point beyond 19
    beta_of_tanh = NodeFunctionLogPdf(x -> 6log(tanh(x)) + 2log(1 - tanh(x)))
    @test all(isfinite, natural_gradient_target(beta_of_tanh, GammaShapeRate(1.0, 1.0), nothing))
    # not where it holds mass
    @test !all(isfinite, natural_gradient_target(NodeFunctionLogPdf(x -> x < 1 ? 0.0 : -Inf), GammaShapeRate(1.0, 1.0), nothing))
end

@testitem "PushforwardMessage: its log-density" tags = [:universal] begin
    using DeltaMessagePassingRules, BayesBase, Distributions, ExponentialFamily
    import DeltaMessagePassingRules: PushforwardMessage

    # through a monotone function of one input, the density of the pushforward
    lognormal = PushforwardMessage(exp, (NormalMeanVariance(0.3, 0.5),))
    for z in (0.1, 1.0, 4.0)
        @test logpdf(lognormal, z) ≈ logpdf(LogNormal(0.3, sqrt(0.5)), z) rtol = 1.0e-10
    end
    @test logpdf(lognormal, -1.0) == -Inf
    # decreasing, of an input on the positive half-line, and saturating: 1/x and tanh of a Gamma
    @test logpdf(PushforwardMessage(x -> 1 / x, (Gamma(3.0, 0.5),)), 0.8) ≈ logpdf(InverseGamma(3.0, 2.0), 0.8) rtol = 1.0e-10
    for y in (0.1, 0.5, 0.999)
        @test logpdf(PushforwardMessage(tanh, (GammaShapeRate(1.0, 1.0),)), y) ≈ -atanh(y) - log(1 - y^2) rtol = 1.0e-8
    end
    # with a constant among the inputs, folded in
    shifted = PushforwardMessage((x, c) -> x + c, (NormalMeanVariance(0.0, 1.0), PointMass(2.0)))
    @test logpdf(shifted, 1.5) ≈ logpdf(Normal(2.0, 1.0), 1.5) rtol = 1.0e-10
    # not monotone, or of two inputs: the moment-matched Gaussian, exact for a linear function
    square = PushforwardMessage(x -> x^2, (NormalMeanVariance(0.0, 1.0),))
    @test logpdf(square, 1.0) ≈ logpdf(Normal(1.0, sqrt(2.0)), 1.0) rtol = 1.0e-8
    sum_of_two = PushforwardMessage(+, (NormalMeanVariance(1.0, 1.0), NormalMeanVariance(2.0, 3.0)))
    @test logpdf(sum_of_two, 2.5) ≈ logpdf(Normal(3.0, 2.0), 2.5) rtol = 1.0e-8
    @test insupport(sum_of_two, 2.5)
end

@testitem "site: NaturalGradientForm, its cases" tags = [:universal] begin
    using DeltaMessagePassingRules, BayesBase, Distributions, ExponentialFamily
    using MessagePassingRulesBase: NodeFunctionLogPdf
    import DeltaMessagePassingRules: NaturalGradientForm, site, natural_gradient_target, TiltedJointMessage, PushforwardMessage
    import ExpectationApproximations as EA

    form = NaturalGradientForm()
    raw = NodeFunctionLogPdf(x -> 2x - exp(x))
    # a raw message waits for a cavity; a closed one is its own site
    waiting = site(form, raw, missing)
    @test ismissing(waiting.site) && waiting.outcome == (method = :waiting,)
    @test site(form, PushforwardMessage(exp, (NormalMeanVariance(0.0, 1.0),)), missing).outcome.method === :waiting
    gaussian = NormalMeanVariance(1.0, 2.0)
    @test site(form, gaussian, missing).site === gaussian
    @test site(form, gaussian, NormalWeightedMeanPrecision(0.1, 2.0)).site === gaussian
    @test site(form, PointMass(1.0), GammaShapeRate(2.0, 1.0)).outcome.method === :closed_form
    # a Gamma enters by shape and rate, whose products with an improper site are closed, and any
    # other family as an ExponentialFamilyDistribution
    own = site(form, Gamma(2.0, 0.5), GammaShapeRate(3.0, 1.0))
    @test own.site isa GammaShapeRate && shape(own.site) == 2.0 && rate(own.site) == 2.0 && own.outcome.method === :closed_form
    @test prod(GenericProd(), own.site, GammaShapeRate(-0.5, 0.5)) == GammaShapeRate(0.5, 2.5)
    beta = site(form, Beta(2.0, 3.0), Beta(1.0, 1.0))
    @test beta.site isa ExponentialFamilyDistribution && convert(Distribution, beta.site) == Beta(2.0, 3.0)
    # against a cavity in no family with sites, a categorical's, or a product, the message passes
    @test site(form, raw, Categorical([0.5, 0.5])).outcome.method === :passed
    @test site(form, raw, ProductOf(NormalMeanVariance(0.0, 1.0), raw)).site === raw

    # the first step starts from the flat site: alpha = 1 and beta = 0 jump to the target at the cavity
    cavity = NormalMeanVariance(0.5, 1.0)
    target = natural_gradient_target(raw, cavity, nothing)
    jump = site(NaturalGradientForm(alpha = 1.0, beta = 0.0), raw, cavity)
    @test jump.outcome.method === :natural_gradient && jump.outcome.iterations == 1 && !jump.outcome.converged
    @test collect(weightedmean_precision(jump.site)) ≈ [target[1], -2 * target[2]]
    @test jump.outcome.η ≈ target && jump.outcome.velocity ≈ target
    # the damped step threads its state through `previous`: v ← βv + α(η⋆ - η), η ← η + v
    first = site(form, raw, cavity)
    @test first.outcome.η ≈ 0.5 * target
    q = prod(GenericProd(), cavity, first.site)
    second = site(form, raw, cavity, first.outcome)
    expected = 0.2 * first.outcome.velocity + 0.5 * (natural_gradient_target(raw, q, nothing) - first.outcome.η)
    @test second.outcome.velocity ≈ expected
    @test second.outcome.η ≈ first.outcome.η + expected && second.outcome.iterations == 2
    # repeated, it settles at the target of its own marginal, the fixed point
    state = let state = first.outcome
        for _ in 1:80
            state = site(form, raw, cavity, state).outcome
        end
        state
    end
    @test state.converged
    marginal = DeltaMessagePassingRules.family_marginal(NormalMeanVariance, [0.5, -0.5] + state.η, nothing)
    @test natural_gradient_target(raw, marginal, nothing) ≈ state.η rtol = 1.0e-8
    # a previous outcome of another family, or of another method, starts afresh
    @test site(form, raw, cavity, (method = :laplace, mode = 0.3)).outcome.iterations == 1

    # the step is clipped to max_step
    clipped = site(NaturalGradientForm(alpha = 1.0, beta = 0.0, max_step = 0.1), raw, cavity)
    @test sqrt(sum(abs2, clipped.outcome.η)) ≈ 0.1

    # halved until the marginal is proper: a message that takes 5 from a Gamma's shape
    shrinking = NodeFunctionLogPdf(x -> -5 * log(x))
    gamma_cavity = GammaShapeRate(2.0, 1.0)
    halved = site(NaturalGradientForm(alpha = 1.0, beta = 0.0), shrinking, gamma_cavity)
    @test halved.site isa GammaShapeRate
    @test shape(prod(GenericProd(), gamma_cavity, halved.site)) > 0
    # the target is (-5, 0), the step halved twice, to (-1.25, 0), the shape at 0.75
    @test halved.outcome.η ≈ [-1.25, 0.0] atol = 1.0e-8
    # a last site that, against a cavity that has moved, gives an improper marginal is halved first
    moved = site(form, shrinking, GammaShapeRate(1.2, 1.0), halved.outcome)
    @test shape(prod(GenericProd(), GammaShapeRate(1.2, 1.0), moved.site)) > 0
    # a target that is not finite leaves the site, or sends the flat one at a first step
    nowhere = NodeFunctionLogPdf(x -> NaN)
    @test site(form, nowhere, cavity, jump.outcome).site === nothing
    flat = site(form, nowhere, cavity)
    @test collect(weightedmean_precision(flat.site)) == [0.0, 0.0] && !flat.outcome.converged

    # a TiltedJointMessage, belief propagation's message `log ∫ f c`: here `log N(1 | a, 3)`,
    # quadratic, so that one undamped step reaches its site
    f(; out, a, b) = logpdf(Normal(a + b, 1.0), out)
    joint = TiltedJointMessage(f, nothing, (PointMass(1.0), nothing, NormalMeanVariance(0.0, 2.0)), (:out, :a, :b), 2, EA.EADefault())
    @test site(form, joint, missing).outcome.method === :waiting
    reached = site(NaturalGradientForm(alpha = 1.0, beta = 0.0), joint, NormalMeanVariance(0.0, 1.0))
    @test collect(weightedmean_precision(reached.site)) ≈ [1 / 3, 1 / 3] rtol = 1.0e-6
    @test site(form, joint, Categorical([0.5, 0.5])).outcome.method === :passed

    @test_throws ArgumentError NaturalGradientForm(alpha = 0.0)
    @test_throws ArgumentError NaturalGradientForm(beta = 1.0)
    @test_throws ArgumentError NaturalGradientForm(max_step = 0.0)
    @test sprint(show, NaturalGradientForm()) == "NaturalGradientForm(alpha = 0.5, beta = 0.2, max_step = Inf, tol = 1.0e-8)"
end

@testitem "natural_gradient_target: a closed form, by a method for the message" tags = [:universal] begin
    using DeltaMessagePassingRules, BayesBase, Distributions, ExponentialFamily
    import DeltaMessagePassingRules: NaturalGradientForm, natural_gradient_target, site, is_closed_form

    # a raw message type of another package, with its target in closed form at a Gamma belief
    struct ShapeShift
        k::Float64
    end
    DeltaMessagePassingRules.is_closed_form(::ShapeShift) = false
    DeltaMessagePassingRules.natural_gradient_target(message::ShapeShift, q::GammaDistributionsFamily, backend) = [message.k, 0.0]
    result = site(NaturalGradientForm(alpha = 1.0, beta = 0.0), ShapeShift(1.5), GammaShapeRate(2.0, 1.0))
    @test result.site isa GammaShapeRate && shape(result.site) == 2.5 && rate(result.site) == 0
    @test result.outcome.η == [1.5, 0.0]
end

@testitem "ExpectationRuleFallback: the free-energy term of each site form" tags = [:universal] begin
    using DeltaMessagePassingRules, BayesBase, Distributions, ExponentialFamily, MessagePassingRulesBase
    using MessagePassingRulesBase: RuleArgs, RuleContext, unwrap_outcome, call_energy_fallback
    import DeltaMessagePassingRules: ExpectationRuleFallback, LaplaceForm, MomentForm, NaturalGradientForm, cavity_energy
    import ExpectationApproximations as EA

    struct Observation{X}
        x::X
    end
    BayesBase.logpdf(d::Observation, y) = -(y - d.x)^2 / (2 * 0.5) - log(2π * 0.5) / 2
    @define_factor_node(node = Observation, type = Stochastic, interfaces = [:out, :x])

    @test ExpectationRuleFallback().energy === :average
    @test ExpectationRuleFallback(LaplaceForm()).energy === :cavity
    @test ExpectationRuleFallback(MomentForm()).energy === :cavity
    @test ExpectationRuleFallback(NaturalGradientForm()).energy === :average
    @test ExpectationRuleFallback(LaplaceForm(); energy = :average).energy === :average
    @test_throws ArgumentError ExpectationRuleFallback(LaplaceForm(); energy = :bethe)

    y, q = 1.2, NormalMeanVariance(0.4, 0.3)
    ctx = RuleContext(expectation = EA.EADefault())
    marginals = (out = PointMass(y), x = q)
    args = RuleArgs(m = (out = PointMass(y), x = NormalMeanVariance(0.3, 2.0)), q = marginals)
    # the average energy -E_q[log f], given the inbound messages or not
    average = -(-((y - mean(q))^2 + var(q)) / (2 * 0.5) - log(2π * 0.5) / 2)
    @test call_energy_fallback(ExpectationRuleFallback(NaturalGradientForm()), Observation, nothing, ctx, args) ≈ average
    @test call_energy_fallback(ExpectationRuleFallback(), Observation, nothing, ctx, RuleArgs(q = marginals)) ≈ average
    # expectation propagation's term under the other forms
    @test first(unwrap_outcome(call_energy_fallback(ExpectationRuleFallback(LaplaceForm()), Observation, nothing, ctx, args))) ≈
        first(unwrap_outcome(cavity_energy(LaplaceForm(), Observation, ctx, args)))
end
