# The point-set methods of MessagePassingRulesApproximations under the Delta node: Gauss–Hermite
# cubature and the generalized unscented transform on normal messages, the latter on messages of
# other families too, and Gauss–Laguerre quadrature on gamma messages. The references are exact:
# for a quadratic function every method here is exact.

@testitem "rules:Delta:gausshermite" tags = [:rules] setup = [DeltaTestNode] begin
    using DeltaMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesApproximations, ExponentialFamily
    using .DeltaTestNode: context, g, h
    cubature = DeltaApproximation(method = GaussHermiteCubature(5))

    # g(x) = x² - 5 under N(2, 3): mean m² + v - 5 = 2, variance 4m²v + 2v² = 66.
    @test_message_update_rule(
        node = DeltaFn{typeof(g)}, target = :out, algorithm = cubature, check_type_promotion = false,
        cases = [
            (m = (in = (NormalMeanVariance(2.0, 3.0),),), ctx = context(g)) => NormalMeanVariance(2.0, 66.0),
            (m = (in = (MvNormalMeanCovariance([2.0], [3.0;;]),),), ctx = context(g)) => MvNormalMeanCovariance([2.0], [66.0;;]),
            # h(x, y) = x² - y: the variance of y adds.
            (m = (in = (NormalMeanVariance(2.0, 3.0), NormalMeanVariance(5.0, 1.0)),), ctx = context(h)) => NormalMeanVariance(2.0, 67.0),
        ],
    )

    # The joint: the forward statistics m̃ = 0, Ṽ = 18 and C = 2mv = 4 of N(2, 1), smoothed with
    # N(2, 3) from `out`: the mean 2 + 4 · 2 / 21 and variance 1 - 16 / 21.
    @test_marginal_update_rule(
        node = DeltaFn, target = (:in,), algorithm = cubature, check_type_promotion = false, atol = 1.0e-10,
        cases = [
            (m = (out = NormalMeanVariance(2.0, 3.0), in = (NormalMeanVariance(2.0, 1.0),)), ctx = context(g)) => JointNormal(NormalMeanVariance(2 + 8 / 21, 1 - 16 / 21), ((),)),
        ],
    )

    # Towards an input: through the inverse of `2x + 1`, exactly, and as the joint's share
    # without one.
    @test_message_update_rule(
        node = DeltaFn, target = (:in, 1), algorithm = DeltaApproximation(method = GaussHermiteCubature(3), inverse = y -> (y - 1) / 2), check_type_promotion = false, atol = 1.0e-12,
        cases = [(m = (out = NormalMeanVariance(3.0, 2.0), in = (nothing,)),) => NormalMeanVariance(1.0, 0.5)],
    )
    @test_message_update_rule(
        node = DeltaFn, target = (:in, 1), algorithm = cubature, check_type_promotion = false, atol = 1.0e-3,
        cases = [
            (m = (in = (NormalMeanVariance(5.0, 10.0), nothing),), clusters = ((:in,) => JointNormal(MvNormalMeanCovariance(ones(2), [1.0 0.1; 0.1 1.0]), ((), ())),)) => NormalWeightedMeanPrecision(0.5, 0.9),
        ],
    )
end

@testitem "rules:Delta:genunscented" tags = [:rules] setup = [DeltaTestNode] begin
    using DeltaMessagePassingRules, MessagePassingRulesBase, MessagePassingRulesTestUtils, MessagePassingRulesApproximations, ExponentialFamily, Distributions, BayesBase
    import MessagePassingRulesApproximations: GenUnscented, GenConstrainedUnscented
    using .DeltaTestNode: context, g
    generalized = DeltaApproximation(method = GenUnscented())

    # Normal inputs: the same exact moments as cubature.
    @test_message_update_rule(
        node = DeltaFn{typeof(g)}, target = :out, algorithm = generalized, check_type_promotion = false,
        cases = [(m = (in = (NormalMeanVariance(2.0, 3.0),),), ctx = context(g)) => NormalMeanVariance(2.0, 66.0)],
    )
    @test_marginal_update_rule(
        node = DeltaFn, target = (:in,), algorithm = generalized, check_type_promotion = false, atol = 1.0e-10,
        cases = [
            (m = (out = NormalMeanVariance(2.0, 3.0), in = (NormalMeanVariance(2.0, 1.0),)), ctx = context(g)) => JointNormal(NormalMeanVariance(2 + 8 / 21, 1 - 16 / 21), ((),)),
        ],
    )

    # A gamma input with shape 2 and scale 1.5: x² has mean α(α + 1)θ² = 13.5 and variance
    # E[x⁴] - E[x²]² = θ⁴ α(α + 1)(α + 2)(α + 3) - 13.5² = 425.25, exact since the sigma points match
    # the gamma's four moments.
    square(x) = x^2
    @test_message_update_rule(
        node = DeltaFn{typeof(square)}, target = :out, algorithm = generalized, check_type_promotion = false, atol = 1.0e-8,
        cases = [
            (m = (in = (GammaShapeScale(2.0, 1.5),),), ctx = context(square)) => NormalMeanVariance(13.5, 425.25),
            (m = (in = (Gamma(2.0, 1.5),),), ctx = context(square)) => NormalMeanVariance(13.5, 425.25),
        ],
    )

    # Bounds keep the points positive; the mean and variance of the identity are still exact.
    positive = DeltaApproximation(method = GenConstrainedUnscented(0.0, Inf))
    @test_message_update_rule(
        node = DeltaFn{typeof(identity)}, target = :out, algorithm = positive, check_type_promotion = false, atol = 1.0e-10,
        cases = [(m = (in = (GammaShapeScale(0.5, 2.0),),), ctx = context(identity)) => NormalMeanVariance(1.0, 2.0)],
    )

    # Through a known inverse, from a message of another family on `out`: y = x² - 5, x = √(y + 5),
    # with y + 5 a gamma: the moments of its square root.
    shifted_root(y) = sqrt(y + 5.0)
    result = @call_message_update_rule(
        node = DeltaFn, target = (:in, 1), algorithm = DeltaApproximation(method = GenUnscented(), inverse = shifted_root),
        m = (out = GammaShapeScale(9.0, 1.0), in = (nothing,)),
    )
    m, v = mean_var(getresult(result))
    @test m ≈ sqrt(14.0) rtol = 1.0e-2
    @test v > 0
end

@testitem "rules:Delta:gausslaguerre" tags = [:rules] setup = [DeltaTestNode] begin
    using DeltaMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesApproximations, ExponentialFamily, Distributions, BayesBase
    using .DeltaTestNode: context
    quadrature = DeltaApproximation(method = GaussLaguerreQuadrature(8))

    square(x) = x^2
    @test_message_update_rule(
        node = DeltaFn{typeof(square)}, target = :out, algorithm = quadrature, check_type_promotion = false, atol = 1.0e-8,
        cases = [
            (m = (in = (GammaShapeScale(2.0, 1.5),),), ctx = context(square)) => NormalMeanVariance(13.5, 425.25),
            (m = (in = (GammaShapeRate(2.0, 1 / 1.5),),), ctx = context(square)) => NormalMeanVariance(13.5, 425.25),
            # An exponential of rate 2: x² has mean 2 / λ² and variance 24 / λ⁴ - (2 / λ²)².
            (m = (in = (Exponential(0.5),),), ctx = context(square)) => NormalMeanVariance(0.5, 1.5 - 0.25),
        ],
    )

    # The joint: a gamma reweighted by N(y | x, 1/τ)… here by a likelihood e^(-βx), which keeps it a
    # gamma, of rate 1/θ + β; the moment match is then exact.
    β = 0.5
    linear_out = ExponentialFamily.Exponential(1 / β)   # logpdf(out, x) = log β - βx
    @test_marginal_update_rule(
        node = DeltaFn{typeof(identity)}, target = (:in,), algorithm = DeltaApproximation(method = GaussLaguerreQuadrature(20)), check_type_promotion = false, atol = 1.0e-8,
        cases = [
            (m = (out = linear_out, in = (GammaShapeScale(3.0, 2.0),)), ctx = context(identity)) => FactorizedJoint((GammaShapeScale(3.0, 1 / (1 / 2.0 + β)),)),
        ],
    )

    # Towards the input: the joint's gamma divided by the input's message.
    @test_message_update_rule(
        node = DeltaFn, target = (:in, 1), algorithm = quadrature, check_type_promotion = false,
        cases = [
            (m = (in = (GammaShapeRate(3.0, 0.5),),), clusters = ((:in,) => FactorizedJoint((GammaShapeRate(3.0, 1.0),)),)) => GammaShapeRate(1.0, 0.5),
        ],
    )

    @test_throws ArgumentError DeltaApproximation(method = GaussLaguerreQuadrature(8), inverse = identity)
end
