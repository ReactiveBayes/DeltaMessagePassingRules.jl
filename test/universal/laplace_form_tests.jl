# `laplace_form`, the projection of `LaplaceForm`: closed forms pass, finite discrete products are
# normalised exactly, and everything else is Laplace's approximation in the unconstrained
# coordinates of its domain.

@testitem "laplace_form: a closed form passes as it is" tags = [:universal] begin
    using DeltaMessagePassingRules, BayesBase, Distributions, ExponentialFamily
    import DeltaMessagePassingRules: LaplaceForm, laplace_form
    for d in (NormalMeanVariance(1.0, 2.0), Gamma(2.0, 1.0), Categorical([0.5, 0.5]), PointMass(1.0), missing)
        result = laplace_form(LaplaceForm(), d)
        @test result.distribution === d && result.logZ === nothing
    end
end

@testitem "laplace_form: a finite discrete product is normalised exactly" tags = [:universal] begin
    using DeltaMessagePassingRules, BayesBase, Distributions
    using MessagePassingRulesBase: NodeFunctionLogPdf
    import DeltaMessagePassingRules: LaplaceForm, laplace_form
    weights = [0.2, 0.3, 0.5] .* [1.0, 2.0, 3.0]
    message = NodeFunctionLogPdf(k -> log([1.0, 2.0, 3.0][k]))
    for product in (ProductOf(Categorical([0.2, 0.3, 0.5]), message), LinearizedProductOf([message, Categorical([0.2, 0.3, 0.5])], 2))
        result = laplace_form(LaplaceForm(), product)
        @test result.distribution isa Categorical
        @test probs(result.distribution) ≈ weights ./ sum(weights)
        @test result.logZ ≈ log(sum(weights))
    end
end

@testitem "laplace_form: Laplace in the domain of the product" tags = [:universal] begin
    using DeltaMessagePassingRules, BayesBase, Distributions, ExponentialFamily
    using MessagePassingRulesBase: NodeFunctionLogPdf
    import DeltaMessagePassingRules: LaplaceForm, laplace_form
    import ExpectationApproximations as EA

    # The real line: a Gaussian prior and a Gaussian likelihood, where Laplace is exact.
    product = ProductOf(NormalMeanVariance(0.0, 1.0), NodeFunctionLogPdf(x -> -(2.0 - x)^2 / (2 * 0.5)))
    result = laplace_form(LaplaceForm(), product)
    @test mean(result.distribution) ≈ 4 / 3 && var(result.distribution) ≈ 1 / 3
    @test result.logZ ≈ EA.laplace(x -> logpdf(product, x), 0.0).logZ
    @test result.converged

    # The half-line: Gamma(2, 1) times x³e⁻²ˣ is Gamma(5, rate 3), in log coordinates
    # N(log(5/3), 1/5).
    product = ProductOf(Gamma(2.0, 1.0), NodeFunctionLogPdf(x -> 3 * log(x) - 2 * x))
    result = laplace_form(LaplaceForm(), product)
    @test result.distribution isa EA.TransformedGaussian && result.distribution.domain === EA.Positive()
    @test mean(result.distribution.gaussian) ≈ log(5 / 3) && std(result.distribution.gaussian) ≈ 1 / sqrt(5)
    @test result.logZ ≈ EA.laplace(x -> logpdf(product, x), 2.0, EA.Positive()).logZ

    # The unit interval: Beta(2, 2) times p³(1 - p)⁵, against the same Laplace by hand.
    product = ProductOf(Beta(2.0, 2.0), NodeFunctionLogPdf(p -> 3 * log(p) + 5 * log1p(-p)))
    result = laplace_form(LaplaceForm(), product)
    reference = EA.laplace(x -> logpdf(product, x), 0.5, EA.UnitInterval())
    @test result.distribution isa EA.TransformedGaussian
    @test mean(result.distribution.gaussian) ≈ mean(reference.distribution.gaussian)
    @test std(result.distribution.gaussian) ≈ std(reference.distribution.gaussian)
    @test result.mode ≈ reference.mode && result.logZ ≈ reference.logZ

    # A vector: an MvNormal prior and a likelihood in both components.
    prior = MvNormalMeanCovariance([0.0, 0.0], [1.0 0.0; 0.0 1.0])
    product = ProductOf(prior, NodeFunctionLogPdf(x -> -sum(abs2, x .- [1.0, -1.0]) / 2))
    result = laplace_form(LaplaceForm(), product)
    @test mean(result.distribution) ≈ [0.5, -0.5] && cov(result.distribution) ≈ [0.5 0.0; 0.0 0.5]
end

@testitem "laplace_form: nested products, and a point mass among the factors" tags = [:universal] begin
    using DeltaMessagePassingRules, BayesBase, Distributions, ExponentialFamily
    using MessagePassingRulesBase: NodeFunctionLogPdf
    import DeltaMessagePassingRules: LaplaceForm, laplace_form

    likelihood = NodeFunctionLogPdf(x -> -(1.0 - x)^2 / 2)
    nested = ProductOf(ProductOf(NormalMeanVariance(0.0, 1.0), likelihood), LinearizedProductOf([likelihood, likelihood, likelihood], 2))
    flat = ProductOf(NormalMeanVariance(0.0, 1.0), NodeFunctionLogPdf(x -> -3 * (1.0 - x)^2 / 2))
    @test mean(laplace_form(LaplaceForm(), nested).distribution) ≈ mean(laplace_form(LaplaceForm(), flat).distribution)
    @test var(laplace_form(LaplaceForm(), nested).distribution) ≈ 1 / 4

    # A point mass among the factors is the product; its normaliser is the others at the point.
    result = laplace_form(LaplaceForm(), ProductOf(likelihood, PointMass(1.5)))
    @test result.distribution === PointMass(1.5) && result.logZ ≈ -0.125
end

@testitem "laplace_form: where it starts, and the warm start" tags = [:universal] begin
    using DeltaMessagePassingRules, BayesBase, Distributions, ExponentialFamily
    using MessagePassingRulesBase: NodeFunctionLogPdf
    import DeltaMessagePassingRules: LaplaceForm, laplace_form
    DomainSets = BayesBase.DomainSets

    product = ProductOf(NormalMeanVariance(0.0, 1.0), NodeFunctionLogPdf(x -> -(2.0 - x)^2 / (2 * 0.5) + sin(3x)))
    cold = laplace_form(LaplaceForm(), product)
    warm = laplace_form(LaplaceForm(), product, cold.mode)
    @test warm.mode ≈ cold.mode && warm.iterations < cold.iterations

    # A log-density with a domain and nothing else starts inside it.
    alone = ContinuousUnivariateLogPdf(DomainSets.HalfLine(), x -> 3 * log(x) - 2 * x)
    @test laplace_form(LaplaceForm(), alone).distribution.domain === DeltaMessagePassingRules.EA.Positive()
    # One with neither a domain nor a dimension is of a number, a likelihood's message alone.
    likelihood = NodeFunctionLogPdf(x -> -(2.0 - x)^2 / 2)
    @test mean(laplace_form(LaplaceForm(), ProductOf(likelihood, likelihood)).distribution) ≈ 2.0
    @test mean(laplace_form(LaplaceForm(), likelihood).distribution) ≈ 2.0
    # One that does not take a number needs a dimension from somewhere.
    vector = NodeFunctionLogPdf(x -> -(x[1] - 1.0)^2 / 2 - (x[2] - 2.0)^2 / 2)
    @test_throws "dimension" laplace_form(LaplaceForm(), vector)
    @test mean(laplace_form(LaplaceForm(), vector, [0.0, 0.0]).distribution) ≈ [1.0, 2.0]
end
