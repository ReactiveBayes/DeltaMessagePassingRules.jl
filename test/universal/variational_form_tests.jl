# `VariationalForm`: LaplaceForm's sibling, the Gaussian minimising KL(q ‖ p) of a product of a
# prior and log-density messages, with the evidence lower bound as its log-normaliser.

@testitem "variational_form: closed forms, point masses and discrete products as laplace_form" tags = [:universal] begin
    using DeltaMessagePassingRules, BayesBase, Distributions, ExponentialFamily
    using MessagePassingRulesBase: NodeFunctionLogPdf
    import DeltaMessagePassingRules: VariationalForm, variational_form, LaplaceForm, laplace_form

    for d in (NormalMeanVariance(1.0, 2.0), Gamma(2.0, 1.0), PointMass(3.0))
        result = variational_form(VariationalForm(), d)
        @test result.distribution === d && result.method === :closed_form && isnothing(result.logZ)
    end
    likelihood = NodeFunctionLogPdf(x -> -(1.0 - x)^2 / 2)
    @test variational_form(VariationalForm(), ProductOf(likelihood, PointMass(1.5))).method === :point_mass
    discrete = ProductOf(Categorical([0.2, 0.8]), NodeFunctionLogPdf(k -> k == 1 ? 0.0 : -1.0))
    @test variational_form(VariationalForm(), discrete).distribution ≈ laplace_form(LaplaceForm(), discrete).distribution
    @test_throws "`nomode` is `:project` or `:leave`" VariationalForm(nomode = :drop)
end

@testitem "variational_form: the Gaussian of a product, on the real line and in a domain" tags = [:universal] begin
    using DeltaMessagePassingRules, BayesBase, Distributions, ExponentialFamily
    using MessagePassingRulesBase: NodeFunctionLogPdf
    import DeltaMessagePassingRules: VariationalForm, variational_form, LaplaceForm, laplace_form
    import ExpectationApproximations as EA

    # Gaussian times Gaussian: the variational Gaussian is the product itself, its bound the evidence
    product = ProductOf(NormalMeanVariance(0.0, 1.0), NodeFunctionLogPdf(x -> -(2.0 - x)^2 / (2 * 0.5)))
    result = variational_form(VariationalForm(), product)
    @test result.distribution isa NormalMeanVariance && result.method === :variational && result.converged
    @test mean(result.distribution) ≈ 4 / 3 && var(result.distribution) ≈ 1 / 3
    @test result.logZ ≈ laplace_form(LaplaceForm(), product).logZ

    # a skewed product: the same as gaussian_variational of its log-density, below Laplace's evidence
    product = ProductOf(NormalMeanVariance(0.0, 4.0), NodeFunctionLogPdf(x -> x - exp(x) / 2))
    result = variational_form(VariationalForm(), product)
    reference = EA.gaussian_variational(x -> logpdf(product, x), 0.0)
    @test mean(result.distribution) ≈ mean(reference.distribution) && var(result.distribution) ≈ var(reference.distribution)
    @test result.logZ ≈ reference.logZ && result.mode ≈ reference.mean

    # Gamma(2, 1) times x³e⁻²ˣ is Gamma(5, rate 3), in log coordinates: a transformed Gaussian
    product = ProductOf(Gamma(2.0, 1.0), NodeFunctionLogPdf(x -> 3 * log(x) - 2 * x))
    result = variational_form(VariationalForm(), product)
    @test result.distribution isa EA.TransformedGaussian && result.distribution.domain === EA.Positive()
    @test mean(result.distribution) ≈ 5 / 3 rtol = 0.02
    # below the exact normaliser, ∫ x e⁻ˣ · x³e⁻²ˣ dx = Γ(5) / 3⁵
    @test result.logZ < log(24 / 243)

    # a vector: an MvNormal prior and a likelihood in both components
    prior = MvNormalMeanCovariance([0.0, 0.0], [1.0 0.0; 0.0 1.0])
    product = ProductOf(prior, NodeFunctionLogPdf(x -> -sum(abs2, x .- [1.0, -1.0]) / 2))
    result = variational_form(VariationalForm(), product)
    @test result.distribution isa MvNormalMeanCovariance
    @test mean(result.distribution) ≈ [0.5, -0.5] && cov(result.distribution) ≈ [0.5 0.0; 0.0 0.5]
end

@testitem "variational_form: a product with no mode, and the backend" tags = [:universal] begin
    using DeltaMessagePassingRules, BayesBase, Distributions, ExponentialFamily
    using MessagePassingRulesBase: NodeFunctionLogPdf
    import DeltaMessagePassingRules: VariationalForm, variational_form
    import ExpectationApproximations as EA

    # log σ(η) alone rises without bound: left unprojected under :leave
    likelihood = NodeFunctionLogPdf(η -> -log1p(exp(-η)))
    left = variational_form(VariationalForm(nomode = :leave), likelihood)
    @test !left.converged && left.method === :unprojected && left.distribution === likelihood

    # a coarser quadrature, given as the backend, changes the answer only a little
    product = ProductOf(NormalMeanVariance(0.0, 4.0), NodeFunctionLogPdf(x -> x - exp(x) / 2))
    fine = variational_form(VariationalForm(), product)
    coarse = variational_form(VariationalForm(backend = EA.GaussHermite(5)), product)
    @test coarse.converged
    @test mean(coarse.distribution) ≈ mean(fine.distribution) rtol = 1.0e-2
    @test mean(coarse.distribution) != mean(fine.distribution)
end
