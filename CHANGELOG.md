# Changelog

All notable changes to DeltaMessagePassingRules.jl are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

The parts of universal inference, for nodes without rules and beliefs without a closed form.
RxInfer's fallback schemes (`LocalLaplace()`, `LaplacePropagation()`, `ExpectationPropagation()`,
`NaturalGradientPropagation()`) combine them. All are public and not exported, and the new
documentation page "Universal inference" describes them.

- `ExpectationRuleFallback`, a rule fallback that computes the messages, joint marginals and
  average energies of a node from its log-density, using the expectation backend of the rule
  context:
  - From a stochastic node, the message is the expected log-density `x ↦ E_q[log f(x, ·)]` over
    all other inputs. A categorical is summed exactly, and the variance of a Gaussian is kept.
    When all other inputs are point masses, the message is exact, and its log scale is known up to
    the normaliser, which a projection computes. Towards `out` of a node that is a distribution,
    the message has the support of that distribution.
  - From a deterministic node, such as a `DeltaFn` without an approximation method, the message
    towards `out` is the moment-matched pushforward of the input messages: a Gaussian, or a point
    mass when the inputs are point masses. Towards an input, it is the message on `out` pulled
    back through the function, in expectation over the other inputs, and exact for a single
    input. A message on `out` in weighted-mean form, which may be improper, is pulled back as its
    kernel `ξᵀz - zᵀWz/2`.
  - The joint marginal of a cluster is the joint Laplace approximation of the node's log-density
    times the messages of its members, in the product of their domains, with ExponentialFamily's
    Gaussian on the real line. Known members, data or constants, are separate blocks in a
    `FactorizedCluster`. A single remaining member is its own marginal, in closed form where the
    node has the rule. The one vector input of a Delta node, with any constant inputs as blocks
    of their own, as `θ` in `dot(x, θ)`, is its own joint, also by Laplace's method. The fallback reports how the joint was computed with
    `with_outcome(joint, (method = :joint_laplace, converged, iterations))`, for the engine's
    events and an inference report.
  - The average energy is `-E_q[log f]` under the marginals of the clusters, including joints and
    groups, so the Bethe free energy is available under both belief propagation and mean field.
  - Towards an input of a stochastic node whose other edges carry an unprojected raw message, such
    as a product of likelihoods with no mode left by `nomode = :leave`, the message is belief
    propagation's `TiltedJointMessage`, the integral taken by Laplace's method, since no point set
    exists for an expectation over such a message.
- `LaplaceForm` and its projection `laplace_form(form, density, previous = nothing)`. A closed form
  is returned as it is, and a finite discrete product is normalised exactly. Any other product is
  replaced by Laplace's approximation in the unconstrained coordinates of its domain, computed by
  ExpectationApproximations, with ExponentialFamily's `NormalMeanVariance` or
  `MvNormalMeanCovariance` on the real line. The projection returns the log of the product's
  normaliser and the way it projected, `method`: `:closed_form`, `:point_mass`, `:discrete`,
  `:laplace`, or `:unprojected`. The last is used when `nomode = :leave` and no mode is found for
  a product, or the product does not tell its dimension, so that an exact rule downstream does
  not receive a Gaussian that carries almost no information. Given a previous mode, the
  projection starts there. A linearised product of many messages of one type is one factor, so
  the projection of a variable is compiled once for any number of messages.
- `VariationalForm` and `variational_form`, the variational counterpart of `LaplaceForm`. It
  projects a belief onto the Gaussian that minimises `KL(q ‖ p)`, in the same coordinates, with
  ExpectationApproximations' `gaussian_variational`, and returns the evidence lower bound as the
  log of the normaliser. This Gaussian is the fixed point of conjugate-computation variational
  inference and the optimal Gaussian under mean field. Natural-gradient propagation uses it to
  project the beliefs that have no sites.
- `MomentForm(; tol, maxiters, nomode, points, backend)` and `moment_form`, the counterpart for
  expectation propagation. It projects onto the Gaussian with the moments of the product, with
  ExpectationApproximations' `moment_matching`, which places its points on the Laplace
  approximation of the product. The quadrature is `GaussHermite(points)`, built once, or any
  `backend` that gives a point set for a Gaussian, such as `SphericalRadialCubature()` with 2d
  points in d dimensions.
- Cavity propagation, where a variable keeps one site per edge:
  - `site(form, message, cavity, previous = nothing)` computes the site
    `proj(message · cavity) / cavity` with a `LaplaceForm` (Laplace propagation) or a `MomentForm`
    (expectation propagation). A closed message is its own site, and if the cavity is not a
    Gaussian, the message passes. The cavity enters through its natural parameters, so an
    improper cavity, singular or indefinite, is handled, and the site is an
    `(Mv)NormalWeightedMeanPrecision` that may be improper. Properness is not checked. A raw
    message whose cavity has not formed passes as it is, with the outcome `:waiting`, and a
    cavity that is an unprojected product of raw messages is replaced by its Laplace
    approximation where that is a Gaussian. A projection that stops somewhere other than a
    maximum of the tilted density, as for one with no mode, gives no site (`nothing`, the edge
    keeps its last one), and one that runs out of iterations at a maximum gives its site with
    `converged = false`.
  - `ExpectationRuleFallback(propagation)` is the fallback under a propagation scheme. Its
    free-energy term is that of expectation propagation, `cavity_energy`,
    `-log ∫ f ∏ c + Σ E_q[log c]`, which needs the node's inbound messages as well as the
    marginals. The integral is taken in the unconstrained coordinates of each variable, read from
    its marginal, with the Jacobian, so a positive or bounded variable is never evaluated outside
    its support. The other terms of the free energy stay Bethe's. The sum is not a bound.
  - `PushforwardMessage(f, inputs)` is its unprojected message towards `out` of a deterministic
    node, the pushforward. Its site is the projection of the inputs' tilted density against the
    cavity of the output. Through a monotone function of one input, its domain is the image of the
    input's support, such as the positive numbers through `exp`.
  - `TiltedJointMessage` is its message towards one input of a node with several random inputs:
    the belief-propagation message, the factor integrated against the messages on the other
    edges. Its site is the target's block of the joint projection. If another input's message is
    not a Gaussian, or comes as a mean-field marginal, the message is the expected log-density.
  - `same_message` and `same_density` compare raw messages and Gaussians up to a tolerance, so
    that a site does not need to be projected again.
- Natural-gradient propagation (Lukashchuk et al., *Information Geometry of Message Passing*):
  - `NaturalGradientForm(; alpha = 0.5, beta = 0.2, max_step = Inf, tol = 1e-8, backend = nothing)`
    is a third site form. `site(::NaturalGradientForm, message, cavity, previous)` takes a damped
    heavy-ball step from the previous site of the edge, `previous.η` and `previous.velocity`,
    towards the natural-gradient target of the message, halved until the marginal is proper. The
    outcome is `(; method = :natural_gradient, converged, iterations, η, velocity)`, or
    `method = :nonfinite` where the target is not finite and the site stays. Where no halving
    keeps the marginal proper, the outcome holds the parameters of the site the edge keeps.
  - Its sites are in the family of the cavity, which can be any exponential family: a Gaussian of
    any dimension in weighted-mean form, a Gamma as `GammaShapeRate`, and another continuous
    univariate family, such as a Beta, as an `ExponentialFamilyDistribution`. A closed message of
    the family is its own site, and if the cavity is in no family, the message passes.
  - `natural_gradient_target(message, q, backend)` computes `G(q)⁻¹ Cov_q[T, ℓ]` in
    ExponentialFamily's natural parameters of the family of `q`. For a Gaussian it uses Stein's
    lemma, with the values of the log-density only. For other families it uses the definition,
    with ExponentialFamily's sufficient statistics and Fisher information, and by default
    ExpectationApproximations' `UnconstrainedLegendre`, Gauss–Legendre panels in the
    unconstrained coordinate of the domain, accurate for Gamma shapes and Beta parameters from
    0.05, those below 1 included. Packages can add closed forms with methods for their message
    types.
  - With ClosedFormExpectations 1.0 loaded (a package extension), the target has a closed form
    wherever ClosedFormExpectations has one for the gradient `BayesBase.expectation_gradient`, at a
    univariate Gaussian or Gamma belief, for a message whose log-density is an integrand it
    knows. Examples are a log-normal or a Laplace message, `Base.Fix1(logpdf, p)`, and a Gamma
    message pulled back through `exp`, ClosedFormExpectations' `LogGammaKernel`.
    `closed_form_integrand(message)` returns the integrand for a message type, and
    `closed_form_target(ℓ, q)` returns the closed form, or `nothing` if there is none.
  - `PushforwardMessage` has a log-density: the exact density of the pushforward for a monotone
    function of a single input, and the density of its moment-matched Gaussian otherwise. It
    takes the backend of that moment matching as a third field.
  - `laplace_form`, `variational_form` and `moment_form` return a proper
    `ExponentialFamilyDistribution`, such as a `Beta`, unchanged, as a closed form.
- `ExpectationRuleFallback(propagation; energy)` selects the free-energy term of a node without
  rules: `:cavity`, the term of expectation propagation and the default under a `LaplaceForm` or
  a `MomentForm`, or `:average`, the average energy and the default under a
  `NaturalGradientForm` and without sites. A `NaturalGradientForm` with `energy = :cavity` is an
  `ArgumentError`.
- `is_closed_form`, `GaussianMessage`, `family_parameters`, `family_site` and
  `weighted_mean_precision` are public, for packages that add messages and site policies.
- The hint for a Delta node without an approximation method names RxInfer's
  `fallback = LocalLaplace()`, which approximates all nodes without rules.

### Changed

- ExpectationApproximations, DomainSets, Compat, ForwardDiff and LinearAlgebra are dependencies;
  BayesBase 1.7 and MessagePassingRulesBase 1.2 are required.
- The errors and hints for a Delta node without a method no longer suggest `CVIProjection()`.
  For messages that are not Gaussian, they name RxInfer's `fallback = NaturalGradientPropagation()`.

### Deprecated

- `CVIProjection`: its constructors warn. They are the ones it had, with and without its type
  parameters, positional and by keyword. It is replaced by natural-gradient propagation,
  RxInfer's `infer(...; fallback = NaturalGradientPropagation())`, which needs no method on the
  Delta node and does not sample. `CVIProjection` will be removed in the next breaking release.

## [1.0.0] - 2026-10-12

The first release: the Delta node, a deterministic function of its inputs, under linearisation, the unscented transform or CVI projection, developed in the
[ReactiveMP](https://github.com/ReactiveBayes/ReactiveMP.jl) repository, whose history this
repository keeps.

[Unreleased]: https://github.com/ReactiveBayes/DeltaMessagePassingRules.jl/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ReactiveBayes/DeltaMessagePassingRules.jl/releases/tag/v1.0.0
