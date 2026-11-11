# Changelog

All notable changes to DeltaMessagePassingRules.jl are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

The pieces of universal inference, for nodes without rules and beliefs without a closed form,
which RxInfer's fallback schemes (`LocalLaplace()`, `LaplacePropagation()`,
`ExpectationPropagation()`, `NaturalGradientPropagation()`) combine. All are public and not exported; the new documentation page
"Universal inference" explains them.

- `ExpectationRuleFallback`, a rule fallback that computes a node's messages, joint marginals and
  average energies from its log-density, through the rule context's expectation backend:
  - from a stochastic node, the expected log-density `x ↦ E_q[log f(x, ·)]` over every other
    input: a categorical summed exactly, a Gaussian's variance kept. With every other input a
    point mass the message is exact, its log scale known up to the normaliser a projection
    computes. Towards `out` of a node that is a distribution, the message has its support;
  - from a deterministic node, such as a `DeltaFn` without an approximation method: towards `out`
    the moment-matched pushforward of the inputs' messages, a Gaussian, or a point mass for point
    masses; towards an input the message on `out` pulled back through the function, in
    expectation over the other inputs, exact for a single one. A message on `out` in weighted-mean
    form, which may be improper, pulls back as its kernel `ξᵀz - zᵀWz/2`;
  - a cluster's joint marginal, the joint Laplace approximation of the node's log-density times
    its members' messages, in the product of their domains, ExponentialFamily's Gaussian on the
    real line. Known members, data or constants, are blocks of their own in a
    `FactorizedCluster`; one member left is its own marginal, in closed form where the node has
    the rule. The fallback says how the joint went, `with_outcome(joint, (method = :joint_laplace,
    converged, iterations))`, for the engine's events and an inference report;
  - an average energy, `-E_q[log f]` under the clusters' marginals, joints and groups included, so
    the Bethe free energy is available under belief propagation and mean field alike.
- `LaplaceForm` and its projection `laplace_form(form, density, previous = nothing)`: a closed form
  passes, a finite discrete product is normalised exactly, and any other product is Laplace's
  approximation in the unconstrained coordinates of its domain, through ExpectationApproximations,
  ExponentialFamily's `NormalMeanVariance` or `MvNormalMeanCovariance` on the real line. It
  reports the log of the product's normaliser and how it projected, `method`: `:closed_form`,
  `:point_mass`, `:discrete`, `:laplace`, or `:unprojected` when `nomode = :leave` leaves a
  product it finds no mode for, so that an exact rule downstream is not fed a Gaussian that says
  almost nothing. It starts at the previous mode when given one.
- `VariationalForm` and `variational_form`, `LaplaceForm`'s sibling: the Gaussian that minimises
  `KL(q ‖ p)` of a belief, in the same coordinates, through ExpectationApproximations'
  `gaussian_variational`, with the evidence lower bound as the log of the normaliser. It is the
  fixed point of conjugate-computation variational inference, under mean field the optimal
  Gaussian, and natural-gradient propagation's projection of the beliefs that have no sites.
- `MomentForm(; tol, maxiters, nomode, points, backend)` and `moment_form`, the sibling for
  expectation propagation: the Gaussian with the product's moments, by ExpectationApproximations'
  `moment_matching`, whose points sit on the product's own Laplace approximation. Its quadrature
  is `GaussHermite(points)`, built once, or any `backend` with a point set for a Gaussian, such as
  `SphericalRadialCubature()`, 2d points in d dimensions.
- Cavity propagation, where a variable keeps one site per edge:
  - `site(form, message, cavity, previous = nothing)`, the site `proj(message · cavity) / cavity`
    by a `LaplaceForm` (Laplace propagation) or a `MomentForm` (expectation propagation). A closed
    message is its own site, and against a cavity that is not a Gaussian the message passes. The
    cavity enters through its natural parameters, so an improper one, singular or indefinite,
    takes part, and the site is an `(Mv)NormalWeightedMeanPrecision`, proper or not. Nothing
    checks proper-ness;
  - `ExpectationRuleFallback(propagation)`, the fallback under a propagation scheme. Its free-energy
    term is expectation propagation's, `cavity_energy`, `-log ∫ f ∏ c + Σ E_q[log c]`, given the
    node's inbound messages as well as the marginals; the other terms of the free energy stay
    Bethe's. The sum is not a bound;
  - `PushforwardMessage(f, inputs)`, its message towards `out` of a deterministic node, the
    pushforward unprojected, whose site is the projection of the inputs' tilted density against
    the output's cavity;
  - `TiltedJointMessage`, its message towards one input of a node with several random inputs:
    belief propagation's, the factor integrated against the other edges' messages, whose site is
    the target's block of the joint projection. An input whose message is not a Gaussian, or that
    comes as a mean-field marginal, keeps the expected log-density;
  - `same_message` and `same_density`, which compare raw messages and Gaussians to a tolerance, so
    that a site need not be projected again.
- Natural-gradient propagation (Lukashchuk et al., *Information Geometry of Message Passing*):
  - `NaturalGradientForm(; alpha = 0.5, beta = 0.2, max_step = Inf, tol = 1e-8, backend = nothing)`,
    a third site form: `site(::NaturalGradientForm, message, cavity, previous)` takes a damped
    heavy-ball step from the edge's previous site, `previous.η` and `previous.velocity`, towards
    the message's natural-gradient target, halved until the marginal is proper. The outcome is
    `(; method = :natural_gradient, converged, iterations, η, velocity)`;
  - its sites are in the family of the cavity, any exponential family: a Gaussian of any dimension
    in weighted-mean form, a Gamma as `GammaShapeRate`, and another continuous family with one
    dimension, such as a Beta, as an `ExponentialFamilyDistribution`. A closed message of the
    family is its own site; a cavity in no family lets the message pass;
  - `natural_gradient_target(message, q, backend)`, `G(q)⁻¹ Cov_q[T, ℓ]` in ExponentialFamily's
    natural parameters of `q`'s family: by Stein's lemma for a Gaussian, from the log-density's
    values only, and by the definition with ExponentialFamily's sufficient statistics and Fisher
    information for any other family, with Gauss–Legendre points in the unconstrained coordinate
    of its domain by default. Packages add closed forms by methods for their message types;
  - `PushforwardMessage` has a log-density: the pushforward's own through a monotone function of a
    single input, its moment-matched Gaussian's otherwise. It takes the backend of that
    moment matching as a third field;
  - `laplace_form`, `variational_form` and `moment_form` pass a proper `ExponentialFamilyDistribution`
    as its distribution, such as a `Beta`.
- `ExpectationRuleFallback(propagation; energy)`: the free-energy term of a node without rules,
  `:cavity`, expectation propagation's, the default under a `LaplaceForm` or a `MomentForm`, or
  `:average`, the average energy, the default under a `NaturalGradientForm` and without sites.
- The hint of a Delta node without an approximation method names RxInfer's
  `fallback = LocalLaplace()`, which approximates every node without rules.

### Changed

- ExpectationApproximations, DomainSets and Compat are dependencies; BayesBase 1.7 and
  MessagePassingRulesBase 1.2 are required.

## [1.0.0] - 2026-10-12

The first release: the Delta node, a deterministic function of its inputs, under linearisation, the unscented transform or CVI projection, developed in the
[ReactiveMP](https://github.com/ReactiveBayes/ReactiveMP.jl) repository, whose history this
repository keeps.

[Unreleased]: https://github.com/ReactiveBayes/DeltaMessagePassingRules.jl/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ReactiveBayes/DeltaMessagePassingRules.jl/releases/tag/v1.0.0
