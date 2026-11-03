# Changelog

All notable changes to DeltaMessagePassingRules.jl are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `LaplaceForm` and its projection `laplace_form(form, density, previous = nothing)`, the form
  every belief takes in universal inference (U4): a closed form passes, a finite discrete product
  is normalised exactly, and any other product is Laplace's approximation in the unconstrained
  coordinates of its domain, through ExpectationApproximations. It reports the log of the
  product's normaliser, and starts at the previous mode when given one.

- `ExpectationRuleFallback`, a rule fallback whose message from a stochastic node is the expected
  log-density `x ↦ E_q[log f(x, ·)]` over every other input, through the rule context's
  expectation backend: a categorical summed exactly, a Gaussian's variance kept. With every other
  input a point mass the message is exact, its log scale known up to the normaliser that
  `LaplaceForm` computes. Towards `out` of a node that is a distribution, the message has its
  support.
- `ExpectationRuleFallback` on a deterministic node, such as a `DeltaFn` without an approximation
  method: towards `out` the moment-matched pushforward of the inputs' messages, a Gaussian, or a
  point mass for point masses; towards an input the message on `out` pulled back through the
  function, in expectation over the other inputs, exact for a single one.
- `ExpectationRuleFallback` for joint marginals and average energies, so the Bethe free energy is
  available on the universal path under belief propagation as well as mean field: a cluster's
  joint is the joint Laplace approximation of the node's log-density times its members' messages,
  in the product of their domains, and the joint over a deterministic node's inputs that of the
  message on `out` at the function times theirs; an average energy is `-E_q[log f]` under the
  clusters' marginals, joints and groups included. Members of a joint are numbers.

- `laplace_form` reports how it projected, as `method`: `:closed_form`, `:point_mass`, `:discrete`
  or `:laplace`, the record an approximation report reads (U5).
- The hint of a Delta node without an approximation method names RxInfer's
  `fallback = LaplaceApproximation()`, which approximates every node without rules.
- `VariationalForm` and its projection `variational_form`, `LaplaceForm`'s sibling (U7b): the
  Gaussian that minimises `KL(q ‖ p)` of a belief with no closed form, in the unconstrained
  coordinates of its domain, through ExpectationApproximations' `gaussian_variational`, with the
  evidence lower bound as the log of the normaliser. The fixed point of conjugate-computation
  variational inference; under mean field, the optimal Gaussian. Closed forms, point masses and
  finite discrete products are projected as by `LaplaceForm`, and `nomode` works the same way.
- `LaplaceForm(nomode = :leave)` leaves a product it finds no mode for, where Newton's iterations
  do not converge, as it is: `laplace_form` reports the method `:unprojected`. `:project`, the
  default, keeps the Gaussian at the last iterate. A message is better left, so that an exact rule
  downstream is not fed a Gaussian that says almost nothing.
- `LaplaceForm`'s Gaussian on the real line is ExponentialFamily's, `NormalMeanVariance` or
  `MvNormalMeanCovariance`, the types the rules of the Gaussian nodes take: the average energy of
  `m ~ MvNormalMeanCovariance(μ, Σ)` under a Laplace posterior of `m` failed on Distributions'
  `MvNormal`, which has no `ndims`.
- The joint Laplace approximation of a cluster is ExponentialFamily's Gaussian on the real line
  too, and `ExpectationRuleFallback` says how it went, `with_outcome(joint, (method =
  :joint_laplace, converged, iterations))`, for the engine's events and an inference report.

### Changed

- ExpectationApproximations, DomainSets and Compat are dependencies; BayesBase 1.7 and
  MessagePassingRulesBase 1.2 are required.

## [1.0.0] - 2026-10-12

The first release: the Delta node, a deterministic function of its inputs, under linearisation, the unscented transform or CVI projection, developed in the
[ReactiveMP](https://github.com/ReactiveBayes/ReactiveMP.jl) repository, whose history this
repository keeps.

[Unreleased]: https://github.com/ReactiveBayes/DeltaMessagePassingRules.jl/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ReactiveBayes/DeltaMessagePassingRules.jl/releases/tag/v1.0.0
