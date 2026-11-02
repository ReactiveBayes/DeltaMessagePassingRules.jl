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

### Changed

- ExpectationApproximations, DomainSets and Compat are dependencies; BayesBase 1.7 and
  MessagePassingRulesBase 1.2 are required.

## [1.0.0] - 2026-10-12

The first release: the Delta node, a deterministic function of its inputs, under linearisation, the unscented transform or CVI projection, developed in the
[ReactiveMP](https://github.com/ReactiveBayes/ReactiveMP.jl) repository, whose history this
repository keeps.

[Unreleased]: https://github.com/ReactiveBayes/DeltaMessagePassingRules.jl/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ReactiveBayes/DeltaMessagePassingRules.jl/releases/tag/v1.0.0
