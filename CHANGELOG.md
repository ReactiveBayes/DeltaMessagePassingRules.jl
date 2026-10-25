# Changelog

All notable changes to DeltaMessagePassingRules.jl are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0]

### Added

- Gauss–Hermite cubature, `GaussHermiteCubature(p)`, for normal messages, with every rule the
  unscented transform has.
- The generalized unscented transform, `GenUnscented()` and `GenConstrainedUnscented(lower,
  upper)`: towards `out` and through a known inverse for messages of any univariate family, by
  their skewness and kurtosis, and for normal ones the joint over the inputs.
- Gauss–Laguerre quadrature, `GaussLaguerreQuadrature(p)`, for a node with one input whose message
  is a gamma or an exponential: the message towards `out`, the joint over the input, matched to a
  gamma, and the message towards the input.

### Changed

- The Gaussian rules keep the method's workspace from MessagePassingRulesApproximations 1.1 as
  their scratch, so a call allocates only its message, and the linearisation's forward
  statistics come from MessagePassingRulesApproximations.

## [1.0.0] - 2026-10-12

The first release: the Delta node, a deterministic function of its inputs, under linearisation, the unscented transform or CVI projection, developed in the
[ReactiveMP](https://github.com/ReactiveBayes/ReactiveMP.jl) repository, whose history this
repository keeps.

[Unreleased]: https://github.com/ReactiveBayes/DeltaMessagePassingRules.jl/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ReactiveBayes/DeltaMessagePassingRules.jl/releases/tag/v1.0.0
