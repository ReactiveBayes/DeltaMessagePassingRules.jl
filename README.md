# DeltaMessagePassingRules

[![CI](https://github.com/ReactiveBayes/DeltaMessagePassingRules.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ReactiveBayes/DeltaMessagePassingRules.jl/actions/workflows/CI.yml)
[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://reactivebayes.github.io/DeltaMessagePassingRules.jl/stable/)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://reactivebayes.github.io/DeltaMessagePassingRules.jl/dev/)
[![Coverage](https://codecov.io/gh/ReactiveBayes/DeltaMessagePassingRules.jl/graph/badge.svg)](https://codecov.io/gh/ReactiveBayes/DeltaMessagePassingRules.jl)
[![Code style: Runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

The Delta node, `out = f(in₁, …, inₙ)` for any deterministic function `f`, and its message
passing rules, for the ReactiveMP engine. The rules approximate the pushforward through `f` by a
method the model names in `DeltaApproximation`:

- `Unscented()` and `Linearization()`, from MessagePassingRulesApproximations, for normal
  messages;
- `CVIProjection()`, sampling and projection onto an exponential family, for other families; its
  rules are a package extension, loaded with `using ExponentialFamilyProjection`.

A known inverse of `f`, when given, sends the messages towards the inputs through it.

```julia
using DeltaMessagePassingRules, MessagePassingRulesBase, MessagePassingRulesApproximations, ExponentialFamily

algorithm = DeltaApproximation(method = Linearization(), inverse = y -> (y - 1) / 2)
result = @call_message_update_rule(
    node = DeltaFn, target = (:in, 1), algorithm = algorithm,
    m = (out = NormalMeanVariance(3.0, 2.0), in = (nothing,)),
)
getresult(result)   # NormalMeanVariance(μ = 1.0, v = 0.5)
```

```julia
import Pkg; Pkg.add("DeltaMessagePassingRules")
```

- Documentation: <https://reactivebayes.github.io/DeltaMessagePassingRules.jl/stable/>.
  `make docs` builds it locally, into `docs/build`.
- Tests: `make test` runs the suite as CI does, ending with the rule-coverage gate: every rule
  must be selected by some test. `make test test_args="name:rules"` selects items by tag, by
  name or by file; `make help` lists the other targets.
- Depends on MessagePassingRulesBase, MessagePassingRulesApproximations, BayesBase,
  ExponentialFamily and Distributions; ExponentialFamilyProjection is a weak dependency, for
  `CVIProjection`. Julia 1.10 or later. MIT licence.
