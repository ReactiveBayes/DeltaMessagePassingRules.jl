```@meta
CurrentModule = DeltaMessagePassingRules
```

# [Universal inference](@id delta-universal)

Where a node has no rule, a rule fallback gives its messages; where a belief has no closed form,
a form constraint gives it one. This package gives a pair that needs nothing written per node or
per variable, the first scheme of universal inference: Laplace's approximation in the
unconstrained coordinates of each variable's domain.

- [`ExpectationRuleFallback`](@ref): the message of a stochastic node is its expected
  log-density, `x ↦ E_q[log f(x, ·)]` over every other input, computed with the expectation
  backend of the rule context. A deterministic node, such as a [`DeltaFn`](@ref) without an
  approximation method, sends the moment-matched pushforward of its inputs towards `out`, and the
  message on `out` pulled back through its function towards an input. It gives joint marginals
  and average energies too, so the free energy is available.
- [`LaplaceForm`](@ref): every belief with no closed form, a product of a prior and such
  messages, becomes Laplace's approximation in the coordinates of its domain, read from the
  first factor that has one; a finite discrete product is normalised exactly, and a closed form
  passes. The projection reports the log of the product's normaliser, which completes the log
  scale of a message whose normaliser was unknown.

In RxInfer both are given once, for every node and variable:

```julia
using RxInfer, ExpectationApproximations

infer(
    model = model(),
    data = data,
    options = (
        rulefallback = DeltaMessagePassingRules.ExpectationRuleFallback(),
        default_form_constraint = DeltaMessagePassingRules.LaplaceForm(),
        context = (expectation = ExpectationApproximations.EADefault(),),
    ),
)
```

## A message of a stochastic node

The node below has a log-density and no rules. Its message towards `a`, with `out` observed and
`b` a point mass, is the node's log-density itself, and its log scale is known up to its
normaliser:

```@example universal
using DeltaMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily
import ExpectationApproximations as EA
using MessagePassingRulesBase: RuleArgs, RuleContext, Target, call_fallback, unwrap_fallback

struct Toy{A, B}
    a::A
    b::B
end
BayesBase.logpdf(d::Toy, x) = -(x - d.a)^2 / (2 * d.b)
@define_factor_node(node = Toy, type = Stochastic, interfaces = [:out, :a, :b])

fallback = DeltaMessagePassingRules.ExpectationRuleFallback()
context = RuleContext(expectation = EA.EADefault())
args = RuleArgs(q = (out = PointMass(2.0), b = PointMass(0.5)))
message, logscale = unwrap_fallback(call_fallback(fallback, Toy, Target(:a), nothing, context, args))
logpdf(message, 1.0), logscale
```

With a Gaussian on `out` instead, the message is the expectation over it, and keeps its
variance, which a plug-in of its mean loses:

```@example universal
args = RuleArgs(m = (out = NormalMeanVariance(2.0, 3.0),), q = (b = PointMass(0.5),))
message, _ = unwrap_fallback(call_fallback(fallback, Toy, Target(:a), nothing, context, args))
logpdf(message, 1.0), -((2.0 - 1.0)^2 + 3.0) / (2 * 0.5)
```

## The projection

A Gamma prior times a likelihood has no closed form here; Laplace's approximation in log
coordinates is a log-normal distribution, with the log of the product's normaliser:

```@example universal
using MessagePassingRulesBase: NodeFunctionLogPdf

product = ProductOf(GammaShapeRate(2.0, 1.0), NodeFunctionLogPdf(x -> 3 * log(x) - 2 * x))
result = DeltaMessagePassingRules.laplace_form(DeltaMessagePassingRules.LaplaceForm(), product)
result.distribution.domain, mean(result.distribution.gaussian), result.logZ
```

```@docs
ExpectationRuleFallback
LaplaceForm
DeltaMessagePassingRules.laplace_form
DeltaMessagePassingRules.ExpectedLogDensity
```
