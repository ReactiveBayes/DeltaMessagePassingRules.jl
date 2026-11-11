```@meta
CurrentModule = DeltaMessagePassingRules
```

# [Universal inference](@id delta-universal)

Message passing needs two things a model may not have: a rule for every message a node sends,
and a closed form for every belief, the product of the messages a variable receives. Where a node
has no rule, a *rule fallback* gives its messages; where a belief has no closed form, a *form
constraint* projects it onto one. This package gives a fallback and three projections that need
nothing written per node or per variable. They compute everything from the nodes' log-densities
and the variables' domains, with the quadrature and the Gaussian approximations of
[ExpectationApproximations](https://github.com/ReactiveBayes/ExpectationApproximations.jl).

In RxInfer, one keyword chooses how they are combined, a *fallback scheme*:

```julia
using RxInfer

infer(model = model(), data = data, fallback = LocalLaplace())
```

| RxInfer's scheme | The rule fallback | The projection | What it is |
|---|---|---|---|
| `LocalLaplace()` | [`ExpectationRuleFallback`](@ref)`()` | [`LaplaceForm`](@ref), on every belief | Laplace's approximation of each belief on its own, at its mode |
| `CVI()` | [`ExpectationRuleFallback`](@ref)`()` | [`VariationalForm`](@ref), on every belief | the Gaussian closest to each belief, the fixed point of conjugate-computation variational inference |
| `LaplacePropagation()` | [`ExpectationRuleFallback`](@ref)`(LaplaceForm())` | [`LaplaceForm`](@ref), on each edge's site | Laplace propagation, see [Cavity propagation](@ref delta-universal-cavity) |
| `ExpectationPropagation()` | [`ExpectationRuleFallback`](@ref)`(MomentForm())` | [`MomentForm`](@ref), on each edge's site | expectation propagation, see [Cavity propagation](@ref delta-universal-cavity) |

The scheme also sets the expectation backend the fallback computes with, by default the
backends loaded in the session, such as ExpectationApproximations' `EADefault()`. A node with a
rule, and a belief with a closed form, are never touched: a conjugate model runs as without a
scheme.

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

## The projections

A Gamma prior times a likelihood has no closed form here. Each projection works in the
unconstrained coordinates of the belief's domain, read from the prior's support: here `log x`,
so each result is a log-normal distribution, with the log of the product's normaliser. Laplace's
approximation is the Gaussian at the mode:

```@example universal
using MessagePassingRulesBase: NodeFunctionLogPdf

product = ProductOf(GammaShapeRate(2.0, 1.0), NodeFunctionLogPdf(x -> 3 * log(x) - 2 * x))
result = DeltaMessagePassingRules.laplace_form(DeltaMessagePassingRules.LaplaceForm(), product)
result.distribution.domain, mean(result.distribution.gaussian), result.logZ
```

The variational Gaussian minimises `KL(q ‖ p)` instead. Its log normaliser is the evidence lower
bound, below the exact `log Γ(5) - 5 log 3 ≈ -2.315` of this product; for this density, a gamma
in `log x`, it happens to equal Laplace's estimate, while its location sits below the mode:

```@example universal
variational = DeltaMessagePassingRules.variational_form(DeltaMessagePassingRules.VariationalForm(), product)
variational.method, mean(variational.distribution.gaussian), variational.logZ
```

Moment matching gives the Gaussian with the product's own mean and variance in those
coordinates, and the product's integral, both by quadrature; here the integral is the exact one:

```@example universal
moments = DeltaMessagePassingRules.moment_form(DeltaMessagePassingRules.MomentForm(), product)
moments.method, mean(moments.distribution.gaussian), moments.logZ
```

```@docs
ExpectationRuleFallback
LaplaceForm
DeltaMessagePassingRules.laplace_form
VariationalForm
DeltaMessagePassingRules.variational_form
MomentForm
DeltaMessagePassingRules.moment_form
DeltaMessagePassingRules.ExpectedLogDensity
```

## [Cavity propagation](@id delta-universal-cavity)

### Why projecting each belief is not enough

Under `LocalLaplace()` and `CVI()`, every product with no closed form is projected on its own. A
variable with `K` connections has `K + 1` such products, its marginal and its `K` outbound
messages, and each holds the same raw message from a likelihood. That message is approximated
`K + 1` times, separately, and the approximations need not agree: the marginal and the messages
next to it describe slightly different beliefs, and the Bethe free energy computed from them is
not a bound. Worse, a message pulled back through a nonlinear function, such as an observation of
`sin(θ)` sent towards `θ`, may have no Gaussian of its own to be projected onto: on its own it
says nothing about some directions.

### Sites and cavities

Cavity propagation approximates each raw message once, where it matters. Every random variable
keeps one *site* `s_k` per edge, a Gaussian. On an edge whose message `m_k` is already a Gaussian,
the site is the message. On an edge with a raw message, the site is

```math
c_k = \prod_{j \neq k} s_j, \qquad s_k = \frac{\operatorname{proj}(m_k \, c_k)}{c_k},
```

where the *cavity* `c_k`, the product of the other edges' sites, is exactly the message the
variable sends out on edge `k`. The variable then multiplies sites, never raw messages: its
marginal and its outbound messages are products of the same Gaussians, one belief seen from each
edge. Four properties follow.

1. **Exact where belief propagation is exact.** A closed message times a Gaussian cavity is its
   own projection, so its site is the message, and conjugate parts of a model do not change.
2. **The beliefs agree.** At a fixed point, the marginal equals `proj(m_k c_k)` for every raw
   edge `k`.
3. **Each projection holds one raw factor** times a Gaussian, the cavity, which gives it the
   dimension and the scale a pulled-back message lacks on its own.
4. **It is expectation propagation with fully factorised sites** (Minka, 2001). For a factor
   `f(x, y)`, belief propagation's message to `x` is `∫ f(x, y) c_y(y) dy`, and its product with
   `c_x` is the `x`-marginal of the tilted density `c_x(x) c_y(y) f(x, y)`; projecting it is EP's
   update for `f ≈ s_x(x) s_y(y)`. With moment matching ([`MomentForm`](@ref)) this is
   expectation propagation; with Laplace's method ([`LaplaceForm`](@ref)) it is Laplace
   propagation (Smola, Vishwanathan and Eskin), whose fixed point on a chain is the global
   Laplace approximation.

[`site`](@ref) computes one site. Here a Poisson observation of a log-rate, a raw message, meets a
Gaussian cavity. Laplace propagation's site and expectation propagation's differ, since the
tilted density is skewed:

```@example universal
raw = NodeFunctionLogPdf(x -> 3x - exp(x))          # y = 3 counts, log-rate x
cavity = NormalMeanVariance(0.0, 1.0)
laplace = DeltaMessagePassingRules.site(DeltaMessagePassingRules.LaplaceForm(), raw, cavity)
ep = DeltaMessagePassingRules.site(DeltaMessagePassingRules.MomentForm(), raw, cavity)
laplace.site, ep.site
```

### Where the sites live

The engine keeps the sites at the variables (ReactiveMP's site policy), not in the nodes. Rules
and rule matching do not change, and a node with a hand-written rule that already returns a
Gaussian, such as Probit's own expectation-propagation rule, passes as its own site. The price is
that a site is one Gaussian per variable: a node with several random inputs gets a site at each
input, not one joint site over them, which would need a rule in the node itself.

### The messages the fallback sends

Under a propagation scheme, [`ExpectationRuleFallback`](@ref)`(propagation)` sends messages for the
sites to project, unprojected:

- towards `out` of a deterministic node, the pushforward of its inputs' messages,
  [`PushforwardMessage`](@ref). Its site at the output is the projection of the inputs' tilted
  density against the output's cavity, which needs no input's message to be proper.
- towards one input of a node with several random inputs, belief propagation's message, the
  factor integrated against the other edges' messages, [`TiltedJointMessage`](@ref). Its site is
  the target's block of the joint projection of the tilted density over all the free inputs.

One limit: the integral needs the other inputs' messages to be Gaussians or point masses. An input
whose message is of another family, such as a Gamma precision, or that comes as a mean-field
marginal, keeps the expected log-density `E[log f]`, as without propagation.

A raw message that arrives again unchanged takes its last site without a projection, which
[`same_message`](@ref) and [`same_density`](@ref) decide.

```@docs
DeltaMessagePassingRules.site
DeltaMessagePassingRules.PushforwardMessage
DeltaMessagePassingRules.TiltedJointMessage
DeltaMessagePassingRules.same_message
DeltaMessagePassingRules.same_density
```

### Improper sites

A site, and so a cavity or an outbound message, may be improper: a Gaussian whose precision is
singular or even indefinite. A pulled-back `sin(θ)` observation has a precision near zero along
directions it does not see, and a negative one where `sin` is concave against the cavity. This is
normal for expectation propagation: only the tilted densities need to be proper. Nothing checks,
and nothing is clipped. Every function here uses a site only through its natural parameters, in
weighted-mean form (`NormalWeightedMeanPrecision`, `MvNormalWeightedMeanPrecision`), never through
its moments: through its moments an indefinite precision would be replaced by a positive definite
one, which describes another Gaussian without saying so.

### The free energy

Expectation propagation's free energy splits into terms each node and variable computes from what
it sees. The variables keep their Bethe term, `(d_i - 1) H[q_i]`, and every node with rules keeps
its average energy less its entropy. A node that sends a raw message takes instead
[`cavity_energy`](@ref),

```math
F_a = -\log \int f_a \prod_i c_{ia} + \sum_i \mathbb{E}_{q_i}[\log c_{ia}],
```

from its inbound messages `c_ia`, the cavities, and the variables' marginals `q_i`. The terms add up
to EP's free energy because the sites multiply to the marginals,
`Σ_a η(c_ia) = (d_i - 1) η(q_i)` in natural parameters: moving each `E_{q_i}[log c_ia]` from the
variable's term to the node's leaves every other term as it is. Rescaling a cavity changes neither
part's sum, so an improper cavity enters through its kernel. The integral must be the scheme's own
estimate, Laplace's or the moments' quadrature: with an exact integral under Laplace propagation
the local terms no longer add up to the global free energy (they differed by 0.34 on a Poisson
count series).

At a fixed point the free energy approximates `-log p(y)`, but it is not a bound: a lower value is
not a better approximation, unlike the evidence lower bound of `CVI()`.

```@docs
DeltaMessagePassingRules.cavity_energy
```

### References

- T. P. Minka, *Expectation propagation for approximate Bayesian inference*, UAI 2001; and
  *Divergence measures and message passing*, Microsoft Research technical report, 2005.
- A. J. Smola, S. V. N. Vishwanathan and E. Eskin, *Laplace propagation*, Advances in Neural
  Information Processing Systems 16, 2004.
