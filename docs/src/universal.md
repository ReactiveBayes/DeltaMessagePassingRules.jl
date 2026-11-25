```@meta
CurrentModule = DeltaMessagePassingRules
```

# [Universal inference](@id delta-universal)

Message passing needs a rule for each message a node sends, and a closed form for each belief,
the product of the messages a variable receives. When a node has no rule, a *rule fallback*
computes its messages. When a belief has no closed form, a *form constraint* projects it onto
one. This package has a rule fallback and several projections that need no code per node or per
variable. They work from the log-densities of the nodes and the domains of the variables, and use
the quadrature rules and Gaussian approximations of
[ExpectationApproximations](https://github.com/ReactiveBayes/ExpectationApproximations.jl).

In RxInfer, a *fallback scheme* combines them. It is selected with the `fallback` keyword of
`infer`:

```julia
using RxInfer

infer(model = model(), data = data, fallback = LocalLaplace())
```

| Scheme in RxInfer | Rule fallback | Projection | Method |
|---|---|---|---|
| `LocalLaplace()` | [`ExpectationRuleFallback`](@ref)`()` | [`LaplaceForm`](@ref) on all beliefs | Laplace approximation of each belief separately, at its mode |
| `LaplacePropagation()` | [`ExpectationRuleFallback`](@ref)`(LaplaceForm())` | [`LaplaceForm`](@ref) on the site of each edge | Laplace propagation, see [Cavity propagation](@ref delta-universal-cavity) |
| `ExpectationPropagation()` | [`ExpectationRuleFallback`](@ref)`(MomentForm())` | [`MomentForm`](@ref) on the site of each edge | Expectation propagation, see [Cavity propagation](@ref delta-universal-cavity) |
| `NaturalGradientPropagation()` | [`ExpectationRuleFallback`](@ref)`(NaturalGradientForm())` | [`NaturalGradientForm`](@ref) on the site of each edge, in any exponential family; [`VariationalForm`](@ref) on beliefs without sites | Natural-gradient propagation, see [Natural-gradient propagation](@ref delta-universal-ngmp) |

The scheme also sets the expectation backend that the fallback uses. By default these are the
backends loaded in the session, such as `EADefault()` from ExpectationApproximations. Nodes that
have rules and beliefs that have a closed form are not affected, so a conjugate model runs the
same with or without a scheme.

## A message of a stochastic node

The node below has a log-density but no rules. With `out` observed and `b` a point mass, its
message towards `a` is the log-density of the node as a function of `a`. The log scale of the
message is known up to the normaliser of that density:

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

When the message on `out` is a Gaussian instead, the message towards `a` is the expected
log-density over `out`. It depends on the variance of `out`, which would be lost if the mean of
`out` were plugged in:

```@example universal
args = RuleArgs(m = (out = NormalMeanVariance(2.0, 3.0),), q = (b = PointMass(0.5),))
message, _ = unwrap_fallback(call_fallback(fallback, Toy, Target(:a), nothing, context, args))
logpdf(message, 1.0), -((2.0 - 1.0)^2 + 3.0) / (2 * 0.5)
```

## The projections

The product of a Gamma prior and a likelihood given as a log-density has no closed form. Each
projection works in the unconstrained coordinates of the belief's domain, which it reads from the
support of the prior. Here the coordinate is `log x`, so each result is a log-normal
distribution. Each projection also returns the log of the product's normaliser. Laplace's
approximation is the Gaussian at the mode:

```@example universal
using MessagePassingRulesBase: NodeFunctionLogPdf

product = ProductOf(GammaShapeRate(2.0, 1.0), NodeFunctionLogPdf(x -> 3 * log(x) - 2 * x))
result = DeltaMessagePassingRules.laplace_form(DeltaMessagePassingRules.LaplaceForm(), product)
result.distribution.domain, mean(result.distribution.gaussian), result.logZ
```

The variational Gaussian minimises `KL(q ‖ p)` instead. Its log normaliser is the evidence lower
bound, which is below the exact value for this product, `log Γ(5) - 5 log 3 ≈ -2.315`. For this
density, a Gamma seen in the coordinate `log x`, the bound happens to equal Laplace's estimate,
but the mean of the variational Gaussian is below the mode:

```@example universal
variational = DeltaMessagePassingRules.variational_form(DeltaMessagePassingRules.VariationalForm(), product)
variational.method, mean(variational.distribution.gaussian), variational.logZ
```

Moment matching gives the Gaussian with the same mean and variance as the product in those
coordinates. The moments and the integral of the product are computed by quadrature. Here the
integral equals the exact value:

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

### Projecting each belief separately

Under `LocalLaplace()`, each product with no closed form is projected separately. A variable
with `K` connections has `K + 1` such products: its marginal and its `K` outbound messages. The
same raw message from a likelihood is part of these products, and each projection approximates
it again. The approximations need not agree, so the marginal and the neighbouring messages
describe slightly different beliefs, and the Bethe free energy computed from them is not a bound.
There is a second problem. A message pulled back through a nonlinear function, such as an
observation of `sin(θ)` sent towards `θ`, may carry no information about some directions. Such
a message has no Gaussian that it can be projected onto.

### Sites and cavities

Cavity propagation approximates each raw message once. Each random variable keeps one Gaussian
*site* `s_k` per edge. If the message `m_k` on an edge is already a Gaussian, the site is that
message. On an edge with a raw message, the site is

```math
c_k = \prod_{j \neq k} s_j, \qquad s_k = \frac{\operatorname{proj}(m_k \, c_k)}{c_k},
```

The *cavity* `c_k` is the product of the sites on the other edges, and it is the message that
the variable sends out on edge `k`. The variable multiplies sites and not raw messages, so its
marginal and its outbound messages are all products of the same Gaussians. They describe one
belief, seen from different edges.

Cavity propagation is exact where belief propagation is exact. A closed message times a Gaussian
cavity is its own projection, so its site is the message, and the conjugate parts of a model do
not change. At a fixed point the beliefs agree: the marginal equals `proj(m_k c_k)` for each
edge `k` with a raw message. Each projection involves one raw factor times a Gaussian, the
cavity, and the cavity gives it the dimension and the scale that a pulled-back message lacks on
its own.

The method is expectation propagation with fully factorised sites (Minka, 2001). For a factor
`f(x, y)`, the belief-propagation message to `x` is `∫ f(x, y) c_y(y) dy`. Its product with `c_x`
is the `x`-marginal of the tilted density `c_x(x) c_y(y) f(x, y)`, and projecting that marginal
is the EP update for `f ≈ s_x(x) s_y(y)`. With moment matching ([`MomentForm`](@ref)) the
method is expectation propagation. With Laplace's method ([`LaplaceForm`](@ref)) it is Laplace
propagation (Smola, Vishwanathan and Eskin), whose fixed point on a chain is the global Laplace
approximation.

[`site`](@ref) computes a single site. In the example below, the raw message is a Poisson
observation of a log-rate, and the cavity is a Gaussian. The tilted density is skewed, so the
sites of Laplace propagation and expectation propagation differ:

```@example universal
raw = NodeFunctionLogPdf(x -> 3x - exp(x))          # y = 3 counts, log-rate x
cavity = NormalMeanVariance(0.0, 1.0)
laplace = DeltaMessagePassingRules.site(DeltaMessagePassingRules.LaplaceForm(), raw, cavity)
ep = DeltaMessagePassingRules.site(DeltaMessagePassingRules.MomentForm(), raw, cavity)
laplace.site, ep.site
```

### Where the sites live

The engine stores the sites at the variables, through ReactiveMP's site policy, and the nodes do
not store them. Rules and rule matching are therefore unchanged. A node with a hand-written rule
that already returns a Gaussian, such as the expectation-propagation rule of the Probit node,
sends a message that is its own site. The drawback is that each site is a Gaussian over a single
variable. A node with several random inputs gets a separate site at each input and no joint site
over them; a joint site would need a rule in the node itself.

### The messages the fallback sends

Under a propagation scheme, [`ExpectationRuleFallback`](@ref)`(propagation)` sends unprojected
messages, and the sites project them:

- Towards `out` of a deterministic node, it sends the pushforward of the input messages, a
  [`PushforwardMessage`](@ref). The site at the output is the projection of the inputs' tilted
  density against the cavity of the output. The input messages do not need to be proper for
  this.
- Towards one input of a node with several random inputs, it sends the belief-propagation
  message, the factor integrated against the messages on the other edges, a
  [`TiltedJointMessage`](@ref). Its site is the block of the target in the joint projection of
  the tilted density over all free inputs.

The integral requires the messages of the other inputs to be Gaussians or point masses. If one of
them is of another family, such as a Gamma message on a precision, or comes as a mean-field
marginal, the message towards the target is the expected log-density `E[log f]`, as without
propagation.

When a raw message arrives again unchanged, its last site is reused without a new projection.
[`same_message`](@ref) and [`same_density`](@ref) decide whether a message is unchanged.

```@docs
DeltaMessagePassingRules.site
DeltaMessagePassingRules.PushforwardMessage
DeltaMessagePassingRules.TiltedJointMessage
DeltaMessagePassingRules.same_message
DeltaMessagePassingRules.same_density
DeltaMessagePassingRules.is_closed_form
DeltaMessagePassingRules.GaussianMessage
DeltaMessagePassingRules.weighted_mean_precision
```

### Improper sites

A site may be improper, and so may a cavity or an outbound message: it can be a Gaussian whose
precision is singular or indefinite. For example, the site of a pulled-back `sin(θ)` observation
has a precision near zero in the directions the observation does not inform, and a negative
precision where `sin` is concave against the cavity. This is normal in expectation propagation,
where only the tilted densities need to be proper. The package does not check sites for
properness and does not clip them. The functions here use a site only through its natural
parameters, in weighted-mean form (`NormalWeightedMeanPrecision`,
`MvNormalWeightedMeanPrecision`), and not through its moments. Computing the moments would
replace an indefinite precision with a positive definite one, which silently describes a
different Gaussian.

### The free energy

The free energy of expectation propagation splits into local terms, each computed by a node or a
variable from its own inputs. Each variable contributes its Bethe term, `(d_i - 1) H[q_i]`, and
each node with rules contributes its average energy minus its entropy. A node that sends a raw
message contributes [`cavity_energy`](@ref) instead,

```math
F_a = -\log \int f_a \prod_i c_{ia} + \sum_i \mathbb{E}_{q_i}[\log c_{ia}],
```

computed from the inbound messages of the node, `c_ia`, which are the cavities, and from the
marginals `q_i` of its variables. The terms add up to the free energy of EP because the sites
multiply to the marginals: in natural parameters, `Σ_a η(c_ia) = (d_i - 1) η(q_i)`. Each
`E_{q_i}[log c_ia]` can therefore be moved from the variable's term to the node's term, and the
other terms stay the same. Rescaling a cavity does not change the sum of the two parts, so an
improper cavity enters through its kernel. The integral must be the scheme's own estimate:
Laplace's estimate, or the quadrature of the moments. If the integral is computed exactly under
Laplace propagation, the local terms no longer add up to the global free energy; on a series of
Poisson counts they differed by 0.34.

At a fixed point the free energy approximates `-log p(y)`, but it is not a bound, and a lower
value does not mean a better approximation.

```@docs
DeltaMessagePassingRules.cavity_energy
```

### References

- T. P. Minka, *Expectation propagation for approximate Bayesian inference*, UAI 2001; and
  *Divergence measures and message passing*, Microsoft Research technical report, 2005.
- A. J. Smola, S. V. N. Vishwanathan and E. Eskin, *Laplace propagation*, Advances in Neural
  Information Processing Systems 16, 2004.

## [Natural-gradient propagation](@id delta-universal-ngmp)

Natural-gradient propagation (Lukashchuk, Yemets, Ledbetter and Şenöz, *Information Geometry of
Message Passing*, 2026) uses the sites of cavity propagation, one per edge of a random variable.
The fallback sends the same belief-propagation messages: a node's log-density, the pushforward
towards the output of a deterministic node, or the factor integrated against the messages on the
other edges. The difference is in how a site is updated from its message, and in the family of
the site.

### The target and the step

Consider an edge with a raw message whose log-density is `ℓ`. Let `q` be the marginal, the cavity
times the site of the edge, in an exponential family with sufficient statistics `T` and Fisher
information `G`. The target of the site is

```math
\eta^\star = G(q)^{-1} \operatorname{Cov}_q[T(x), \ell(x)] = \nabla_\mu \mathbb{E}_q[\ell],
```

in natural parameters. It is the gradient of the expected log-message with respect to the mean
parameters `μ` of `q`, and [`natural_gradient_target`](@ref) computes it. In a Gaussian family,
Stein's lemma gives `Λ⋆ = -E_q[∇²ℓ]` and `ξ⋆ = E_q[∇ℓ] + Λ⋆ m`, computed from the values of `ℓ`
only. The site moves from its last natural parameters `η` towards the target by a damped
heavy-ball step,

```math
v \leftarrow \beta v + \alpha (\eta^\star - \eta), \qquad \eta \leftarrow \eta + v,
```

The velocity `v` is shortened to length `max_step` and then halved until the marginal is proper.
`η` and `v` are the state of the edge, and the engine keeps them in the outcome of the site. At a
fixed point each site equals its target, and each marginal satisfies
`λ = λ_prior + Σ_k ∇_μ E_q[ℓ_k]`, the stationarity condition of the variational free energy over
the family. With Gaussian cavities this is the fixed point of conjugate-computation variational
inference (Khan and Lin, 2017). For a model with one latent variable, it is the variational
Gaussian of the joint.

[`site`](@ref) with a [`NaturalGradientForm`](@ref) takes one step. In the example below, the
message is a log-normal, a normal pushed through `exp`, and the cavity is a Gamma. This is the
situation of a rate with a log-normal prior and a Poisson count. The site is a Gamma. After
repeated steps the site settles at its own target:

```@example universal
import DeltaMessagePassingRules: NaturalGradientForm, PushforwardMessage, natural_gradient_target, site

lognormal = PushforwardMessage(exp, (NormalMeanVariance(1.0, 0.5),))
cavity = GammaShapeRate(5.0, 1.0)            # a count of 4: its message is Gamma(5, 1)
form = NaturalGradientForm()
step = site(form, lognormal, cavity)
for _ in 1:60
    global step = site(form, lognormal, cavity, step.outcome)
end
step.site, step.outcome.converged
```

```@example universal
marginal = prod(GenericProd(), cavity, step.site)
natural_gradient_target(lognormal, marginal, nothing) ≈ step.outcome.η
```

### Sites in any exponential family

In natural-gradient propagation, a site is in the family of its cavity, the product of the
variable's other sites. This family need not be Gaussian.

- A closed message of the cavity's family, such as a Gaussian towards a Gaussian or a Gamma
  towards a Gamma, is its own site, so the conjugate parts of a model do not change.
- A Gaussian cavity gives a Gaussian site in weighted-mean form, and a Gamma cavity gives a
  `GammaShapeRate` site. Any other continuous univariate family, such as a Beta, gives an
  `ExponentialFamilyDistribution` of that family. Products of sites in these types are closed,
  whether the sites are proper or not.
- A raw message, or a closed message of another family such as a log-normal towards a Gamma,
  takes a natural-gradient step.
- If the cavity is in none of these families, for example a discrete distribution or a
  multivariate one that is not a Gaussian, the message passes as it is. The variable's belief is
  then projected as a product onto its [`VariationalForm`](@ref) Gaussian, like the belief of a
  variable with one connection.

The first step starts from the flat site, `η = 0`, so the marginal equals the cavity, which must
be proper: a prior or an initial message, for example. A raw message that has no cavity yet
passes as it is and takes its first step once a cavity has formed. The family of a variable is the family that its first site takes, which is the prior's
family when the prior's message arrives first.

In a multivariate Gaussian family, the expectations of the target are computed with 16
Gauss–Hermite points per dimension. For a univariate belief of any family they use
ExpectationApproximations' `UnconstrainedLegendre`, Gauss–Legendre panels in the unconstrained
coordinate of the domain, `x` for a Gaussian, `log x` for a Gamma and `logit x` for a Beta, placed
around the mode of the density there. On a broad Gaussian belief the panels resolve a
log-density that bends on a smaller scale, as a logistic likelihood's does at zero, where 16
Gauss–Hermite points put the target's precision 2% off at a variance of 100. The sufficient statistics of a
Gamma include `log x`, and Gauss–Laguerre points, which are exact for polynomials, converge slowly
on its expectation when the shape is small. In the unconstrained coordinate the targets are
accurate to about 1e-12 for Gamma shapes and Beta parameters from 0.05, a Beta with a parameter
below 1 included. A target that is still not finite leaves the site where it is, with the
outcome `:nonfinite`. `NaturalGradientForm(backend = …)` selects another backend.
A package can add a closed form of the target with a method of
[`natural_gradient_target`](@ref) for its message type and the family of `q`.

When [ClosedFormExpectations](https://github.com/biaslab/ClosedFormExpectations.jl) is loaded,
the target has a closed form wherever ClosedFormExpectations has one for the gradient
`∇ E_q[ℓ] = E_q[ℓ(x) ∇ log q(x)]` of the message's log-density `ℓ`, which it computes with
`BayesBase.expectation_gradient(ClosedFormExpectation(), ℓ, q)`. This covers univariate Gaussian
and Gamma beliefs. The integrand `ℓ` is `Base.Fix1(logpdf, p)` for a message that is a
distribution `p`, such as a log-normal or a Laplace message, and ClosedFormExpectations'
`LogGammaKernel` for a Gamma message pulled back through `exp`, which is the message of a Poisson
count towards its log rate, `(a - 1) z - b eᶻ`. A package can map its own message type to such an
integrand with a method of [`closed_form_integrand`](@ref).

### The free energy

The fixed point is a stationary point of the Bethe free energy in which each node contributes its
average energy under the marginals, `-E_q[log f]`, and not of the free energy of expectation
propagation. The rule fallback therefore uses the average energy:
`ExpectationRuleFallback(NaturalGradientForm()).energy === :average`. On a Gaussian tree whose
only non-conjugate factors are leaves, this free energy is the negative evidence lower bound. On
other models it is a diagnostic, and it need not decrease from one iteration to the next.

### Limits

- The velocity of the heavy-ball step is carried from one marginal to the next in natural
  parameters directly, without the vector transport of information geometry. The default
  damping, together with the step halving that keeps each marginal proper, is enough for long
  chains.
- The iteration is a fixed-point scheme that updates one site at a time with natural gradients.
  It converges on the models above, but there is no proof that it converges in general.
- If the other inputs of a node are not all Gaussians or point masses, for example when one of
  them is a Gamma precision, the message towards an input is the expected log-density `E[log f]`
  and not the belief-propagation message. One natural-gradient step on that message is
  non-conjugate variational message passing.
- The pushforward of a deterministic node has its exact density only for a single input through
  a monotone function. Otherwise its log-density is that of its moment-matched Gaussian.

```@docs
NaturalGradientForm
DeltaMessagePassingRules.natural_gradient_target
DeltaMessagePassingRules.closed_form_integrand
DeltaMessagePassingRules.closed_form_target
DeltaMessagePassingRules.family_parameters
DeltaMessagePassingRules.family_site
```

### References

- M. Lukashchuk, K. Yemets, A. Ledbetter and İ. Şenöz, *Information geometry of message passing*,
  arXiv:2608.15922, 2026.
- M. E. Khan and W. Lin, *Conjugate-computation variational inference: converting variational
  inference in non-conjugate models to inferences in conjugate models*, AISTATS 2017.
