# Natural-gradient propagation: a variable keeps one site per edge, as under cavity propagation,
# but a raw message's site is a damped step towards its natural-gradient target, in the family of
# the cavity, any exponential family: a Gaussian, a Gamma, a Beta. The docs page "Universal
# inference" explains it.

"""
    NaturalGradientForm(; alpha = 0.5, beta = 0.2, max_step = Inf, tol = 1e-8, backend = nothing)

The site form of natural-gradient propagation (Lukashchuk et al., *Information Geometry of
Message Passing*), the third site form after [`LaplaceForm`](@ref) and [`MomentForm`](@ref). For a
raw message,
[`site`](@ref) takes a damped step from the previous site of the edge towards

```math
\\eta^\\star = G(q)^{-1} \\operatorname{Cov}_q[T(x), \\ell(x)] = \\nabla_\\mu \\mathbb{E}_q[\\ell],
```

computed by [`natural_gradient_target`](@ref). Here `ℓ` is the log-density of the message, and `q`
is the marginal of the variable, the cavity times the previous site. `T` and `G` are the
sufficient statistics and the Fisher information of the cavity's exponential family. The step is
a heavy-ball step,

```math
v \\leftarrow \\beta v + \\alpha (\\eta^\\star - \\eta), \\qquad \\eta \\leftarrow \\eta + v,
```

in natural parameters. `v` is shortened to length `max_step` and then halved until the marginal,
the cavity plus the new site, is proper. At a fixed point each site equals its target, and the
marginal is a stationary point of the variational free energy over the family. With a Gaussian
cavity, it is the Gaussian of conjugate-computation variational inference.

# Keywords
- `alpha`, `beta`: the step size and the momentum, with `0 < alpha` and `0 ≤ beta < 1`. With
  `alpha = 1` and `beta = 0` the site jumps to the target;
- `max_step`: the maximum length of the step `v` in natural parameters, `Inf` for no limit;
- `tol`: a site has `converged` when `‖η⋆ - η‖ ≤ tol (1 + ‖η‖)`;
- `backend`: the expectation backend for the targets, any backend of ExpectationApproximations
  that prepares a point set for the families of the sites. The default, `nothing`, uses the rules
  of [`natural_gradient_target`](@ref): Gauss–Hermite points for a multivariate Gaussian, and
  ExpectationApproximations' `UnconstrainedLegendre`, Gauss–Legendre panels in the unconstrained
  coordinate of the domain, for a univariate belief of any family.
"""
struct NaturalGradientForm{T <: Real, B}
    alpha::T
    beta::T
    max_step::T
    tol::T
    backend::B
    function NaturalGradientForm(alpha::Real, beta::Real, max_step::Real, tol::Real, backend::B = nothing) where {B}
        alpha > 0 || throw(ArgumentError("`alpha` is positive, got $(alpha)"))
        0 <= beta < 1 || throw(ArgumentError("`beta` is in [0, 1), got $(beta)"))
        max_step > 0 || throw(ArgumentError("`max_step` is positive, got $(max_step)"))
        tol >= 0 || throw(ArgumentError("`tol` is not negative, got $(tol)"))
        alpha, beta, max_step, tol = promote(float(alpha), float(beta), float(max_step), float(tol))
        return new{typeof(alpha), B}(alpha, beta, max_step, tol, backend)
    end
end

NaturalGradientForm(; alpha::Real = 0.5, beta::Real = 0.2, max_step::Real = Inf, tol::Real = 1.0e-8, backend = nothing) =
    NaturalGradientForm(alpha, beta, max_step, tol, backend)

# The free energy at a fixed point is Bethe's with the average energies, not expectation propagation's
default_energy(::NaturalGradientForm) = :average

Base.show(io::IO, form::NaturalGradientForm) =
    print(io, "NaturalGradientForm(alpha = ", form.alpha, ", beta = ", form.beta, ", max_step = ", form.max_step, ", tol = ", form.tol, ")")

# The points of a target's expectations over `q`: the backend's, or by default `GaussHermite(16)`
# for a multivariate Gaussian, `16`ᵈ points, and for any univariate family ExpectationApproximations'
# `UnconstrainedLegendre`, Gauss–Legendre panels in the unconstrained coordinate of its domain,
# `x` for a Gaussian, `log x` for a Gamma and `logit x` for a Beta, placed around the mode of the
# density there. On a broad Gaussian belief a message's log-density can bend sharply on a scale
# far below the belief's, as a logistic likelihood's at zero under N(0, 100), where 16
# Gauss–Hermite points put the target's precision 2% off and 32 points 0.4%; the panels are
# accurate to about 1e-12 there, with about 200 points. The
# sufficient statistics of a Gamma and a Beta hold `log x`, whose expectations Gauss–Laguerre and
# Gauss–Jacobi rules, exact for polynomials, approach only as a power of their points, set by the
# shape: at a Gamma's shape 0.7, 256 points are 2% off in `E[log x]`. In the unconstrained
# coordinate the density is smooth and the rule is accurate to about 1e-12 for shapes and Beta
# parameters from 0.05, including a Beta's second parameter below 1, whose upper quantiles round
# to 1.
const TARGET_GAUSS_HERMITE = EA.GaussHermite(16)
const TARGET_UNCONSTRAINED = EA.UnconstrainedLegendre()

prepare_target(backend, q) = prepare_expectation(backend, q)
prepare_target(::Nothing, q) = prepare_expectation(EA.EADefault(), q)
prepare_target(::Nothing, q::GaussianMessage) = prepare_expectation(TARGET_GAUSS_HERMITE, q)
prepare_target(::Nothing, q::UnivariateNormalDistributionsFamily) =
    prepare_expectation(TARGET_UNCONSTRAINED, Normal(mean(q), std(q)))
prepare_target(::Nothing, q::ContinuousUnivariateDistribution) =
    prepare_expectation(TARGET_UNCONSTRAINED, q isa GammaDistributionsFamily ? Gamma(shape(q), scale(q)) : q)

"""
    natural_gradient_target(message, q, backend) -> AbstractVector

Return the natural-gradient target of `message` at the belief `q`, a proper member of an
exponential family. With `ℓ` the log-density of the message, the target is
`η⋆ = G(q)⁻¹ Cov_q[T, ℓ] = ∇_μ E_q[ℓ]`. It is expressed in ExponentialFamily's natural parameters
of the family of `q`, those of `getnaturalparameters(convert(ExponentialFamilyDistribution, q))`:
`(ξ, -Λ/2)` for a Gaussian, with the precision `Λ` flattened in more than one dimension, and
`(a - 1, -b)` for a Gamma with shape `a` and rate `b`. At the fixed point of
[`NaturalGradientForm`](@ref), the site of `message` equals its target.

The expectations are computed by `backend` over `q`; `nothing` selects the rules described in
[`NaturalGradientForm`](@ref). They use the values of `ℓ` at the points only, not its
derivatives:

- if `q` is a Gaussian of any dimension, by Stein's lemma (Bonnet and Price). With `m` and `Σ` the
  mean and covariance of `q`, `E[∇ℓ] = Σ⁻¹ E[(x - m) ℓ]` and
  `E[∇²ℓ] = Σ⁻¹ E[((x - m)(x - m)ᵀ - Σ) ℓ] Σ⁻¹`, so that `Λ⋆ = -E[∇²ℓ]` and
  `ξ⋆ = E[∇ℓ] + Λ⋆ m`;
- for any other family, by the definition, with a centred covariance and the sufficient
  statistics and Fisher information of ExponentialFamily.

`ℓ` can be the log-density of a raw message, the `log ∫ f ∏ cⱼ` of a
[`TiltedJointMessage`](@ref) (the belief-propagation message), the density of a
[`PushforwardMessage`](@ref), or the log-density of a distribution from a family other than that
of `q`, such as a log-normal towards a Gamma.

# Extending

A package with a message type whose target has a closed form can add a method for its message
type and the family of `q`, such as
`natural_gradient_target(::MyMessage, q::GammaDistributionsFamily, backend)`. That method is then
used instead of quadrature. The package must also declare that the raw message type is not a
closed form, `DeltaMessagePassingRules.is_closed_form(::MyMessage) = false`, so that it gets a
site.

When ClosedFormExpectations is loaded, the target has a closed form wherever the gradient
`∇ E_q[ℓ] = E_q[ℓ ∇ log q]` has one there, for a message whose log-density is an integrand of
that package, such as a distribution's. See [`closed_form_integrand`](@ref).
"""
function natural_gradient_target end

function natural_gradient_target(message, q::GaussianMessage, backend)
    target = closed_form_target_of(message, q)
    return isnothing(target) ? gaussian_target(log_density_function(message), q, backend) : target
end
function natural_gradient_target(message, q, backend)
    target = closed_form_target_of(message, q)
    return isnothing(target) ? family_target(log_density_function(message), q, backend) : target
end

"""
    closed_form_integrand(message)

Return a callable integrand `ℓ` equal to the log-density of `message` up to a constant, of a type
that ClosedFormExpectations may have closed forms for, or `nothing`. When ClosedFormExpectations
is loaded, the [`natural_gradient_target`](@ref) of such a message has a closed form wherever
ClosedFormExpectations has one for the gradient
`BayesBase.expectation_gradient(ClosedFormExpectation(), ℓ, q)`, at a univariate Gaussian or
Gamma belief `q`. Otherwise the target is computed by quadrature. The constant does not change
the target.

For a distribution `p`, `ℓ` is `Base.Fix1(logpdf, p)`. With ClosedFormExpectations loaded, a Gamma
message pulled back through `exp`, the message of a Poisson count towards its log rate,
`(a - 1) z - b eᶻ`, also has one: ClosedFormExpectations' `LogGammaKernel(1 / b, a - 1)`, which
is defined for any shape, a count of zero included. A package with its own message type can add a
method that returns such an integrand. A method that returns a distribution is taken as its
log-density.
"""
closed_form_integrand(message) = nothing
closed_form_integrand(message::Distribution) = Base.Fix1(logpdf, message)

"""
    closed_form_target(ℓ, q)

Return the natural-gradient target of the integrand `ℓ`, from [`closed_form_integrand`](@ref), at
the belief `q` in closed form, in the parameters of [`natural_gradient_target`](@ref), or `nothing`
if there is no closed form. The package extension for ClosedFormExpectations defines it from that
package's closed form of `BayesBase.expectation_gradient`. Without the extension it returns
`nothing`.
"""
closed_form_target(ℓ, q) = nothing

function closed_form_target_of(message, q)
    ℓ = closed_form_integrand(message)
    isnothing(ℓ) && return nothing
    return closed_form_target(ℓ isa Distribution ? Base.Fix1(logpdf, ℓ) : ℓ, q)
end

# `x ↦ ℓ(x)`, once per message: a pushforward prepares its density
log_density_function(message) = x -> message_logdensity(message, x)

# A constant less, so that the products with `ℓ` do not cancel: `ℓ` at the mean, where it is finite
function reference_value(ℓ, m)
    value = try
        ℓ(m)
    catch error
        error isa Union{DomainError, ArgumentError} || rethrow()
        zero(float(eltype(m)))
    end
    return isfinite(value) ? value : zero(value)
end

# The covariances of `ℓ` with the statistics `A(x)`, from one expectation in which each point of
# `ℓ` is computed once. A point where `ℓ` is not finite, as where a message's log-density
# underflows at the far end of the quadrature (`log(1 - tanh(x))` at `x > 19`), is left out and the
# rest renormalised, as long as such points hold less than 1e-6 of `q`'s mass; otherwise the
# covariances are not finite.
function target_covariances(ℓ, A, x0, prepared)
    ℓ0 = reference_value(ℓ, x0)
    n = length(A(x0))
    e = expectation(
        prepared, function (x)
            l = ℓ(x) - ℓ0
            isfinite(l) || return zeros(2n + 2)
            a = A(x)
            return vcat(one(l), l, a .* l, a)
        end
    )
    mass = e[1]
    1 - mass <= 1.0e-6 || return fill(NaN, n)
    e = e ./ mass
    return e[3:(n + 2)] .- e[(n + 3):end] .* e[2]
end

# Stein's lemma: the covariances of `ℓ` with `x - m` and `(x - m)(x - m)ᵀ`
function gaussian_target(ℓ, q, backend)
    m, Σ = mean_cov(q)
    prepared = prepare_target(backend, q)
    if m isa Real
        c = target_covariances(ℓ, x -> [x - m, (x - m)^2], m, prepared)
        Λ = -c[2] / Σ^2
        return [c[1] / Σ + Λ * m, -Λ / 2]
    end
    k = length(m)
    c = target_covariances(ℓ, x -> (d = x - m; vcat(d, vec(d * d'))), m, prepared)
    all(isfinite, c) || return fill(NaN, k + k^2)
    S = cholesky(Symmetric(Matrix(Σ)))
    gradient = S \ c[1:k]
    C = reshape(c[(k + 1):end], k, k)
    Λ = -(S \ ((S \ ((C + C') / 2))'))
    Λ = (Λ + Λ') / 2
    return vcat(gradient + Λ * m, vec(-Λ / 2))
end

# The definition: `G⁻¹ Cov_q[T, ℓ]`, through ExponentialFamily
function family_target(ℓ, q, backend)
    ef = convert(ExponentialFamilyDistribution, q)
    statistics = x -> flat_statistics(sufficientstatistics(ef, x))
    return fisherinformation(ef) \ target_covariances(ℓ, statistics, mean(q), prepare_target(backend, q))
end

flat_statistics(t::Tuple) = reduce(vcat, map(s -> s isa Number ? [float(s)] : vec(float.(collect(s))), t))
flat_statistics(t) = flat_statistics((t,))

"""
    family_parameters(d) -> (tag, η, conditioner) or nothing

The family of a site of natural-gradient propagation: ExponentialFamily's type tag, the natural
parameters and the conditioner of `d`, a proper density or not. `nothing` for a density in no
family that has sites: a discrete one, whose messages are exact products, or a multivariate one
that is not a Gaussian. See [`family_site`](@ref).
"""
function family_parameters end

function family_parameters(d::GaussianMessage)
    ξ, W = weightedmean_precision(d)
    ξ isa Real && return (NormalMeanVariance, [float(ξ), -float(W) / 2], nothing)
    return (MvNormalMeanCovariance, vcat(float.(ξ), vec(-float.(Matrix(W)) ./ 2)), nothing)
end
family_parameters(d::GammaDistributionsFamily) = (Gamma, [float(shape(d)) - 1, -float(rate(d))], nothing)
family_parameters(d::ExponentialFamilyDistribution) =
    family_supported(ExponentialFamily.exponential_family_typetag(d)) ? (ExponentialFamily.exponential_family_typetag(d), getnaturalparameters(d), ExponentialFamily.getconditioner(d)) : nothing
family_parameters(d::PointMass) = nothing
function family_parameters(d::ContinuousUnivariateDistribution)
    # without the check of `convert(ExponentialFamilyDistribution, d)`, which an improper site fails
    return try
        tag = ExponentialFamily.exponential_family_typetag(d)
        space, natural = ExponentialFamily.DefaultParametersSpace(), ExponentialFamily.NaturalParametersSpace()
        parameters, conditioner = ExponentialFamily.separate_conditioner(tag, params(space, d))
        (tag, collect(float.(ExponentialFamily.pack_parameters(natural, tag, map(space => natural, tag, parameters, conditioner)))), conditioner)
    catch error
        error isa MethodError || rethrow()
        nothing
    end
end
family_parameters(d) = nothing

# The families of ExponentialFamilyDistribution that have sites: continuous univariate ones
family_supported(tag) = tag <: ContinuousUnivariateDistribution

"""
    family_site(tag, η, conditioner)

A site, or a marginal, from the natural parameters `η` of the family `tag`, as
[`family_parameters`](@ref) gives them, in the public type of its family and unchecked, so that a
site may be improper. A Gaussian is in weighted-mean form and a Gamma by shape and rate, whose
products with any member of the family are closed; any other family is an
`ExponentialFamilyDistribution`.
"""
function family_site end

family_site(::Type{NormalMeanVariance}, η, conditioner) = NormalWeightedMeanPrecision(η[1], -2 * η[2])
function family_site(::Type{MvNormalMeanCovariance}, η, conditioner)
    k = gaussian_dimension(η)
    W = -2 .* reshape(η[(k + 1):end], k, k)
    return MvNormalWeightedMeanPrecision(η[1:k], (W + W') / 2)
end
family_site(::Type{Gamma}, η, conditioner) = GammaShapeRate(η[1] + 1, -η[2])
family_site(tag, η, conditioner) = ExponentialFamilyDistribution(tag, η, conditioner, nothing)

gaussian_dimension(η) = round(Int, (sqrt(1 + 4 * length(η)) - 1) / 2)

# The marginal at which a target is taken, proper: a distribution the expectation backends take
family_marginal(tag::Type{<:Union{NormalMeanVariance, MvNormalMeanCovariance, Gamma}}, η, conditioner) = family_site(tag, η, conditioner)
family_marginal(tag, η, conditioner) = convert(Distribution, ExponentialFamilyDistribution(tag, η, conditioner, nothing))

function family_isproper(::Type{NormalMeanVariance}, η, conditioner)
    return all(isfinite, η) && η[2] < 0
end
function family_isproper(::Type{MvNormalMeanCovariance}, η, conditioner)
    all(isfinite, η) || return false
    k = gaussian_dimension(η)
    W = -2 .* reshape(η[(k + 1):end], k, k)
    return isposdef(Symmetric((W + W') / 2))
end
family_isproper(::Type{Gamma}, η, conditioner) = all(isfinite, η) && η[1] > -1 && η[2] < 0
family_isproper(tag, η, conditioner) = all(isfinite, η) && isproper(ExponentialFamily.NaturalParametersSpace(), tag, η, conditioner)

# A closed message that is its own site, in the form its products with the family's sites are
# closed in: a Gamma by shape and rate (Distributions' `Gamma` times an improper site checks its
# shape), any other continuous univariate family as an `ExponentialFamilyDistribution`, whose
# product with a Distributions member is not closed. A Gaussian, and the rest, as they are.
site_representation(message::GaussianMessage) = message
site_representation(message::GammaShapeRate) = message
site_representation(message::GammaDistributionsFamily) = GammaShapeRate(shape(message), rate(message))
function site_representation(message::ContinuousUnivariateDistribution)
    parameters = family_parameters(message)
    isnothing(parameters) && return message
    return family_site(parameters...)
end
site_representation(message) = message

# How many times a step is halved to keep the marginal proper before it is given up
const MAX_HALVINGS = 40

# Shortened to the length `max_step`
function clip_step(v, max_step)
    isinf(max_step) && return v
    n = norm(v)
    return n > max_step ? v .* (max_step / n) : v
end

"""
    site(form::NaturalGradientForm, message, cavity, previous = nothing) -> (; site, outcome)

Return the site of natural-gradient propagation, in the family of `cavity`. The family can be any
exponential family that has sites: a Gaussian of any dimension, a Gamma, or another continuous
univariate family.

- A point mass is its own site.
- A closed message, a distribution, is its own site if the cavity is `missing` or of the same
  family. A Gamma is converted to a `GammaShapeRate`, and a member of another non-Gaussian family
  to an `ExponentialFamilyDistribution`, so that its products with the sites of the family are
  closed, whether those sites are proper or not.
- A raw message against a `missing` cavity passes as it is, as under belief propagation, with the
  outcome `(; method = :waiting)`; it takes a step once a cavity has formed.
- If the cavity is in no family that has sites, such as a discrete one, the message passes, as
  under [`LaplaceForm`](@ref). A cavity that is an unprojected product of raw messages is first
  replaced by its Laplace approximation where that is a Gaussian on the real line.
- Any other message takes a damped step. This covers a raw log-density, a
  [`TiltedJointMessage`](@ref), a [`PushforwardMessage`](@ref), and a distribution of another
  family, such as a log-normal towards a Gamma. The step starts from the previous site of the
  edge, `previous.η`, with the velocity `previous.velocity`, and goes towards the
  [`natural_gradient_target`](@ref) at the marginal `q`, the cavity times that site. The first
  time, the previous site is the flat site, `η = 0`, so `q` is the cavity, which must then be
  proper. The step is halved until the marginal is proper, at most $(MAX_HALVINGS) times, and is
  dropped otherwise. If the cavity has moved and the previous site no longer gives a proper
  marginal, the previous site is halved first.

The outcome of a step is `(; method = :natural_gradient, converged, iterations, η, velocity)`.
`converged` is true when `‖η⋆ - η‖ ≤ tol (1 + ‖η‖)` before the step, `iterations` is the number of
steps taken on the edge, and `η` and `velocity` are the natural parameters and the velocity of
the site after the step, to be passed as the next `previous`. If the target is not finite, as
where the message's log-density is not finite over the marginal, the last site stays as it is
(`site` is `nothing`), or the flat site is sent at the first step of an edge; the outcome is then
`(; method = :nonfinite, converged = false, iterations, η, velocity)`. If no site makes the
marginal proper against the cavity, the last site stays, and the outcome holds its `η`; if no
part of the step does, the site does not move, with a zero velocity and `converged = false`.
"""
function site(form::NaturalGradientForm, message, cavity, previous = nothing)
    message isa PointMass && return (; site = message, outcome = (; method = :closed_form))
    closed = is_closed_form(message)
    if ismissing(cavity)
        closed && return (; site = site_representation(message), outcome = (; method = :closed_form))
        return (; site = message, outcome = (; method = :waiting))
    end
    cavity = seed_cavity(cavity)
    family = family_parameters(cavity)
    isnothing(family) && return (; site = message, outcome = (; method = :passed))
    if closed
        own = family_parameters(message)
        !isnothing(own) && own[1] === family[1] && return (; site = site_representation(message), outcome = (; method = :closed_form))
    end
    return natural_gradient_step(form, message, family, previous)
end

# Not the cavity propagation of the other forms
site(form::NaturalGradientForm, message::PushforwardMessage, cavity, previous = nothing) = natural_gradient_site(form, message, cavity, previous)
site(form::NaturalGradientForm, message::TiltedJointMessage, cavity, previous = nothing) = natural_gradient_site(form, message, cavity, previous)

function natural_gradient_site(form, message, cavity, previous)
    ismissing(cavity) && return (; site = message, outcome = (; method = :waiting))
    cavity = seed_cavity(cavity)
    family = family_parameters(cavity)
    isnothing(family) && return (; site = message, outcome = (; method = :passed))
    return natural_gradient_step(form, message, family, previous)
end

# The edge's last site and velocity, where the previous outcome has them in the same family
function previous_state(previous, ηc)
    if previous isa NamedTuple && haskey(previous, :η) && haskey(previous, :velocity) && length(previous.η) == length(ηc)
        return (collect(float.(previous.η)), collect(float.(previous.velocity)), get(previous, :iterations, 0))
    end
    return (zero(ηc), zero(ηc), 0)
end

function natural_gradient_step(form::NaturalGradientForm, message, (tag, ηc, conditioner), previous)
    kept, v, n = previous_state(previous, ηc)
    proper(x) = family_isproper(tag, ηc .+ x, conditioner)
    # the last site against a cavity that has moved: halved until the marginal is proper; if no
    # halving makes it proper, the engine keeps the last site, and so does the outcome
    η, shrunk = kept, false
    halvings = 0
    while !proper(η)
        halvings += 1
        if halvings > MAX_HALVINGS
            return (; site = nothing, outcome = (; method = :natural_gradient, converged = false, iterations = n, η = kept, velocity = zero(v)))
        end
        η, v, shrunk = η ./ 2, zero(v), true
    end
    # where the target is not finite, the site stays, or is the flat site at an edge's first step
    # or after a halving; the outcome says so, `:nonfinite`
    target = natural_gradient_target(message, family_marginal(tag, ηc .+ η, conditioner), form.backend)
    if !all(isfinite, target)
        changed = shrunk || n == 0
        return (; site = changed ? family_site(tag, η, conditioner) : nothing, outcome = (; method = :nonfinite, converged = false, iterations = n, η = changed ? η : kept, velocity = zero(v)))
    end
    converged = norm(target .- η) <= form.tol * (1 + norm(η))
    step = clip_step(form.beta .* v .+ form.alpha .* (target .- η), form.max_step)
    halvings = 0
    while !proper(η .+ step)
        halvings += 1
        if halvings > MAX_HALVINGS
            # no part of the step keeps the marginal proper: the site stays where it is, unmoved
            return (; site = family_site(tag, η, conditioner), outcome = (; method = :natural_gradient, converged = false, iterations = n + 1, η, velocity = zero(step)))
        end
        step = step ./ 2
    end
    η = η .+ step
    return (; site = family_site(tag, η, conditioner), outcome = (; method = :natural_gradient, converged, iterations = n + 1, η, velocity = step))
end
