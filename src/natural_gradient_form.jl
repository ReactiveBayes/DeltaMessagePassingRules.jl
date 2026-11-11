# Natural-gradient propagation: a variable keeps one site per edge, as under cavity propagation,
# but a raw message's site is a damped step towards its natural-gradient target, in the family of
# the cavity, any exponential family: a Gaussian, a Gamma, a Beta. The docs page "Universal
# inference" explains it.

"""
    NaturalGradientForm(; alpha = 0.5, beta = 0.2, max_step = Inf, tol = 1e-8, backend = nothing)

The site form of natural-gradient propagation (Lukashchuk et al., *Information Geometry of
Message Passing*), the third beside [`LaplaceForm`](@ref) and [`MomentForm`](@ref): a raw message's
site, at [`site`](@ref), is a damped step from the edge's previous site towards

```math
\\eta^\\star = G(q)^{-1} \\operatorname{Cov}_q[T(x), \\ell(x)] = \\nabla_\\mu \\mathbb{E}_q[\\ell],
```

[`natural_gradient_target`](@ref), with `ℓ` the message's log-density and `q` the variable's
marginal, the cavity times the previous site, in the cavity's exponential family: its sufficient
statistics `T` and Fisher information `G`. The step is the heavy ball

```math
v \\leftarrow \\beta v + \\alpha (\\eta^\\star - \\eta), \\qquad \\eta \\leftarrow \\eta + v,
```

in natural parameters, `v` shortened to the length `max_step` and then halved until the marginal,
the cavity plus the new site, is proper. At a fixed point every site is its target, and the
marginal is a stationary point of the variational free energy over the family: under a Gaussian
cavity, the Gaussian of conjugate-computation variational inference.

# Keywords
- `alpha`, `beta`: the step size and the momentum, `0 < alpha` and `0 ≤ beta < 1`. `alpha = 1`
  and `beta = 0` jump to the target;
- `max_step`: the longest step `v` in natural parameters, `Inf` for none;
- `tol`: a site has `converged` when `‖η⋆ - η‖ ≤ tol (1 + ‖η‖)`;
- `backend`: the expectation backend of the targets, any backend of ExpectationApproximations
  that prepares a point set for the families of the sites; `nothing`, the default, for the rules
  of [`natural_gradient_target`](@ref): Gauss–Hermite points for a Gaussian, and Gauss–Legendre
  points in the unconstrained coordinate of any other family's domain.
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
# for a Gaussian, `16`ᵈ points, and for any other family Gauss–Legendre points in the
# unconstrained coordinate of its domain, `log x` for a Gamma and `logit x` for a Beta, between the
# quantiles 1e-15 and 1 - 1e-15. The sufficient statistics of a Gamma and a Beta hold `log x`,
# whose expectations Gauss–Laguerre and Gauss–Jacobi rules, exact for polynomials, approach only as
# a power of their points, set by the shape: at a Gamma's shape 0.7, 256 points are 2% off in
# `E[log x]`. In the unconstrained coordinate the density is smooth and the rule converges
# geometrically: 128 points are exact to the precision of floats for shapes above 1.
const TARGET_GAUSS_HERMITE = EA.GaussHermite(16)
const TARGET_LEGENDRE_POINTS = 128
const TARGET_LEGENDRE = EA.gausslegendre(TARGET_LEGENDRE_POINTS)

prepare_target(backend, q) = prepare_expectation(backend, q)
prepare_target(::Nothing, q) = prepare_expectation(EA.EADefault(), q)
prepare_target(::Nothing, q::GaussianMessage) = prepare_expectation(TARGET_GAUSS_HERMITE, q)
function prepare_target(::Nothing, q::ContinuousUnivariateDistribution)
    distribution = q isa GammaDistributionsFamily ? Gamma(shape(q), scale(q)) : q
    domain = something(EA.domain(distribution), EA.RealLine())
    bound(p) = clamp(EA.inverse(domain, quantile(distribution, p)), -700.0, 700.0)
    lower, upper = bound(1.0e-15), bound(1 - 1.0e-15)
    nodes, weights = TARGET_LEGENDRE
    half, middle = (upper - lower) / 2, (upper + lower) / 2
    u = middle .+ half .* nodes
    points = map(v -> EA.forward(domain, v), u)
    logw = log.(weights .* half) .+ map((x, v) -> logpdf(distribution, x) + EA.logjacobian(domain, v), points, u)
    w = exp.(logw .- maximum(logw))
    return BayesBase.ExpectationPointSet(points, w ./ sum(w))
end

"""
    natural_gradient_target(message, q, backend) -> AbstractVector

The natural-gradient target of `message` at the belief `q`, a proper member of an exponential
family: `η⋆ = G(q)⁻¹ Cov_q[T, ℓ] = ∇_μ E_q[ℓ]`, with `ℓ` the message's log-density, in
ExponentialFamily's natural parameters of `q`'s family, those of
`getnaturalparameters(convert(ExponentialFamilyDistribution, q))`: `(ξ, -Λ/2)` for a Gaussian, its
precision `Λ` flattened in more than one dimension, and `(a - 1, -b)` for a Gamma of shape `a`
and rate `b`. It is the site of `message` at the fixed point of [`NaturalGradientForm`](@ref).
The expectations are by `backend` over `q`, `nothing` for the rules of
[`NaturalGradientForm`](@ref), and use `ℓ` only at its points, never its derivatives:

- `q` a Gaussian of any dimension, by Stein's lemma (Bonnet and Price): with `m` and `Σ` the mean
  and covariance of `q`, `E[∇ℓ] = Σ⁻¹ E[(x - m) ℓ]` and
  `E[∇²ℓ] = Σ⁻¹ E[((x - m)(x - m)ᵀ - Σ) ℓ] Σ⁻¹`, so that `Λ⋆ = -E[∇²ℓ]` and
  `ξ⋆ = E[∇ℓ] + Λ⋆ m`;
- any other family by the definition, the covariance centred, with the sufficient statistics and
  the Fisher information of ExponentialFamily.

`ℓ` is a raw message's log-density, a [`TiltedJointMessage`](@ref)'s `log ∫ f ∏ cⱼ` (belief
propagation's message), a [`PushforwardMessage`](@ref)'s density, or a distribution's of another
family than `q`'s, such as a log-normal towards a Gamma.

# Extending

A package with a message type whose target has a closed form adds a method for its message and
the family of `q`, such as `natural_gradient_target(::MyMessage, q::GammaDistributionsFamily, backend)`;
it is then used in place of the quadrature. Such a raw message type also says it is not closed,
`DeltaMessagePassingRules.is_closed_form(::MyMessage) = false`, so that it gets a site.
"""
function natural_gradient_target end

natural_gradient_target(message, q::GaussianMessage, backend) = gaussian_target(log_density_function(message), q, backend)
natural_gradient_target(message, q, backend) = family_target(log_density_function(message), q, backend)

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

# The family of a site: ExponentialFamily's tag, the natural parameters and the conditioner, of
# a proper density or not; `nothing` for a density in no family that has sites, a discrete one,
# whose messages are exact products, or a multivariate one that is not a Gaussian.
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

# A site, or a marginal, from its natural parameters: in the public type of its family, unchecked,
# so that a site may be improper. A Gaussian is in weighted-mean form and a Gamma by shape and rate,
# whose products with any member of the family are closed; any other family is an
# `ExponentialFamilyDistribution`.
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

The site of natural-gradient propagation, in the family of `cavity`, any exponential family that
has sites: a Gaussian of any dimension, a Gamma, or another continuous univariate family.

- A point mass is its own site.
- A closed message, a distribution, is its own site against a `missing` cavity, or one of its
  own family. A Gamma enters as a `GammaShapeRate`, and a member of another family that is not a
  Gaussian as an `ExponentialFamilyDistribution`, so that its products with the family's sites,
  proper or not, are closed.
- A raw message against a `missing` cavity has none yet, `missing`, with the outcome
  `(; method = :waiting)`.
- Against a cavity in no family that has sites, such as a discrete one, the message passes, as
  under [`LaplaceForm`](@ref).
- Any other message, a raw log-density, a [`TiltedJointMessage`](@ref), a
  [`PushforwardMessage`](@ref), or a distribution of another family, such as a log-normal towards a
  Gamma, takes a damped step from the edge's previous site, `previous.η` with the velocity
  `previous.velocity`, towards its [`natural_gradient_target`](@ref) at the marginal `q`, the
  cavity times that site: the flat site, `η = 0`, the first time, so that `q` is the cavity, which
  must then be proper. The step is halved until the marginal is proper, at most $(MAX_HALVINGS)
  times, and left out otherwise; a previous site that, against a cavity that has moved, no
  longer gives a proper marginal is halved first.

The outcome of a step is `(; method = :natural_gradient, converged, iterations, η, velocity)`:
`converged` when `‖η⋆ - η‖ ≤ tol (1 + ‖η‖)` before the step, the steps taken on the edge, and the
site's natural parameters and velocity after it, the next `previous`. A target that is not finite
leaves the last site as it is, the site `nothing`, or sends the flat site at an edge's first step;
a cavity no site makes proper leaves the last site.
"""
function site(form::NaturalGradientForm, message, cavity, previous = nothing)
    message isa PointMass && return (; site = message, outcome = (; method = :closed_form))
    closed = is_closed_form(message)
    if ismissing(cavity)
        closed && return (; site = site_representation(message), outcome = (; method = :closed_form))
        return (; site = missing, outcome = (; method = :waiting))
    end
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
    ismissing(cavity) && return (; site = missing, outcome = (; method = :waiting))
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
    η, v, n = previous_state(previous, ηc)
    proper(x) = family_isproper(tag, ηc .+ x, conditioner)
    # the last site against a cavity that has moved: halved until the marginal is proper
    shrunk = false
    halvings = 0
    while !proper(η)
        halvings += 1
        if halvings > MAX_HALVINGS
            return (; site = nothing, outcome = (; method = :natural_gradient, converged = false, iterations = n, η, velocity = v))
        end
        η, v, shrunk = η ./ 2, zero(v), true
    end
    # where the target is not finite, the site stays, or is the flat site at an edge's first step
    unchanged = shrunk || n == 0 ? family_site(tag, η, conditioner) : nothing
    target = natural_gradient_target(message, family_marginal(tag, ηc .+ η, conditioner), form.backend)
    if !all(isfinite, target)
        return (; site = unchanged, outcome = (; method = :natural_gradient, converged = false, iterations = n, η, velocity = v))
    end
    converged = norm(target .- η) <= form.tol * (1 + norm(η))
    step = clip_step(form.beta .* v .+ form.alpha .* (target .- η), form.max_step)
    halvings = 0
    while !proper(η .+ step)
        halvings += 1
        halvings > MAX_HALVINGS && (step = zero(step); break)
        step = step ./ 2
    end
    η = η .+ step
    return (; site = family_site(tag, η, conditioner), outcome = (; method = :natural_gradient, converged, iterations = n + 1, η, velocity = step))
end
