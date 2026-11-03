"""
    LaplaceForm(; tol = 1e-10, maxiters = 100, nomode = :project)

A form constraint for universal inference: the projection of a belief with no closed form, such
as the product of a prior and the messages of [`ExpectationRuleFallback`](@ref), onto the form
every belief takes, by [`laplace_form`](@ref):

- a closed form, a distribution or a point mass, passes as it is, so a conjugate variable never
  changes;
- a product with a finite discrete factor, a categorical, is normalised exactly over its support;
- any other product is Laplace's approximation in the unconstrained coordinates of its domain:
  a Gaussian on the real line, and an `ExpectationApproximations.TransformedGaussian` on any
  other domain. The domain is that of the first factor that has one, a prior's support.

Each projection reports the log of the product's normaliser, Laplace's evidence approximation,
which completes the log scale of a message whose normaliser was unknown.

The type is a plain description of the projection; an engine applies it as a form constraint,
RxInfer through its option `default_form_constraint = LaplaceForm()`.

# Keywords
- `tol`, `maxiters`: of the Newton iterations of `ExpectationApproximations.laplace`.
- `nomode`: what a product becomes when Newton's iterations do not converge, so that Laplace
  found no mode, as for a likelihood's message alone that rises without bound. `:project`, the
  default, keeps the Gaussian at the last iterate, whose curvature is floored; `:leave` leaves the
  product as it is, unprojected. A message is better left: a Gaussian Laplace did not find would
  feed an exact rule downstream, where the product unprojected goes to the rule fallback. RxInfer's
  `LaplaceApproximation` projects marginals with `:project` and messages with `:leave`.
"""
struct LaplaceForm{T <: Real}
    tol::T
    maxiters::Int
    nomode::Symbol
    function LaplaceForm(tol::T, maxiters::Integer, nomode::Symbol = :project) where {T <: Real}
        nomode ∈ (:project, :leave) || throw(ArgumentError("`nomode` is `:project` or `:leave`, got `:$(nomode)`"))
        return new{T}(tol, Int(maxiters), nomode)
    end
end

LaplaceForm(; tol::Real = 1.0e-10, maxiters::Integer = 100, nomode::Symbol = :project) = LaplaceForm(tol, maxiters, nomode)

"""
    laplace_form(form::LaplaceForm, density, previous = nothing)
        -> (; distribution, logZ, mode, converged, iterations, method)

The projection of [`LaplaceForm`](@ref): `distribution` is the projected `density`, and `logZ` the
log of the normaliser of `density`, `nothing` for a closed form that passes. `method` says how it
was projected: `:closed_form` (passed as it is), `:point_mass`, `:discrete` (normalised exactly)
or `:laplace`, whose Newton iterations `converged` in `iterations` steps; or `:unprojected`, a
product Laplace found no mode for, left as it is under `nomode = :leave`. `previous`, the mode
of an earlier projection of the same variable, is where Laplace's iterations start; without one
they start at the mean of the first factor that has one, or in the middle of the domain. A product
that does not tell its dimension, such as a likelihood's message alone, is taken to be of a number.

# Throws
An `ArgumentError` when the product has no factor that tells its dimension, a distribution or a
log-density with a domain, no `previous` mode, and does not take a number.
"""
function laplace_form(form::LaplaceForm, density, previous = nothing)
    is_closed_form(density) && return (; distribution = density, logZ = nothing, mode = previous, converged = true, iterations = 0, method = :closed_form)
    parts = factors(density)
    point = findfirst(part -> part isa PointMass, parts)
    isnothing(point) || return point_mass_form(density, parts, parts[point])
    discrete = findfirst(part -> !isnothing(finite_support(part)), parts)
    isnothing(discrete) || return discrete_form(density, finite_support(parts[discrete]))
    return continuous_form(form, density, parts, previous)
end

# A closed form passes: anything but a product with no closed form or a log-density alone
is_closed_form(density) = !(density isa Union{ProductOf, LinearizedProductOf, NodeFunctionLogPdf, ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf})

# The factors of a product, nested or linearised, in order
factors(density::ProductOf) = (factors(BayesBase.getleft(density))..., factors(BayesBase.getright(density))...)
factors(density::LinearizedProductOf) = foldl((acc, part) -> (acc..., factors(part)...), view(density.vector, 1:min(density.length, length(density.vector))); init = ())
factors(density) = (density,)

# A product with a point mass is that point mass; its normaliser is the other factors at the point
function point_mass_form(density, parts, point)
    x = mean(point)
    logZ = sum(part -> part isa PointMass ? 0.0 : logpdf(part, x), parts)
    return (; distribution = point, logZ, mode = x, converged = true, iterations = 0, method = :point_mass)
end

finite_support(part::DiscreteUnivariateDistribution) = isfinite(minimum(part)) && isfinite(maximum(part)) ? support(part) : nothing
finite_support(part) = nothing

function discrete_form(density, values)
    logweights = [logpdf(density, value) for value in values]
    shift = maximum(logweights)
    logZ = shift + log(sum(w -> exp(w - shift), logweights))
    probabilities = exp.(logweights .- logZ)
    distribution = values == 1:length(values) ? Categorical(probabilities) : DiscreteNonParametric(collect(values), probabilities)
    return (; distribution, logZ, mode = values[argmax(probabilities)], converged = true, iterations = 0, method = :discrete)
end

function continuous_form(form, density, parts, previous)
    domain = something(EA.domain(density), EA.RealLine())
    # -Inf outside the support of a factor, so that a Newton step leaving it is shortened
    logp = x -> all(part -> insupport(part, x), parts) ? logpdf(density, x) : -Inf
    x0 = start_point(logp, domain, parts, previous)
    result = EA.laplace(logp, x0, domain; tol = form.tol, maxiters = form.maxiters)
    result.converged || form.nomode === :project || return (; distribution = density, logZ = nothing, mode = previous, converged = false, result.iterations, method = :unprojected)
    return (; result.distribution, result.logZ, result.mode, result.converged, result.iterations, method = :laplace)
end

# Where Laplace's iterations start: the previous mode, the mean of a factor, or the middle of the
# domain, the first of them where the log-density is finite
function start_point(logp, domain, parts, previous)
    candidates = Any[]
    isnothing(previous) || push!(candidates, previous)
    for part in parts
        part isa Distribution && (m = factor_mean(part); isnothing(m) || push!(candidates, m))
    end
    dimension = product_dimension(parts, previous)
    # A product that does not tell its dimension, such as a likelihood's message alone, is taken
    # to be of a number, as long as it takes one.
    isnothing(dimension) && return number_start(logp, domain)
    push!(candidates, EA.forward(domain, dimension == 0 ? 0.0 : zeros(dimension)))
    for x0 in candidates
        isfinite(logp(x0)) && return x0
    end
    return last(candidates)
end

# 0 for a number, the length for a vector, `nothing` when no factor tells
function product_dimension(parts, previous)
    isnothing(previous) || return previous isa Real ? 0 : length(previous)
    for part in parts
        part isa Distribution{Univariate} && return 0
        part isa Distribution{Multivariate} && return length(part)
        part isa ContinuousUnivariateLogPdf && return 0
        part isa ContinuousMultivariateLogPdf && DomainSets.dimension(support(part)) isa Integer && return DomainSets.dimension(support(part))
    end
    return nothing
end

function number_start(logp, domain)
    x0 = EA.forward(domain, 0.0)
    try
        logp(x0)
    catch error
        error isa Union{MethodError, DimensionMismatch, BoundsError} || rethrow()
        throw(
            ArgumentError(
                "`LaplaceForm` cannot tell the dimension of the product: none of its factors is a distribution or a log-density with a domain, " *
                    "and it does not take a number. A prior on the variable gives one, or an initial marginal, the previous mode"
            )
        )
    end
    return x0
end

# The mean of a factor, where it has one: Distributions gives no default for a user's distribution
function factor_mean(part)
    try
        return mean(part)
    catch error
        error isa MethodError || rethrow()
        return nothing
    end
end
