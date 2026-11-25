"""
    LaplaceForm(; tol = 1e-10, maxiters = 100, nomode = :project)

A form constraint for universal inference. It projects a belief that has no closed form, such as
the product of a prior and the messages of [`ExpectationRuleFallback`](@ref), with
[`laplace_form`](@ref):

- a closed form, a distribution or a point mass, is returned as it is, so a conjugate variable
  does not change;
- a product with a finite discrete factor, such as a categorical, is normalised exactly over its
  support;
- any other product is replaced by Laplace's approximation in the unconstrained coordinates of its
  domain. On the real line this is a Gaussian of ExponentialFamily, `NormalMeanVariance` or
  `MvNormalMeanCovariance`, the types that the rules of the Gaussian nodes take. On any other
  domain it is an `ExpectationApproximations.TransformedGaussian`. The domain is taken from the
  first factor that has one, such as the support of a prior.

Each projection also returns the log of the product's normaliser, Laplace's approximation of the
evidence. It completes the log scale of a message whose normaliser was unknown.

The type only describes the projection. An engine applies it as a form constraint; RxInfer does
so with the option `default_form_constraint = LaplaceForm()`.

# Keywords
- `tol`, `maxiters`: the tolerance and iteration limit of the Newton iterations of
  `ExpectationApproximations.laplace`.
- `nomode`: what to do when the Newton iterations do not converge and Laplace's method finds no
  mode, as for the message of a likelihood alone that grows without bound. With `:project`, the
  default, the result is the Gaussian at the last iterate, with a floored curvature. With
  `:leave`, the product is returned unprojected. For a message, `:leave` is the better choice: a
  Gaussian without a mode would be fed to an exact rule downstream, while an unprojected product
  goes to the rule fallback. RxInfer's `LocalLaplace` projects marginals with `:project` and
  messages with `:leave`.
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

Project `density` as [`LaplaceForm`](@ref) describes. `distribution` is the projected density, and
`logZ` is the log of the normaliser of `density`, or `nothing` for a closed form that is
returned as it is. `method` says how `density` was projected:

- `:closed_form`, returned as it is;
- `:point_mass`;
- `:discrete`, normalised exactly;
- `:laplace`, with `converged` and `iterations` from the Newton iterations;
- `:unprojected`, a product for which Laplace's method found no mode, returned as it is under
  `nomode = :leave`.

The Newton iterations start at `previous`, the mode of an earlier projection of the same
variable. Without it they start at the mean of the first factor that has one, or in the middle of
the domain. A product that does not give its dimension, such as the message of a likelihood
alone, is assumed to be over a number.

# Throws
An `ArgumentError` when the product has no factor that gives its dimension (a distribution or a
log-density with a domain), there is no `previous` mode, and the product does not accept a
number.
"""
laplace_form(form::LaplaceForm, density, previous = nothing) = project_form(form, density, previous)

# The projection of a form: closed forms, point masses and finite discrete products alike, any
# other product by the form's own method, `continuous_form`
function project_form(form, density, previous)
    is_closed_form(density) && return (; distribution = closed_distribution(density), logZ = nothing, mode = previous, converged = true, iterations = 0, method = :closed_form)
    parts = factors(density)
    point = findfirst(part -> part isa PointMass, parts)
    isnothing(point) || return point_mass_form(density, parts, parts[point])
    discrete = findfirst(part -> !isnothing(finite_support(part)), parts)
    isnothing(discrete) || return discrete_form(density, finite_support(parts[discrete]))
    return continuous_form(form, density, parts, previous)
end

# A closed form passes: anything but a product with no closed form or a log-density alone
is_closed_form(density) = !(density isa Union{ProductOf, LinearizedProductOf, NodeFunctionLogPdf, ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf})

# A closed form as it passes: a proper `ExponentialFamilyDistribution`, the product of the sites of
# natural-gradient propagation in a family other than the Gaussians and the Gammas, as its
# distribution, such as a `Beta`
closed_distribution(density) = density
closed_distribution(density::ExponentialFamilyDistribution) = isproper(density) ? convert(Distribution, density) : density

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

function continuous_form(form::LaplaceForm, density, parts, previous)
    domain = something(EA.domain(density), EA.RealLine())
    # -Inf outside the support of a factor, so that a Newton step leaving it is shortened
    logp = x -> all(part -> insupport(part, x), parts) ? logpdf(density, x) : -Inf
    x0 = form_start(form, logp, domain, parts, previous)
    isnothing(x0) && return unprojected(density, previous)
    result = EA.laplace(logp, x0, domain; tol = form.tol, maxiters = form.maxiters)
    result.converged || form.nomode === :project || return (; distribution = density, logZ = nothing, mode = previous, converged = false, result.iterations, method = :unprojected)
    return (; distribution = exponential_family_gaussian(result.distribution), result.logZ, result.mode, result.converged, result.iterations, method = :laplace)
end

# Laplace's Gaussian on the real line in ExponentialFamily's types, which the rules of the Gaussian
# nodes take; a Gaussian pushed forward onto another domain stays as it is
exponential_family_gaussian(distribution::Normal) = NormalMeanVariance(mean(distribution), var(distribution))
exponential_family_gaussian(distribution::MvNormal) = MvNormalMeanCovariance(mean(distribution), cov(distribution))
exponential_family_gaussian(distribution) = distribution

# Where the iterations of `form` start. A form that leaves a product it cannot project as it is
# (`nomode = :leave`) also leaves one it cannot start on, `nothing` here: a raw message alone,
# such as one that passes while its variable's sites form, gives no dimension.
function form_start(form, logp, domain, parts, previous)
    form.nomode === :leave || return start_point(logp, domain, parts, previous)
    return try
        start_point(logp, domain, parts, previous)
    catch error
        error isa ArgumentError || rethrow()
        nothing
    end
end

unprojected(density, previous) = (; distribution = density, logZ = nothing, mode = previous, converged = false, iterations = 0, method = :unprojected)

# Where Laplace's iterations start: the previous mode, the mean of a factor, or the middle of the
# domain, the first of them where the log-density is finite
function start_point(logp, domain, parts, previous)
    candidates = Any[]
    isnothing(previous) || push!(candidates, previous)
    for part in parts
        m = part isa Distribution ? factor_mean(part) : pushed_mean(part)
        isnothing(m) || push!(candidates, m)
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
        m = pushed_mean(part)
        isnothing(m) || return m isa Real ? 0 : length(m)
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
