"""
    VariationalForm(; tol = 1e-10, maxiters = 100, nomode = :project, backend = nothing)

A form constraint for universal inference, the variational counterpart of
[`LaplaceForm`](@ref). It projects a belief that has no closed form onto the Gaussian that
minimises `KL(q ‖ p)` in the unconstrained coordinates of its domain, with
[`variational_form`](@ref). This Gaussian is the fixed point of conjugate-computation variational
inference. Under mean field, where the belief of a variable is the product of its prior and the
expected log-densities of [`ExpectationRuleFallback`](@ref), it is the optimal Gaussian, while
Laplace's method uses the mode and the curvature of the same product. Natural-gradient
propagation ([`NaturalGradientForm`](@ref)) has the same fixed point in a Gaussian family, and
uses this form for the beliefs that have no sites: those of a variable with one connection, or
whose cavity is in no family with sites.

- a closed form, a distribution or a point mass, is returned as it is, so a conjugate variable
  does not change;
- a product with a finite discrete factor, such as a categorical, is normalised exactly over its
  support;
- any other product is replaced by `ExpectationApproximations.gaussian_variational` in the
  unconstrained coordinates of its domain. On the real line this is ExponentialFamily's
  `NormalMeanVariance` or `MvNormalMeanCovariance`, and on any other domain an
  `ExpectationApproximations.TransformedGaussian`. The domain is taken from the first factor that
  has one.

Each projection returns the evidence lower bound of the product as the log of its normaliser, so
the value is a lower bound on the log normaliser.

# Keywords
- `tol`, `maxiters`: the tolerance and iteration limit of the Newton iterations on the bound;
- `nomode`: what to do when the iterations do not converge, as for the message of a likelihood
  alone that grows without bound. `:project` keeps the last Gaussian, and `:leave` returns the
  product unprojected (see [`LaplaceForm`](@ref));
- `backend`: the expectation backend for the quadrature of the bound, one that gives a point set
  for a Gaussian. The default, `nothing`, means `ExpectationApproximations.EADefault()`. RxInfer's
  `NaturalGradientPropagation` scheme passes the backend of the inference session.
"""
struct VariationalForm{T <: Real, B}
    tol::T
    maxiters::Int
    nomode::Symbol
    backend::B
    function VariationalForm(tol::T, maxiters::Integer, nomode::Symbol = :project, backend::B = nothing) where {T <: Real, B}
        nomode ∈ (:project, :leave) || throw(ArgumentError("`nomode` is `:project` or `:leave`, got `:$(nomode)`"))
        return new{T, B}(tol, Int(maxiters), nomode, backend)
    end
end

VariationalForm(; tol::Real = 1.0e-10, maxiters::Integer = 100, nomode::Symbol = :project, backend = nothing) =
    VariationalForm(tol, maxiters, nomode, backend)

"""
    variational_form(form::VariationalForm, density, previous = nothing)
        -> (; distribution, logZ, mode, converged, iterations, method)

Project `density` as [`VariationalForm`](@ref) describes. The result is as for
[`laplace_form`](@ref): `distribution` is the projected density, and `logZ` is its evidence lower
bound, or `nothing` for a closed form that is returned as it is. `method` is `:closed_form`,
`:point_mass`, `:discrete`, `:variational` (with `converged` and `iterations` from its
iterations), or `:unprojected` under `nomode = :leave`. `mode` is the mean of the Gaussian, pushed
forward; the next projection of the same variable starts there, as `previous`.
"""
variational_form(form::VariationalForm, density, previous = nothing) = project_form(form, density, previous)

function continuous_form(form::VariationalForm, density, parts, previous)
    domain = something(EA.domain(density), EA.RealLine())
    logp = x -> all(part -> insupport(part, x), parts) ? logpdf(density, x) : -Inf
    x0 = start_point(logp, domain, parts, previous)
    backend = something(form.backend, EA.EADefault())
    result = try
        EA.gaussian_variational(logp, x0, domain; backend, tol = form.tol, maxiters = form.maxiters)
    catch error
        # a bound that is not finite at the start: no Gaussian to begin from
        error isa ArgumentError || rethrow()
        nothing
    end
    if isnothing(result) || !result.converged
        form.nomode === :leave && return (; distribution = density, logZ = nothing, mode = previous, converged = false, iterations = isnothing(result) ? 0 : result.iterations, method = :unprojected)
        isnothing(result) && throw(ArgumentError("`VariationalForm` cannot start: the evidence lower bound of the product is not finite at its starting Gaussian"))
    end
    return (; distribution = exponential_family_gaussian(result.distribution), result.logZ, mode = result.mean, result.converged, result.iterations, method = :variational)
end
