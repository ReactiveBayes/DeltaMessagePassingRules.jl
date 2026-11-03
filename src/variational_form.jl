"""
    VariationalForm(; tol = 1e-10, maxiters = 100, nomode = :project, backend = nothing)

A form constraint for universal inference, [`LaplaceForm`](@ref)'s variational sibling: the
projection of a belief with no closed form onto the Gaussian that minimises `KL(q ‖ p)` in the
unconstrained coordinates of its domain, by [`variational_form`](@ref), the fixed point of
conjugate-computation variational inference (CVI). Under mean field, where a variable's belief is
the product of its prior and the expected log-densities of [`ExpectationRuleFallback`](@ref), it is
the optimal Gaussian; Laplace's method takes the mode and the curvature of the same product.

- a closed form, a distribution or a point mass, passes as it is, so a conjugate variable never
  changes;
- a product with a finite discrete factor, a categorical, is normalised exactly over its support;
- any other product is `ExpectationApproximations.gaussian_variational` in the unconstrained
  coordinates of its domain: on the real line ExponentialFamily's `NormalMeanVariance` or
  `MvNormalMeanCovariance`, and an `ExpectationApproximations.TransformedGaussian` on any other
  domain, the domain being that of the first factor that has one.

Each projection reports the evidence lower bound of the product as the log of its normaliser, a
lower bound on it.

# Keywords
- `tol`, `maxiters`: of the Newton iterations on the bound;
- `nomode`: what a product becomes when the iterations do not converge, as for a likelihood's
  message alone that rises without bound: `:project` keeps the last Gaussian, `:leave` leaves
  the product unprojected (see [`LaplaceForm`](@ref));
- `backend`: the expectation backend of the bound's quadrature, a point set for a Gaussian;
  `nothing`, the default, for `ExpectationApproximations.EADefault()`. RxInfer's `CVI` scheme
  gives it the session's.
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

The projection of [`VariationalForm`](@ref), as [`laplace_form`](@ref)'s: `distribution` is the
projected `density` and `logZ` its evidence lower bound, `nothing` for a closed form that passes.
`method` is `:closed_form`, `:point_mass`, `:discrete`, `:variational` (whose iterations
`converged` in `iterations` steps), or `:unprojected` under `nomode = :leave`. `mode` is the
Gaussian's mean pushed forward, where the next projection of the same variable starts,
`previous`.
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
