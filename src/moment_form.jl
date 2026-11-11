"""
    MomentForm(; tol = 1e-10, maxiters = 100, nomode = :project, points = 20, backend = nothing)

A form constraint for universal inference, [`LaplaceForm`](@ref)'s sibling for expectation
propagation: the projection of a belief with no closed form onto the Gaussian with its mean and
covariance, by [`moment_form`](@ref), in the unconstrained coordinates of its domain.

- a closed form, a distribution or a point mass, passes as it is;
- a product with a finite discrete factor, a categorical, is normalised exactly over its support;
- any other product is `ExpectationApproximations.moment_matching`: the backend's points on the
  product's own Laplace approximation, weighted by the product against it. On the real line it
  is ExponentialFamily's `NormalMeanVariance` or `MvNormalMeanCovariance`, and an
  `ExpectationApproximations.TransformedGaussian` on any other domain.

Each projection reports the log of the product's normaliser, from the same quadrature.

# Keywords
- `tol`, `maxiters`: of the Newton iterations that place the points;
- `nomode`: what a product becomes when they do not converge, as for [`LaplaceForm`](@ref):
  `:project` keeps the moments on a unit Gaussian at the start, `:leave` leaves the product;
- `points`: Gauss–Hermite points per dimension;
- `backend`: the quadrature that places the points, any backend of ExpectationApproximations that
  prepares a point set for a Gaussian, such as `SphericalRadialCubature()` (2d points in d
  dimensions, exact to degree 3) or `GenUnscented()`; `nothing` is `GaussHermite(points)`, a
  product grid of `points`ᵈ points, built once.
"""
struct MomentForm{T <: Real, B}
    tol::T
    maxiters::Int
    nomode::Symbol
    points::Int
    backend::B
    function MomentForm(tol::T, maxiters::Integer, nomode::Symbol = :project, points::Integer = 20, backend = nothing) where {T <: Real}
        nomode ∈ (:project, :leave) || throw(ArgumentError("`nomode` is `:project` or `:leave`, got `:$(nomode)`"))
        points > 0 || throw(ArgumentError("`points` is positive, got $(points)"))
        backend = something(backend, EA.GaussHermite(points))
        return new{T, typeof(backend)}(tol, Int(maxiters), nomode, Int(points), backend)
    end
end

MomentForm(; tol::Real = 1.0e-10, maxiters::Integer = 100, nomode::Symbol = :project, points::Integer = 20, backend = nothing) = MomentForm(tol, maxiters, nomode, points, backend)

"""
    moment_form(form::MomentForm, density, previous = nothing)
        -> (; distribution, logZ, mode, converged, iterations, method)

The projection of [`MomentForm`](@ref), as [`laplace_form`](@ref)'s: `distribution` is the projected
`density` and `logZ` the log of its normaliser, `nothing` for a closed form that passes. `method`
is `:closed_form`, `:point_mass`, `:discrete`, `:moment_matching`, or `:unprojected` under
`nomode = :leave`. `mode` is the Gaussian's mean, where the next projection of the same variable
starts, `previous`.
"""
moment_form(form::MomentForm, density, previous = nothing) = project_form(form, density, previous)

function continuous_form(form::MomentForm, density, parts, previous)
    domain = something(EA.domain(density), EA.RealLine())
    logp = x -> all(part -> insupport(part, x), parts) ? logpdf(density, x) : -Inf
    x0 = start_point(logp, domain, parts, previous)
    result = EA.moment_matching(logp, x0, domain; backend = form.backend, tol = form.tol, maxiters = form.maxiters)
    result.converged || form.nomode === :project || return (; distribution = density, logZ = nothing, mode = previous, converged = false, result.iterations, method = :unprojected)
    return (; distribution = exponential_family_gaussian(result.distribution), result.logZ, mode = result.mean, result.converged, result.iterations, method = :moment_matching)
end
