# Cavity propagation (design B, the hub's `design/cavity_propagation.md`): a variable keeps one site
# per edge, `proj(m · c) / c` against its outbound message on that edge, the cavity `c`, and every
# belief is a product of sites. `site` computes one; a node that sends a raw message takes the
# free-energy term of EP, `cavity_energy`. A site, a cavity or a message out may be improper, a
# Gaussian with a singular or indefinite precision: nothing checks, and these functions use a
# Gaussian only through its natural parameters, never its moments or its normaliser.

const GaussianMessage = Union{UnivariateNormalDistributionsFamily, MultivariateNormalDistributionsFamily}

"""
    site(form, message, cavity, previous = nothing) -> (; site, outcome)

The site of `message` on an edge whose cavity, the variable's outbound message on it, is `cavity`:
`proj(message · cavity) / cavity`, by the projection of `form`, a [`LaplaceForm`](@ref) (Laplace
propagation) or a [`MomentForm`](@ref) (expectation propagation).

- A closed message, a distribution or a point mass, is its own site.
- Against a cavity that is not a Gaussian, the message passes as it is: the variable's beliefs are
  then projected as products, as without sites.
- Otherwise the projection of `x ↦ log message(x) + ξᵀx - xᵀWx/2`, with the cavity's natural
  parameters `(ξ, W)`, so that an improper cavity takes part as well, started at `previous`, the
  last site's mode, or the cavity's mean where it has one. The site is the projection less the
  cavity in natural parameters, an `(Mv)NormalWeightedMeanPrecision`, proper or not.

`outcome` is `(; method, converged, iterations, logZ, mode)`, `method` being `:closed_form`,
`:passed`, `:laplace` or `:moment_matching`, and `logZ` the log of the tilted density's integral
with the cavity as a kernel.
"""
function site(form, message, cavity, previous = nothing)
    is_closed_form(message) && return (; site = message, outcome = (; method = :closed_form))
    cavity isa GaussianMessage || return (; site = message, outcome = (; method = :passed))
    ξ, W = weightedmean_precision(cavity)
    logp = x -> insupport(message, x) ? logpdf(message, x) + gaussian_logkernel(ξ, W, x) : -Inf
    x0 = something(previous, kernel_start(ξ, W))
    result = cavity_projection(form, logp, x0)
    ξq, Wq = natural_parameters(result.distribution)
    return (; site = weighted_mean_precision(ξq - ξ, Wq - W), outcome = (; result.method, result.converged, result.iterations, result.logZ, result.mode))
end

# The projection of a tilted density on the real line, by each form's own method
function cavity_projection(form::LaplaceForm, logp, x0)
    result = EA.laplace(logp, x0; tol = form.tol, maxiters = form.maxiters)
    return (; result.distribution, result.logZ, result.mode, result.converged, result.iterations, method = :laplace)
end
function cavity_projection(form::MomentForm, logp, x0)
    result = EA.moment_matching(logp, x0; backend = EA.GaussHermite(form.points), tol = form.tol, maxiters = form.maxiters)
    return (; result.distribution, result.logZ, mode = result.mean, result.converged, result.iterations, method = :moment_matching)
end

# `ξᵀx - xᵀWx/2`, a Gaussian's log-density up to its normaliser, defined for any symmetric `W`
gaussian_logkernel(ξ::Real, w::Real, x::Real) = ξ * x - w * x^2 / 2
gaussian_logkernel(ξ::AbstractVector, W::AbstractMatrix, x::AbstractVector) = dot(ξ, x) - dot(x, W * x) / 2

# Where a projection against a cavity starts: its mean where its precision solves, zero otherwise
kernel_start(ξ::Real, w::Real) = iszero(w) ? zero(float(ξ)) : ξ / w
function kernel_start(ξ::AbstractVector, W::AbstractMatrix)
    x = try
        W \ ξ
    catch error
        error isa LinearAlgebra.SingularException || rethrow()
        nothing
    end
    return isnothing(x) || !all(isfinite, x) ? zeros(float(eltype(ξ)), length(ξ)) : x
end

natural_parameters(q::Normal) = (mean(q) / var(q), inv(var(q)))
natural_parameters(q::MvNormal) = (W = inv(Symmetric(cov(q))); (W * mean(q), Matrix(W)))

weighted_mean_precision(ξ::Real, w::Real) = NormalWeightedMeanPrecision(ξ, w)
weighted_mean_precision(ξ::AbstractVector, W::AbstractMatrix) = MvNormalWeightedMeanPrecision(ξ, (W + W') / 2)

# A message as a log-density of a value. A Gaussian in weighted-mean form, the form a site and a
# product of sites take and which may be improper, is the log of its kernel, through its natural
# parameters; any other message its log-density. The two differ by the normaliser, a constant,
# which neither a site, a pulled-back message's density nor `cavity_energy` depends on.
const WeightedMeanGaussian = Union{NormalWeightedMeanPrecision, MvNormalWeightedMeanPrecision}
message_logdensity(message::WeightedMeanGaussian, x) = ((ξ, W) = weightedmean_precision(message); gaussian_logkernel(ξ, W, x))
message_logdensity(message, x) = logpdf(message, x)

# The log of a weighted-mean Gaussian's normaliser, `log ∫ exp(ξᵀx - xᵀWx/2) dx`: what the kernel
# lacks of its log-density. NaN where `W` is not positive definite, as for an improper message.
function gaussian_lognormaliser(message::WeightedMeanGaussian)
    ξ, W = weightedmean_precision(message)
    d = ξ isa Real ? 1 : length(ξ)
    F = ξ isa Real ? (W > 0 ? (log(W), ξ^2 / W) : nothing) : (C = cholesky(Symmetric(Matrix(W)); check = false); issuccess(C) ? (logdet(C), dot(ξ, C \ ξ)) : nothing)
    isnothing(F) && return NaN
    return (d * log(2π) - F[1] + F[2]) / 2
end

# What a pulled-back message's log scale gains where the message on out enters as its kernel
kernel_correction(outbound::WeightedMeanGaussian) = -gaussian_lognormaliser(outbound)
kernel_correction(outbound) = 0.0

"""
    cavity_energy(form, node, ctx, args)

The free-energy term of a node that sends a raw message, under cavity propagation:

    F = -log ∫ f ∏ᵢ cᵢ + Σᵢ E_{qᵢ}[log cᵢ]

from its inbound messages `cᵢ` in `args.m`, the cavities, and its variables' marginals `qᵢ` in
`args.q`, by interface. The integral is the projection of `form`'s estimate: Laplace's or the
moments' quadrature. Every other term of the Bethe free energy stays as it is: the variables'
`(d - 1) H[q]` and the average energies less entropies of the nodes with rules. Rescaling a
cavity changes neither part's sum, so each enters as the log of its kernel, an improper one too.
A point mass, data or a constant, is plugged in. `nothing` for a node it cannot integrate: a
stochastic node without a log-density or with a group of interfaces, or a deterministic one with
no instance in `ctx` or an observed output.
"""
function cavity_energy(form, node, ctx, args)
    sdtype(node) isa Deterministic && return deterministic_cavity_energy(form, node, ctx, args)
    (applicable(nodefunction, node) && isempty(interface_groups(node))) || return nothing
    messages, marginals = Dict(rule_inputs(node, args.m)), Dict(rule_inputs(node, args.q))
    names = Tuple(interfaces(node))
    all(name -> haskey(messages, name), names) || return nothing
    free = Tuple(name for name in names if !(messages[name] isa PointMass))
    isempty(free) && return nothing
    f = nodefunction(node)
    layout = map(name -> value_dimension(messages[name]), free)
    value(v, name) = name in free ? unpack(v, layout, findfirst(isequal(name), free)) : mean(messages[name])
    logp = v -> f(; map(name -> name => value(v, name), names)...) + sum(i -> message_logdensity(messages[free[i]], unpack(v, layout, i)), eachindex(free))
    x0 = pack(map(name -> mean(marginals[name]), free))
    result = cavity_projection(form, logp, x0)
    cross = sum(name -> expected_logdensity(ctx, marginals[name], messages[name]), free)
    return with_outcome(-result.logZ + cross, (; result.method, result.converged, result.iterations))
end

# A deterministic node, `out = f(in...)`: its factor is the delta of `out - f(in...)`, so the
# integral is over the inputs, of the message on out at `f` times the inputs' messages
function deterministic_cavity_energy(form, node, ctx, args)
    instance = ctx.node
    isnothing(instance) && return nothing
    f = getnodefn(instance, Target(:out))
    inputs = rule_inputs(node, args.m)
    messages, marginals = Dict(inputs), Dict(rule_inputs(node, args.q))
    (haskey(messages, :out) && !(messages[:out] isa PointMass)) || return nothing
    order = input_order(node, Tuple(key for key in map(first, inputs) if key !== :out))
    isnothing(order) && return nothing
    free = Tuple(key for key in order if !(messages[key] isa PointMass))
    isempty(free) && return nothing
    layout = map(key -> value_dimension(messages[key]), free)
    value(v, key) = key in free ? unpack(v, layout, findfirst(isequal(key), free)) : mean(messages[key])
    logp = v -> message_logdensity(messages[:out], f(map(key -> value(v, key), order)...)) + sum(i -> message_logdensity(messages[free[i]], unpack(v, layout, i)), eachindex(free))
    x0 = pack(map(key -> mean(marginals[key]), free))
    result = cavity_projection(form, logp, x0)
    cross = expected_logdensity(ctx, marginals[:out], messages[:out]) + sum(key -> expected_logdensity(ctx, marginals[key], messages[key]), free)
    return with_outcome(-result.logZ + cross, (; result.method, result.converged, result.iterations))
end

# 0 for a number, the length for a vector
value_dimension(density::Distribution{Univariate}) = 0
value_dimension(density) = length(density)

# The values of several inputs, numbers and vectors, as one vector, and back
pack(values) = reduce(vcat, map(v -> v isa Real ? [float(v)] : collect(float.(v)), values))
function unpack(v, layout, i)
    offset = sum(d -> max(d, 1), layout[1:(i - 1)]; init = 0)
    return layout[i] == 0 ? v[offset + 1] : v[(offset + 1):(offset + layout[i])]
end

# E_q[log c] with `log c` as `message_logdensity` takes it, the same in both parts of the energy:
# in closed form for a kernel, by the context's backend otherwise
function expected_logdensity(ctx, q::GaussianMessage, c::WeightedMeanGaussian)
    ξ, W = weightedmean_precision(c)
    m, V = mean_cov(q)
    return m isa Real ? ξ * m - W * (V + m^2) / 2 : dot(ξ, m) - (tr(W * V) + dot(m, W * m)) / 2
end
expected_logdensity(ctx, q, c) = expectation(prepare_expectation(expectation_backend(ctx), q), x -> message_logdensity(c, x))
