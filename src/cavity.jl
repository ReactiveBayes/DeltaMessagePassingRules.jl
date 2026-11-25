# Cavity propagation: a variable keeps one site per edge, `proj(m · c) / c` against its outbound
# message on that edge, the cavity `c`, and every belief is a product of sites (the docs page
# "Universal inference" explains it). `site` computes one; a node that sends a raw message takes
# the free-energy term of EP, `cavity_energy`. A site, a cavity or a message out may be improper,
# a Gaussian with a singular or indefinite precision: nothing checks, and these functions use a
# Gaussian only through its natural parameters, never its moments or its normaliser: for an
# indefinite precision, FastCholesky's fallback (GMW81) silently inverts a different, positive
# definite matrix, so moments computed from it describe another Gaussian.

"""
    GaussianMessage

The Gaussians of ExponentialFamily, univariate and multivariate, in any parametrisation: the
messages, cavities and sites of cavity propagation that are Gaussian, proper or not.
"""
const GaussianMessage = Union{UnivariateNormalDistributionsFamily, MultivariateNormalDistributionsFamily}

"""
    site(form, message, cavity, previous = nothing) -> (; site, outcome)

Return the site of `message` on an edge whose cavity is `cavity`. The cavity is the variable's
outbound message on that edge. The site is `proj(message · cavity) / cavity`, where the
projection is that of `form`: a [`LaplaceForm`](@ref) (Laplace propagation) or a
[`MomentForm`](@ref) (expectation propagation).

- A closed message, a distribution or a point mass, is its own site.
- If the cavity is not a Gaussian, the message passes as it is, and the variable's beliefs are
  projected as products, as without sites. A cavity that is an unprojected product of raw
  messages, as before the other edges have sites, is first replaced by its Laplace approximation
  where that is a Gaussian on the real line.
- Otherwise `form` projects `x ↦ log message(x) + ξᵀx - xᵀWx/2`, where `(ξ, W)` are the natural
  parameters of the cavity, so an improper cavity is handled too. The projection starts at
  `previous`, the mode of the last site, or else at the mean of the cavity if it has one. The
  site is the projection minus the cavity in natural parameters, an
  `(Mv)NormalWeightedMeanPrecision` that may be improper. A projection that does not converge
  gives a site, with `converged = false` in the outcome, if it stopped at a maximum of the
  tilted density, as when it runs out of iterations. If it stopped elsewhere, as for a tilted
  density with no mode, `site` is `nothing` and the edge keeps its last site, or, before the edge
  has one (`previous` is `nothing`), the message passes with the outcome `(; method = :waiting)`.

`outcome` is `(; method, converged, iterations, logZ, mode)`. `method` is `:closed_form`,
`:passed`, `:waiting`, `:laplace` or `:moment_matching`, and `logZ` is the log of the integral of the tilted
density, with the cavity entering as a kernel. While no other edge has a site, the cavity is
`missing`; a raw message then passes as it is, as under belief propagation, and the outcome is
`(; method = :waiting)`: the site is to be projected again once a cavity has formed.
"""
function site(form, message, cavity, previous = nothing)
    is_closed_form(message) && return (; site = message, outcome = (; method = :closed_form))
    # while no other edge of the variable has a site, a raw message passes as it is, as under
    # belief propagation, and waits to be projected once a cavity has formed
    ismissing(cavity) && return (; site = message, outcome = (; method = :waiting))
    cavity = seed_cavity(cavity)
    cavity isa GaussianMessage || return (; site = message, outcome = (; method = :passed))
    ξ, W = weightedmean_precision(cavity)
    logp = x -> insupport(message, x) ? logpdf(message, x) + gaussian_logkernel(ξ, W, x) : -Inf
    x0 = something(previous, kernel_start(ξ, W))
    result = cavity_projection(form, logp, x0)
    usable_projection(result, logp) || return failed_site(message, result, previous)
    ξq, Wq = natural_parameters(result.distribution)
    return (; site = weighted_mean_precision(ξq - ξ, Wq - W), outcome = (; result.method, result.converged, result.iterations, result.logZ, result.mode))
end

# Whether a projection gives a site. One that converged does; one that did not still does where
# it stopped at a maximum of the tilted density, as when it ran out of iterations, and the site
# then has `converged = false`. One that stopped elsewhere, as for a tilted density with no mode,
# whose last iterate is far out, does not.
function usable_projection(result, logp)
    result.converged && return true
    x = result.mode
    (isnothing(x) || !all(isfinite, x)) && return false
    all(isfinite, mean(result.distribution)) || return false
    return at_maximum(logp, x)
end

function at_maximum(logp, x)
    H = try
        x isa Real ? ForwardDiff.derivative(y -> ForwardDiff.derivative(logp, y), x) : ForwardDiff.hessian(logp, x)
    catch error
        error isa Union{DomainError, ArgumentError, MethodError} || rethrow()
        return false
    end
    all(isfinite, H) || return false
    return x isa Real ? H < 0 : isposdef(Symmetric(-H))
end

# No site from a projection: the edge keeps its last site (`nothing`), and the projection its last
# mode as its next start; before the edge has one, the message passes, as while it waits for a
# cavity
function failed_site(message, result, previous)
    isnothing(previous) && return (; site = message, outcome = (; method = :waiting, converged = false, result.iterations))
    return (; site = nothing, outcome = (; result.method, converged = false, result.iterations, logZ = nothing, mode = previous))
end

# While the raw messages of the other edges pass, before any of them has a site, the cavity is
# their unprojected product. Its Laplace approximation on the real line stands in for it, so that
# the first sites form against a Gaussian; after that the cavity is a product of Gaussian sites. A
# product with no mode, or on another domain, is returned as it is, and the message passes.
function seed_cavity(cavity)
    is_closed_form(cavity) && return cavity
    result = try
        project_form(LaplaceForm(), cavity, nothing)
    catch error
        error isa Union{ArgumentError, DomainError, LinearAlgebra.SingularException, LinearAlgebra.PosDefException} || rethrow()
        nothing
    end
    (isnothing(result) || !result.converged) && return cavity
    return result.distribution
end

"""
    PushforwardMessage(f, inputs, backend = nothing)

The message of a deterministic node `out = f(in...)` towards `out` under cavity propagation. It is
the pushforward through `f` of the input messages `inputs`, which are the cavities of the inputs,
and it is not projected.

Its site at the output variable ([`site`](@ref)) is the projection of the pushforward against the
cavity of the output, `c_out`, minus `c_out`. The projection uses the tilted density of the
inputs, `x ↦ ∏ cᵢ(xᵢ) · c_out(f(x))`. A [`MomentForm`](@ref) computes the moments of `f(x)`
under it by quadrature, and a [`LaplaceForm`](@ref) linearises `f` at its mode. This is the site
of expectation propagation for the node in the direction of the output. Only the tilted density
needs to be proper; the input messages do not.

Under a [`NaturalGradientForm`](@ref), the site takes a step towards the natural-gradient target
of the log-density `logpdf(message, z)`, which is:

- with one random input that is a proper distribution over numbers, and `f` strictly monotone over
  it, the exact density of the pushforward, `log c(x) - log |f'(x)|` at `x = f⁻¹(z)`, with `f`
  inverted by bisection. A log-normal through `exp` is an example. Here `f` counts as monotone if
  its derivative has one sign over the median of the input ± 8 standard deviations, both measured
  in the unconstrained coordinate of the input's domain;
- otherwise the density of the moment-matched Gaussian of the pushforward, computed by `backend`
  over the inputs (`nothing` means `ExpectationApproximations.EADefault()`). This is exact when
  `f` is linear.
"""
struct PushforwardMessage{F, I, B}
    f::F
    inputs::I
    backend::B
end

PushforwardMessage(f, inputs) = PushforwardMessage(f, inputs, nothing)

is_closed_form(::PushforwardMessage) = false

BayesBase.insupport(::PushforwardMessage, z) = true

# Where a projection of a pushforward starts: the function at its inputs' means, where its density
# is not zero, as it can be at zero, `exp`'s
pushed_mean(message::PushforwardMessage) = message.f(map(input_start, message.inputs)...)
pushed_mean(part) = nothing
BayesBase.logpdf(message::PushforwardMessage, z) = log_density_function(message)(z)

# The pushforward's log-density, prepared once: through the inverse of a monotone function of a
# single input, in the unconstrained coordinate of the input's domain, or of its moment-matched
# Gaussian
function log_density_function(message::PushforwardMessage)
    inputs = message.inputs
    monotone = monotone_pushforward(message)
    if !isnothing(monotone)
        (; input, f, domain, g, u0, s) = monotone
        return function (z)
            u = invert_monotone(g, plain_value(z), u0, s)
            isnothing(u) && return -Inf
            # bisection does not carry derivatives: two Newton steps from its root do, to the
            # third order, for the projections that differentiate this density
            for _ in 1:2
                u = u + (z - g(u)) / ForwardDiff.derivative(g, u)
            end
            x = EA.forward(domain, u)
            return logpdf(input, x) - log(abs(ForwardDiff.derivative(f, x)))
        end
    end
    all(is_proper_message, inputs) || return z -> NaN
    backend = something(message.backend, EA.EADefault())
    m, V = EA.mean_cov(backend, x -> message.f(x...), FactorizedJoint(inputs))
    gaussian = m isa Real ? NormalMeanVariance(m, V) : MvNormalMeanCovariance(m, V)
    return z -> logpdf(gaussian, z)
end

# One random input that is a proper distribution over numbers, through a function monotone over
# it: the input, the function of it, its domain, the function of its unconstrained coordinate, and
# where that lies; `nothing` otherwise
function monotone_pushforward(message::PushforwardMessage)
    inputs = message.inputs
    free = Tuple(i for i in eachindex(inputs) if !(inputs[i] isa PointMass))
    (length(free) == 1 && inputs[only(free)] isa ContinuousUnivariateDistribution && is_proper_message(inputs[only(free)])) || return nothing
    input = inputs[only(free)]
    f = x -> message.f(map(i -> i in free ? x : mean(inputs[i]), Tuple(eachindex(inputs)))...)
    domain = something(EA.domain(input), EA.RealLine())
    g = u -> f(EA.forward(domain, u))
    u0, s = unconstrained_location_scale(input, domain)
    is_monotone(g, u0, s) || return nothing
    return (; input, f, domain, g, u0, s)
end

# The domain of a pushforward through a monotone function: the image of its input's support, as
# `exp` maps the real line onto the positive numbers, so that the product it is a factor of is
# projected there; `nothing` where it is not known
function EA.domain(message::PushforwardMessage)
    monotone = monotone_pushforward(message)
    isnothing(monotone) && return nothing
    ends = try
        map(monotone.f, (minimum(monotone.input), maximum(monotone.input)))
    catch error
        error isa Union{MethodError, DomainError, ArgumentError} || rethrow()
        return nothing
    end
    lower, upper = minmax(ends...)
    (ends[1] isa Real && !isnan(lower) && !isnan(upper) && lower < upper) || return nothing
    return EA.interval(float(lower), float(upper))
end

# Where an input lies in the unconstrained coordinate of its domain: its median, and half the
# distance between its 16% and 84% quantiles, a Gaussian's standard deviation
function unconstrained_location_scale(input, domain)
    input isa UnivariateNormalDistributionsFamily && return (mean(input), std(input))
    u(p) = EA.inverse(domain, quantile(input, p))
    return (u(0.5), (u(0.8413447460685429) - u(0.15865525393145707)) / 2)
end

# A number without its dual parts
plain_value(z::ForwardDiff.Dual) = plain_value(ForwardDiff.value(z))
plain_value(z) = z

# Whether `g` maps numbers to numbers with a derivative of one sign over `m ± 8s`, nonzero over
# `m ± 2s`: a function that saturates, such as `tanh`, has a derivative that underflows to zero
# far out
function is_monotone(g, m, s)
    (isfinite(m) && isfinite(s) && s > 0) || return false
    steps = -8:0.25:8
    derivatives = try
        [ForwardDiff.derivative(g, m + s * u) for u in steps]
    catch error
        error isa Union{MethodError, DomainError, ArgumentError, DimensionMismatch} || rethrow()
        return false
    end
    all(isfinite, derivatives) || return false
    inner = derivatives[abs.(steps) .<= 2]
    return (all(>=(0), derivatives) && all(>(0), inner)) || (all(<=(0), derivatives) && all(<(0), inner))
end

# `x` with `g(x) = z`, for a monotone `g`: bracketed by steps that double away from `m`, then
# bisected; `nothing` for a `z` outside the range of `g`
function invert_monotone(g, z, m, s)
    increasing = ForwardDiff.derivative(g, m) > 0
    below(x) = (g(x) < z) == increasing
    side = below(m)
    direction = side ? 1 : -1
    a, b, width = m, m, s
    for _ in 1:1100
        b = m + direction * width
        below(b) != side && break
        a, width = b, 2 * width
        isfinite(width) || return nothing
    end
    below(b) == side && return nothing
    for _ in 1:200
        c = (a + b) / 2
        (c == a || c == b) && break
        below(c) == side ? (a = c) : (b = c)
    end
    return (a + b) / 2
end

function site(form, message::PushforwardMessage, cavity, previous = nothing)
    cavity isa Union{Missing, GaussianMessage} || return (; site = message, outcome = (; method = :passed))
    inputs = message.inputs
    free = Tuple(i for i in eachindex(inputs) if !(inputs[i] isa PointMass))
    layout = map(i -> value_dimension(inputs[i]), free)
    point(v) = map(i -> i in free ? unpack(v, layout, findfirst(isequal(i), free)) : mean(inputs[i]), Tuple(eachindex(inputs)))
    # the output's cavity as a kernel; none yet is flat, the pushforward of the inputs alone
    logc = ismissing(cavity) ? (z -> zero(eltype(z))) : (z -> message_logdensity_kernel(cavity, z))
    logp = v -> sum(k -> message_logdensity(inputs[free[k]], unpack(v, layout, k)), eachindex(free)) + logc(message.f(point(v)...))
    x0 = something(previous, pack(map(i -> input_start(inputs[i]), free)))
    result = pushforward_projection(form, message.f, point, logp, x0)
    # Laplace's linearisation at an iterate that is not a mode; moment matching keeps its points
    # on a broad Gaussian instead, and is used unless its moments are not a Gaussian's
    form isa LaplaceForm && !usable_projection(result, logp) && return failed_site(message, result, previous)
    ξq, Wq = natural_parameters(result.distribution)
    (all(isfinite, ξq) && all(isfinite, Wq)) || return failed_site(message, result, previous)
    ξc, Wc = ismissing(cavity) ? (zero(ξq), zero(Wq)) : weightedmean_precision(cavity)
    return (; site = weighted_mean_precision(ξq - ξc, Wq - Wc), outcome = (; result.method, result.converged, result.iterations, result.logZ, result.mode))
end

"""
    TiltedJointMessage(f, outbound, inputs, names, index, backend)

The message of a node with several random inputs towards its `index`-th input under cavity
propagation. It is the belief-propagation message, `x ↦ log ∫ f(…, x, …) ∏ cⱼ(yⱼ) dy`: the
node's factor integrated against the messages `inputs` on the other edges, which are their
cavities. It is not projected. `inputs` holds the message of each edge in the factor's order,
with `nothing` in the place of the target. The factor is either the keyword log-density `f` of a
stochastic node with interface names `names` (and `outbound` is `nothing`), or, for a
deterministic node, its message on `out` evaluated at its function, `log outbound(f(…))` (and
`names` is `nothing`).

Its site at the target variable ([`site`](@ref)) is the target's block of the projection of the
joint tilted density `f · c_x · ∏ cⱼ` over the target and the other free inputs, minus the
cavity `c_x` of the target. This is the expectation-propagation update for a factor with several
inputs. Only the tilted density needs to be proper, not each cavity. Where a variable projects
the message as part of a product, its log-density is the integral computed by the expectation
`backend` over the messages of the other inputs, or by Laplace's method if one of them is
improper.

The messages of the other inputs must be Gaussians or point masses. If an input has a message of
another family, or comes as a mean-field marginal, the node sends the expected log-density
instead (see [`ExpectationRuleFallback`](@ref)).
"""
struct TiltedJointMessage{F, O, I, N, B}
    f::F
    outbound::O
    inputs::I
    names::N
    index::Int
    backend::B
end

is_closed_form(::TiltedJointMessage) = false

# The node's log-factor at the values of every edge, in its order
joint_logfactor(message::TiltedJointMessage{F, Nothing}, values) where {F} = message.f(; map(Pair, message.names, values)...)
joint_logfactor(message::TiltedJointMessage, values) = message_logdensity(message.outbound, message.f(values...))

# The other edges that are free, not point masses, and the values of every edge from theirs and `x`
function joint_layout(message::TiltedJointMessage)
    inputs, k = message.inputs, message.index
    return Tuple(i for i in eachindex(inputs) if i != k && !(inputs[i] isa PointMass))
end
joint_values(message::TiltedJointMessage, x, others, free) =
    map(i -> i == message.index ? x : (i in free ? others[findfirst(isequal(i), free)] : mean(message.inputs[i])), Tuple(eachindex(message.inputs)))

BayesBase.insupport(::TiltedJointMessage, x) = true

function BayesBase.logpdf(message::TiltedJointMessage, x)
    free = joint_layout(message)
    densities = map(i -> message.inputs[i], free)
    logf = others -> joint_logfactor(message, joint_values(message, x, others, free))
    if all(density -> is_closed_form(density) && is_proper_message(density), densities)
        return first(EA.tilted_mean_cov(message.backend, logf, FactorizedJoint(densities)))
    end
    # an improper or a raw message among them, which no backend has points for: the integral of
    # the tilted density by Laplace's method
    layout = map(value_dimension, densities)
    logp = v -> (others = map(j -> unpack(v, layout, j), eachindex(free)); logf(others) + sum(j -> message_logdensity(densities[j], others[j]), eachindex(free)))
    return EA.laplace(logp, pack(map(input_start, densities))).logZ
end

function site(form, message::TiltedJointMessage, cavity, previous = nothing)
    # the target's own dimension comes from its cavity: none yet, and the message passes
    ismissing(cavity) && return (; site = message, outcome = (; method = :waiting))
    cavity = seed_cavity(cavity)
    cavity isa GaussianMessage || return (; site = message, outcome = (; method = :passed))
    ξ, W = weightedmean_precision(cavity)
    free = joint_layout(message)
    densities = map(i -> message.inputs[i], free)
    # the target first, then the other free inputs, as one vector
    layout = (value_dimension(cavity), map(value_dimension, densities)...)
    logp = function (v)
        x, others = unpack(v, layout, 1), map(j -> unpack(v, layout, j + 1), eachindex(free))
        return joint_logfactor(message, joint_values(message, x, others, free)) + gaussian_logkernel(ξ, W, x) +
            sum(j -> message_logdensity(densities[j], others[j]), eachindex(free); init = zero(eltype(v)))
    end
    start = pack((kernel_start(ξ, W), map(input_start, densities)...))
    x0 = previous isa AbstractVector && length(previous) == length(start) ? previous : start
    result = cavity_projection(form, logp, x0)
    usable_projection(result, logp) || return failed_site(message, result, previous)
    # the target's block of the joint Gaussian, its marginal
    block = layout[1] == 0 ? (1:1) : (1:layout[1])
    m, V = mean(result.distribution)[block], cov(result.distribution)[block, block]
    Wq = inv(Symmetric(V))
    ξq, Wq = layout[1] == 0 ? (only(Wq * m), only(Wq)) : (Wq * m, Matrix(Wq))
    return (; site = weighted_mean_precision(ξq - ξ, Wq - W), outcome = (; result.method, result.converged, result.iterations, result.logZ, result.mode))
end

"""
    same_message(a, b, tol = 0) -> Bool

Return whether two raw messages are the same density, up to a relative tolerance `tol` in natural
parameters. If they are, a site computed against the same cavity does not need to be projected
again. A message pulled back through a deterministic node is compared by its function and its
message on `out`, a pushforward by its function and its inputs, and a
[`TiltedJointMessage`](@ref) by its factor and the messages on the other edges. Any other message
is compared by identity, because a log-density that is a closure cannot be compared.
"""
same_message(a, b, tol = 0) = a === b
same_message(a::NodeFunctionLogPdf{<:PulledBack}, b::NodeFunctionLogPdf{<:PulledBack}, tol = 0) =
    a.logpdf.f === b.logpdf.f && same_density(a.logpdf.outbound, b.logpdf.outbound, tol)
same_message(a::PushforwardMessage, b::PushforwardMessage, tol = 0) =
    a.f === b.f && length(a.inputs) == length(b.inputs) && all(i -> same_density(a.inputs[i], b.inputs[i], tol), eachindex(a.inputs))
same_message(a::TiltedJointMessage, b::TiltedJointMessage, tol = 0) =
    a.f === b.f && a.index == b.index && a.names == b.names && same_density(a.outbound, b.outbound, tol) &&
    length(a.inputs) == length(b.inputs) && all(i -> same_density(a.inputs[i], b.inputs[i], tol), eachindex(a.inputs))

"""
    same_density(a, b, tol = 0) -> Bool

Return whether two densities agree up to a relative tolerance `tol`. Two Gaussians are compared in
natural parameters, and two point masses agree if they are at the same point. Anything else is
compared by identity.
"""
same_density(a, b, tol = 0) = a === b
same_density(a::PointMass, b::PointMass, tol = 0) = mean(a) == mean(b)
function same_density(a::GaussianMessage, b::GaussianMessage, tol = 0)
    (ξa, Wa), (ξb, Wb) = weightedmean_precision(a), weightedmean_precision(b)
    size(ξa) == size(ξb) || return false
    scale = 1 + max(maximum(abs, ξb), maximum(abs, Wb))
    return max(maximum(abs, ξa .- ξb), maximum(abs, Wa .- Wb)) <= tol * scale
end

message_logdensity_kernel(c, z) = ((ξ, W) = weightedmean_precision(c); gaussian_logkernel(ξ, W, z))

# Where the inputs' tilted density starts: each input's mean where its precision solves
input_start(input::GaussianMessage) = ((ξ, W) = weightedmean_precision(input); kernel_start(ξ, W))
input_start(input) = mean(input)
# a raw message, which has no mean: the mean of a factor that has one, or the middle of its domain
function input_start(input::RawDensity)
    for part in map(representative, factors(input))
        m = part isa Distribution ? factor_mean(part) : pushed_mean(part)
        isnothing(m) || return m
    end
    domain = something(EA.domain(input), EA.RealLine())
    dimension = value_dimension(input)
    return EA.forward(domain, dimension == 0 ? 0.0 : zeros(dimension))
end

# The moments of f(x) under the tilted density of the inputs: on its Laplace approximation by
# quadrature, weighted by the density against it, as `moment_matching` places its points
function pushforward_projection(form::MomentForm, f, point, logp, x0)
    laplace = EA.laplace(logp, x0; tol = form.tol, maxiters = form.maxiters)
    base = laplace.converged ? laplace.distribution : (x0 isa Real ? Normal(float(x0), 1.0) : MvNormal(float.(collect(x0)), Matrix{Float64}(LinearAlgebra.I, length(x0), length(x0))))
    set = prepare_expectation(form.backend, base)
    xs = collect(set.points)
    logw = [log(w) + logp(x) - logpdf(base, x) for (x, w) in zip(xs, set.weights)]
    top = maximum(logw)
    w = exp.(logw .- top)
    total = sum(w)
    w ./= total
    zs = [output_vector(f(point(x)...)) for x in xs]
    m = sum(w .* zs)
    V = Matrix(Symmetric(sum(wi * (z - m) * (z - m)' for (wi, z) in zip(w, zs))))
    # moments that are not a Gaussian's, as when the weights fall on one point: Laplace's instead
    (all(isfinite, m) && all(isfinite, V) && LinearAlgebra.isposdef(V)) ||
        return merge(linearised(f, point, laplace), (; converged = false, method = :moment_matching))
    distribution = f(point(first(xs))...) isa Real ? Normal(only(m), sqrt(only(V))) : MvNormal(m, V)
    return (; distribution, logZ = top + log(total), mode = laplace.mode, converged = laplace.converged && isfinite(total), laplace.iterations, method = :moment_matching)
end

# Laplace's: f linearised at the mode of the tilted density, its Gaussian pushed through
pushforward_projection(form::LaplaceForm, f, point, logp, x0) =
    linearised(f, point, EA.laplace(logp, x0; tol = form.tol, maxiters = form.maxiters))

function linearised(f, point, laplace)
    x̂ = laplace.mode
    g = v -> output_vector(f(point(v)...))
    J = x̂ isa Real ? reshape(ForwardDiff.derivative(g, x̂), :, 1) : ForwardDiff.jacobian(g, x̂)
    Σ = x̂ isa Real ? fill(var(laplace.distribution), 1, 1) : Matrix(cov(laplace.distribution))
    m, V = g(x̂), J * Σ * J'
    distribution = f(point(x̂)...) isa Real ? Normal(only(m), sqrt(only(V))) : MvNormal(m, Matrix(Symmetric(V)))
    return (; distribution, laplace.logZ, mode = x̂, laplace.converged, laplace.iterations, method = :laplace)
end

output_vector(z::Real) = [z]
output_vector(z) = collect(z)

# The projection of a tilted density on the real line, by each form's own method
function cavity_projection(form::LaplaceForm, logp, x0)
    result = EA.laplace(logp, x0; tol = form.tol, maxiters = form.maxiters)
    return (; result.distribution, result.logZ, result.mode, result.converged, result.iterations, method = :laplace)
end
function cavity_projection(form::MomentForm, logp, x0)
    result = EA.moment_matching(logp, x0; backend = form.backend, tol = form.tol, maxiters = form.maxiters)
    return (; result.distribution, result.logZ, mode = result.mean, result.converged, result.iterations, method = :moment_matching)
end
# in the unconstrained coordinates of `domain`, with its Jacobian; on the real line, as above
cavity_projection(form, logp, x0, ::Nothing) = cavity_projection(form, logp, x0)
function cavity_projection(form::LaplaceForm, logp, x0, domain::EA.Domain)
    result = EA.laplace(logp, x0, domain; tol = form.tol, maxiters = form.maxiters)
    return (; result.distribution, result.logZ, result.mode, result.converged, result.iterations, method = :laplace)
end
function cavity_projection(form::MomentForm, logp, x0, domain::EA.Domain)
    result = EA.moment_matching(logp, x0, domain; backend = form.backend, tol = form.tol, maxiters = form.maxiters)
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

"""
    weighted_mean_precision(ξ, W)

The Gaussian with weighted mean `ξ` and precision `W`, a `NormalWeightedMeanPrecision` for numbers
and an `MvNormalWeightedMeanPrecision`, with `W` symmetrised, for a vector and a matrix. `W` need
not be positive definite: a site or a cavity may be improper.
"""
weighted_mean_precision(ξ::Real, w::Real) = NormalWeightedMeanPrecision(ξ, w)
weighted_mean_precision(ξ::AbstractVector, W::AbstractMatrix) = MvNormalWeightedMeanPrecision(ξ, (W + W') / 2)

# A message as a log-density of a value. A Gaussian in weighted-mean form, the form a site and a
# product of sites take and which may be improper, is the log of its kernel, through its natural
# parameters; any other message its log-density. The two differ by the normaliser, a constant,
# which neither a site, a pulled-back message's density nor `cavity_energy` depends on.
const WeightedMeanGaussian = Union{NormalWeightedMeanPrecision, MvNormalWeightedMeanPrecision}
message_logdensity(message::WeightedMeanGaussian, x) = ((ξ, W) = weightedmean_precision(message); gaussian_logkernel(ξ, W, x))
message_logdensity(message, x) = logpdf(message, x)

# Whether a message has a normaliser: a Gaussian kernel with a positive definite precision
is_proper_message(message::WeightedMeanGaussian) = ((_, W) = weightedmean_precision(message); W isa Real ? W > 0 : isposdef(Symmetric(Matrix(W))))
is_proper_message(message) = true

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

The free-energy term, under cavity propagation, of a node that sends a raw message:

    F = -log ∫ f ∏ᵢ cᵢ + Σᵢ E_{qᵢ}[log cᵢ]

It is computed from the node's inbound messages `cᵢ` in `args.m`, which are the cavities, and the
marginals `qᵢ` of its variables in `args.q`, by interface. The integral is the estimate of the
projection of `form`: Laplace's estimate, or the quadrature of the moments. The other terms of
the Bethe free energy do not change: the `(d - 1) H[q]` of the variables, and the average energy
minus the entropy of each node with rules. Rescaling a cavity does not change the sum of the two
parts, so each cavity enters as the log of its kernel, including an improper one. A point mass,
such as data or a constant, is plugged in.

Returns `nothing` for a node that it cannot integrate: a stochastic node without a log-density
or with a group of interfaces, or a deterministic node with no instance in `ctx` or with an
observed output.
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
    domain = energy_domain(map(name -> (marginals[name], messages[name]), free), layout)
    result = cavity_projection(form, logp, x0, domain)
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
    domain = energy_domain(map(key -> (marginals[key], messages[key]), free), layout)
    result = cavity_projection(form, logp, x0, domain)
    cross = expected_logdensity(ctx, marginals[:out], messages[:out]) + sum(key -> expected_logdensity(ctx, marginals[key], messages[key]), free)
    return with_outcome(-result.logZ + cross, (; result.method, result.converged, result.iterations))
end

# The domain of the free variables of a cavity energy, packed as one vector: each variable's own,
# read from its marginal or else its message, so that the integral is taken in the unconstrained
# coordinates of a positive or bounded variable, with the Jacobian, as its message's projection is,
# and never evaluates the node outside its support; `nothing` on the real line. A vector variable
# is on the real line.
function energy_domain(densities, layout)
    domains = map(density -> something(EA.domain(density[1]), EA.domain(density[2]), Some(EA.RealLine())), densities)
    all(domain -> domain isa EA.RealLine, domains) && return nothing
    any(i -> layout[i] != 0 && !(domains[i] isa EA.RealLine), eachindex(layout)) && return nothing
    return EA.Componentwise(Tuple(Iterators.flatten(map((domain, d) -> d == 0 ? (domain,) : ntuple(_ -> EA.RealLine(), d), domains, layout))))
end

# 0 for a number, the length for a vector; a raw message, a number unless a factor tells otherwise
value_dimension(density::Distribution{Univariate}) = 0
value_dimension(density::RawDensity) = something(product_dimension(factors(density), nothing), 0)
value_dimension(density) = length(density)

# A raw message on numbers, as a likelihood's message left unprojected for want of a mode, is a
# member of a joint as a distribution on numbers is
is_number_density(density::RawDensity) = value_dimension(density) == 0

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
