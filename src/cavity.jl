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
with the cavity as a kernel. A raw message against a `missing` cavity, while no other edge has a
site, has a site `missing`: none yet.
"""
function site(form, message, cavity, previous = nothing)
    is_closed_form(message) && return (; site = message, outcome = (; method = :closed_form))
    # a raw message waits for a cavity, while no other edge of the variable has a site
    ismissing(cavity) && return (; site = missing, outcome = (; method = :waiting))
    cavity isa GaussianMessage || return (; site = message, outcome = (; method = :passed))
    ξ, W = weightedmean_precision(cavity)
    logp = x -> insupport(message, x) ? logpdf(message, x) + gaussian_logkernel(ξ, W, x) : -Inf
    x0 = something(previous, kernel_start(ξ, W))
    result = cavity_projection(form, logp, x0)
    ξq, Wq = natural_parameters(result.distribution)
    return (; site = weighted_mean_precision(ξq - ξ, Wq - W), outcome = (; result.method, result.converged, result.iterations, result.logZ, result.mode))
end

"""
    PushforwardMessage(f, inputs)

The message of a deterministic node `out = f(in...)` towards `out` under cavity propagation: the
pushforward of its inputs' messages, `inputs`, the cavities of its inputs, through `f`,
unprojected. Its site, at the output's variable, is the projection of the pushforward against
the output's cavity `c_out` ([`site`](@ref)): of the tilted density of the inputs,
`x ↦ ∏ cᵢ(xᵢ) · c_out(f(x))`, the moments of `f(x)` by quadrature for a [`MomentForm`](@ref), or
`f` linearised at its mode for a [`LaplaceForm`](@ref), less `c_out`. Expectation propagation's
site of the node in the output's direction; it needs no input's message to be proper, only the
tilted density to be.
"""
struct PushforwardMessage{F, I}
    f::F
    inputs::I
end

is_closed_form(::PushforwardMessage) = false

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
    ξq, Wq = natural_parameters(result.distribution)
    ξc, Wc = ismissing(cavity) ? (zero(ξq), zero(Wq)) : weightedmean_precision(cavity)
    return (; site = weighted_mean_precision(ξq - ξc, Wq - Wc), outcome = (; result.method, result.converged, result.iterations, result.logZ, result.mode))
end

"""
    TiltedJointMessage(f, outbound, inputs, names, index, backend)

The message of a node with several random inputs towards one of them, the `index`-th, under
cavity propagation: `x ↦ log ∫ f(…, x, …) ∏ cⱼ(yⱼ) dy`, the node's factor integrated against the
other edges' messages `inputs`, their cavities, as belief propagation defines it, unprojected.
`inputs` holds every edge's message in the factor's order, `nothing` at the target's place. The
factor is a stochastic node's keyword log-density `f` with its interface `names` (`outbound` is
`nothing`), or a deterministic node's message on out at its function, `log outbound(f(…))`
(`names` is `nothing`).

Its site, at the target's variable ([`site`](@ref)), is the target's block of the projection of
the joint tilted density `f · c_x · ∏ cⱼ` over the target and the other free inputs, less the
target's cavity `c_x`: expectation propagation's update for a factor with several inputs, which
needs no single cavity to be proper, only the tilted density to be. Its log-density, where a
variable projects it as a product, is the integral by the expectation `backend` over the other
inputs' messages, or by Laplace's method where one of them is improper.
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
    if all(is_proper_message, densities)
        return first(EA.tilted_mean_cov(message.backend, logf, FactorizedJoint(densities)))
    end
    # an improper message among them: the integral of the tilted density by Laplace's method
    layout = map(value_dimension, densities)
    logp = v -> (others = map(j -> unpack(v, layout, j), eachindex(free)); logf(others) + sum(j -> message_logdensity(densities[j], others[j]), eachindex(free)))
    return EA.laplace(logp, pack(map(input_start, densities))).logZ
end

function site(form, message::TiltedJointMessage, cavity, previous = nothing)
    # the target's own dimension comes from its cavity: none yet, and the message waits
    ismissing(cavity) && return (; site = missing, outcome = (; method = :waiting))
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
    # the target's block of the joint Gaussian, its marginal
    block = layout[1] == 0 ? (1:1) : (1:layout[1])
    m, V = mean(result.distribution)[block], cov(result.distribution)[block, block]
    Wq = inv(Symmetric(V))
    ξq, Wq = layout[1] == 0 ? (only(Wq * m), only(Wq)) : (Wq * m, Matrix(Wq))
    return (; site = weighted_mean_precision(ξq - ξ, Wq - W), outcome = (; result.method, result.converged, result.iterations, result.logZ, result.mode))
end

"""
    same_message(a, b, tol = 0) -> Bool

Whether two raw messages are the same density, to `tol` relative in natural parameters, so that a
site computed against the same cavity need not be projected again: a message pulled back through a
deterministic node, by its function and its message on `out`, a pushforward, by its function
and its inputs, and a [`TiltedJointMessage`](@ref), by its factor and the other edges' messages. Any other message is compared by identity, as a log-density that is a closure
cannot be compared.
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

Whether two densities agree to `tol` relative: two Gaussians in natural parameters, two point
masses at the same point; anything else by identity.
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
