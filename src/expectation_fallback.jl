"""
    ExpectationRuleFallback()

A rule fallback for universal inference: where no rule fits, the messages, joint marginals and
average energies of a node, computed from its log-density with the expectation backend of the
rule context, `ctx.expectation`. An engine consults it through
[`call_fallback`](@extref MessagePassingRulesBase.call_fallback) and
[`call_energy_fallback`](@extref MessagePassingRulesBase.call_energy_fallback), never in place of
a rule that exists.

Towards an interface of a stochastic node, the message is the expected log-density
`x ↦ E_q[log f(x, ·)]` over every other input, messages and marginals alike, as the
[`nodefunction`](@extref MessagePassingRulesBase.nodefunction) of the node defines `f`: a point
mass is plugged in, a categorical summed exactly and a Gaussian integrated by the backend. Where
every other input is a point mass the message is the node's log-density itself, and its log scale
is known up to its normaliser, which a projection such as [`LaplaceForm`](@ref) computes.

A deterministic node, such as the [`DeltaFn`](@ref) of `z := f(x, y)` without an approximation
method, has its function from the engine's node object,
[`getnodefn`](@extref MessagePassingRulesBase.getnodefn)`(ctx.node, Target(:out))`. Towards `out`
the message is the moment-matched pushforward of the inputs' messages through it, a Gaussian, or
a point mass when every input is one. Towards an input it is the message on `out` pulled back
through the function, `x ↦ E[log m_out(f(…, x, …))]` over the other inputs: exact for a single
input, whose log scale is then the message on `out`'s up to the normaliser.

Unlike [`NodeFunctionRuleFallback`](@extref MessagePassingRulesBase.NodeFunctionRuleFallback),
which plugs in the inputs' means, the message keeps their spread: a categorical's probabilities and
a Gaussian's variance.

# Throws
An `ArgumentError` when the context has no expectation backend; RxInfer gives one with
`options = (context = (expectation = ExpectationApproximations.EADefault(),),)`.

See also [`LaplaceForm`](@ref).
"""
struct ExpectationRuleFallback end

function MessagePassingRulesBase.call_fallback(::ExpectationRuleFallback, node, target::Union{Target, IndexedTarget}, algorithm, ctx, args)
    sdtype(node) isa Deterministic && return deterministic_message(node, target, ctx, args)
    return stochastic_message(node, target, ctx, args)
end

# The backend the fallback computes its expectations with: the context's, which RxInfer sets with
# its `context` option.
function expectation_backend(ctx)
    backend = ctx.expectation
    isnothing(backend) && throw(
        ArgumentError(
            "`ExpectationRuleFallback` computes expectations with the rule context's expectation backend, and the context has none. " *
                "Give one with the context option, e.g. RxInfer's `options = (context = (expectation = ExpectationApproximations.EADefault(),),)`"
        )
    )
    return backend
end

# The message towards `target`: the node's log-density with the target free, in expectation over
# every other input
function stochastic_message(node, target, ctx, args)
    applicable(nodefunction, node) || return nothing
    inputs = (rule_inputs(node, args.m)..., rule_inputs(node, args.q)...)
    slots = interface_slots(node, map(first, inputs), target)
    isnothing(slots) && return nothing
    values = map(last, inputs)
    prepared = prepare_expectation(expectation_backend(ctx), FactorizedJoint(values))
    density = ExpectedLogDensity(nodefunction(node), prepared, slots)
    message = outbound_density(node, target, inputs, density)
    all(value -> value isa PointMass, values) || return message
    # every other input a point mass: the node's log-density itself, normaliser unknown
    return with_logscale(message, UndefinedLogScale(:unnormalised_product, 0.0))
end

"""
    ExpectedLogDensity(f, prepared, slots)

`x ↦ E[f(; …)]`, the keyword log-density `f` of a node in expectation over its inputs, prepared
once (`prepared`, `nothing` for none), at `x` for the target. `slots` names each keyword argument of `f` and where its
value comes from: the position of an input in the prepared expectation, [`Free`](@ref) for the
target, or a tuple of either for a group.
"""
struct ExpectedLogDensity{N, S, F, P}
    f::F
    prepared::P
end

ExpectedLogDensity(f::F, prepared::P, slots::NamedTuple{N, S}) where {F, P, N, S} = ExpectedLogDensity{N, slots, F, P}(f, prepared)

(density::ExpectedLogDensity{N, S})(x) where {N, S} = expectation(density.prepared, NodeIntegrand{N, S}(density.f, x))
# no input to take the expectation over
(density::ExpectedLogDensity{N, S, F, Nothing})(x) where {N, S, F} = NodeIntegrand{N, S}(density.f, x)(())

# The slot of the target in a node's keyword arguments, filled by the point evaluated, `Free()`,
# or by its `I`-th component, `Free{I}()`, for a joint; and the `J`-th member of the `I`-th input,
# a joint, `Component{I, J}()`
struct Free{I} end
Free() = Free{0}()
struct Component{I, J} end

# The node's log-density at the inputs `v`, a tuple, and the target `x`
struct NodeIntegrand{N, S, F, X}
    f::F
    x::X
end

NodeIntegrand{N, S}(f::F, x::X) where {N, S, F, X} = NodeIntegrand{N, S, F, X}(f, x)

(integrand::NodeIntegrand{N, S})(v) where {N, S} = integrand.f(; map(slot -> fill_slot(slot, v, integrand.x), S)...)

fill_slot(slot::Int, v, x) = v[slot]
fill_slot(::Free{0}, v, x) = x
fill_slot(::Free{I}, v, x) where {I} = x[I]
fill_slot(::Component{I, J}, v, x) where {I, J} = v[I][J]
fill_slot(slots::Tuple, v, x) = map(slot -> fill_slot(slot, v, x), slots)

# Where each keyword of the node's log-density comes from, given the keys of the inputs and the
# target, or `nothing` when an interface has no input or an input is a joint
function interface_slots(node, keys, target)
    position(key) = findfirst(isequal(key), keys)
    groups = interface_groups(node)
    names = interfaces(node)
    slots = map(names) do name
        if name in groups
            members = sort!([key[2] for key in keys if key isa Tuple{Symbol, Integer} && key[1] === name])
            target isa IndexedTarget && target_edge(target) === name && push!(members, target_index(target))
            isempty(members) && return nothing
            sort!(members)
            members == 1:length(members) || return nothing
            return Tuple(target isa IndexedTarget && target_edge(target) === name && k == target_index(target) ? Free() : position((name, k)) for k in members)
        end
        target isa Target && target_edge(target) === name && return Free()
        return position(name)
    end
    any(isnothing, slots) && return nothing
    # every input is used: none is a joint, none is left over
    count(slot -> slot isa Int, flatten_slots(slots)) == length(keys) || return nothing
    return NamedTuple{Tuple(names)}(slots)
end

flatten_slots(slots::Tuple) = foldl((acc, slot) -> (acc..., flatten_slots(slot)...), slots; init = ())
flatten_slots(slot) = (slot,)

# The message as a density: with the support of the node's distribution towards `out`, read where
# the node is a distribution of the inputs' means that says its support; an unnormalised
# log-density supported
# everywhere otherwise, whose domain a product takes from its other factors
function outbound_density(node, target, inputs, density)
    domain = outbound_domain(node, target, inputs)
    isnothing(domain) && return NodeFunctionLogPdf(density)
    return ContinuousUnivariateLogPdf(domain, density)
end

outbound_domain(node, target, inputs) = nothing

function outbound_domain(node::Type{<:ContinuousUnivariateDistribution}, ::Target{:out}, inputs)
    isempty(interface_groups(node)) || return nothing
    values = Dict(inputs)
    parameters = Iterators.drop(interfaces(node), 1)
    all(name -> haskey(values, name), parameters) || return nothing
    distribution = node(map(name -> mean(values[name]), Tuple(parameters))...)
    # the support is information a distribution may not give: Distributions has no default
    lower, upper = try
        minimum(distribution), maximum(distribution)
    catch error
        error isa MethodError || rethrow()
        return nothing
    end
    return DomainSets.Interval(lower, upper)
end

# A deterministic node's messages, through its function of the inputs in order: towards `out` the
# moment-matched pushforward of the inputs, towards an input the message on `out` pulled back
function deterministic_message(node, target, ctx, args)
    instance = ctx.node
    isnothing(instance) && return nothing
    f = getnodefn(instance, Target(:out))
    inputs = (rule_inputs(node, args.m)..., rule_inputs(node, args.q)...)
    order = input_order(node, map(first, inputs), target)
    isnothing(order) && return nothing
    values = Dict(inputs)
    target isa Target{:out} && return pushforward_message(f, map(key -> values[key], order), ctx)
    haskey(values, :out) || return nothing
    return pullback_message(f, values[:out], order, target, values, ctx, args)
end

function pushforward_message(f, inputs, ctx)
    all(input -> input isa PointMass, inputs) && return with_logscale(PointMass(f(map(mean, inputs)...)), 0.0)
    m, V = EA.mean_cov(expectation_backend(ctx), x -> f(x...), FactorizedJoint(inputs))
    return m isa Real ? NormalMeanVariance(m, V) : MvNormalMeanCovariance(m, V)
end

function pullback_message(f, outbound, order, target, values, ctx, args)
    k = findfirst(isequal(target_key(target)), order)
    others = Tuple(key for key in order if key != target_key(target))
    if isempty(others)
        density = NodeFunctionLogPdf(PulledBack(f, outbound))
        logscale = outbound_logscale(args)
        isnothing(logscale) && return density
        # the message on out at f(x): its log scale, and a normaliser in x that is unknown
        return with_logscale(density, isdefined_logscale(logscale) ? UndefinedLogScale(:unnormalised_product, logscale) : logscale)
    end
    prepared = prepare_expectation(expectation_backend(ctx), FactorizedJoint(map(key -> values[key], others)))
    return NodeFunctionLogPdf(ExpectedPullBack(f, outbound, prepared, Val(k)))
end

# The log scale of the message on out, where the engine tracks log scales
outbound_logscale(args) = isnothing(args.logscale) ? nothing : args.logscale.m[:out]

# `x ↦ log m_out(f(x))`, for a single input
struct PulledBack{F, M}
    f::F
    outbound::M
end

(pulled::PulledBack)(x) = logpdf(pulled.outbound, pulled.f(x))

# `x ↦ E[log m_out(f(…, x, …))]` over the other inputs, `x` the `K`-th
struct ExpectedPullBack{K, F, M, P}
    f::F
    outbound::M
    prepared::P
end

ExpectedPullBack(f::F, outbound::M, prepared::P, ::Val{K}) where {K, F, M, P} = ExpectedPullBack{K, F, M, P}(f, outbound, prepared)

(pulled::ExpectedPullBack{K})(x) where {K} = expectation(pulled.prepared, others -> logpdf(pulled.outbound, pulled.f(insert_at(others, K, x)...)))

insert_at(others::Tuple, k, x) = (others[1:(k - 1)]..., x, others[k:end]...)

# The inputs of a deterministic node in its function's order, the target among them: the
# interfaces after `out`, a group by its members' indices
function input_order(node, keys, target = nothing)
    all_keys = isnothing(target) ? keys : (keys..., target_key(target))
    order = Any[]
    for name in Iterators.drop(interfaces(node), 1)
        if name in interface_groups(node)
            members = sort!([key[2] for key in all_keys if key isa Tuple{Symbol, Integer} && key[1] === name])
            append!(order, [(name, k) for k in members])
        else
            name in all_keys || return nothing
            push!(order, name)
        end
    end
    return Tuple(order)
end

target_key(::Target{E}) where {E} = E
target_key(target::IndexedTarget) = (target_edge(target), target_index(target))
