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
    sdtype(node) isa Stochastic || return nothing
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
once (`prepared`), at `x` for the target. `slots` names each keyword argument of `f` and where its
value comes from: the position of an input in the prepared expectation, [`Free`](@ref) for the
target, or a tuple of either for a group.
"""
struct ExpectedLogDensity{N, S, F, P}
    f::F
    prepared::P
end

ExpectedLogDensity(f::F, prepared::P, slots::NamedTuple{N, S}) where {F, P, N, S} = ExpectedLogDensity{N, slots, F, P}(f, prepared)

(density::ExpectedLogDensity{N, S})(x) where {N, S} = expectation(density.prepared, NodeIntegrand{N, S}(density.f, x))

"`Free()`: the slot of the target in a node's keyword arguments, filled by the point evaluated."
struct Free end

# The node's log-density at the inputs `v`, a tuple, and the target `x`
struct NodeIntegrand{N, S, F, X}
    f::F
    x::X
end

NodeIntegrand{N, S}(f::F, x::X) where {N, S, F, X} = NodeIntegrand{N, S, F, X}(f, x)

(integrand::NodeIntegrand{N, S})(v) where {N, S} = integrand.f(; map(slot -> fill_slot(slot, v, integrand.x), S)...)

fill_slot(slot::Int, v, x) = v[slot]
fill_slot(::Free, v, x) = x
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
# the node is a distribution of the inputs' means; an unnormalised log-density supported
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
    return DomainSets.Interval(minimum(distribution), maximum(distribution))
end
