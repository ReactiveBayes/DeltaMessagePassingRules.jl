# The joints and average energies of `ExpectationRuleFallback`: where a cluster's joint has no
# rule, the joint Laplace approximation of the node's log-density times its members' messages,
# in expectation over the other clusters; where an average energy has none, -E_q[log f] under the
# clusters' marginals, joints included. Members of a joint are numbers.

function MessagePassingRulesBase.call_fallback(::ExpectationRuleFallback, node, target::ClusterTarget, algorithm, ctx, args)
    members = cluster_members(target)
    sdtype(node) isa Deterministic && return deterministic_joint(node, members, ctx, args)
    return stochastic_joint(node, members, ctx, args)
end

function stochastic_joint(node, members, ctx, args)
    applicable(nodefunction, node) || return nothing
    all(member -> member isa Symbol, members) || return nothing
    messages = Dict(rule_inputs(node, args.m))
    all(member -> haskey(messages, member), members) || return nothing
    incoming = map(member -> messages[member], members)
    all(is_number_density, incoming) || return nothing
    others = rule_inputs(node, args.q)
    slots = joint_slots(node, members, map(first, others))
    isnothing(slots) && return nothing
    prepared = isempty(others) ? nothing : prepare_expectation(expectation_backend(ctx), FactorizedJoint(map(last, others)))
    logf = ExpectedLogDensity(nodefunction(node), prepared, slots)
    return joint_laplace(u -> logf(u) + incoming_logpdf(incoming, u), incoming)
end

# The joint over a deterministic node's inputs: the message on out at the function, times the
# inputs' messages
function deterministic_joint(node, members, ctx, args)
    instance = ctx.node
    (isnothing(instance) || :out in members) && return nothing
    f = getnodefn(instance, Target(:out))
    inputs = rule_inputs(node, args.m)
    messages = Dict(inputs)
    haskey(messages, :out) || return nothing
    order = input_order(node, Tuple(key for key in map(first, inputs) if key !== :out))
    isnothing(order) && return nothing
    incoming = map(key -> messages[key], order)
    all(is_number_density, incoming) || return nothing
    outbound = messages[:out]
    return joint_laplace(u -> logpdf(outbound, f(u...)) + incoming_logpdf(incoming, u), incoming)
end

is_number_density(density) = density isa Distribution{Univariate}

incoming_logpdf(incoming, u) = sum(i -> logpdf(incoming[i], u[i]), eachindex(incoming))

# Laplace in the product of the members' domains, from their messages' means
function joint_laplace(logp, incoming)
    domains = map(density -> something(EA.domain(density), EA.RealLine()), incoming)
    domain = all(d -> d isa EA.RealLine, domains) ? EA.RealLine() : EA.Componentwise(domains...)
    x0 = [mean(density) for density in incoming]
    return EA.laplace(logp, x0, domain).distribution
end

# The keywords of a node's log-density for a joint of `members`: the `i`-th member is the `i`-th
# component of the point, every other interface an input
function joint_slots(node, members, keys)
    isempty(interface_groups(node)) || return nothing
    names = interfaces(node)
    slots = map(names) do name
        i = findfirst(isequal(name), members)
        isnothing(i) || return Free{i}()
        return findfirst(isequal(name), keys)
    end
    any(isnothing, slots) && return nothing
    count(slot -> slot isa Int, slots) == length(keys) || return nothing
    return NamedTuple{Tuple(names)}(slots)
end

# The average energy -E_q[log f] of a stochastic node under its clusters' marginals
function MessagePassingRulesBase.call_energy_fallback(::ExpectationRuleFallback, node, algorithm, ctx, args)
    (sdtype(node) isa Stochastic && applicable(nodefunction, node)) || return nothing
    inputs = rule_inputs(node, args.q)
    slots = energy_slots(node, map(first, inputs))
    isnothing(slots) && return nothing
    prepared = prepare_expectation(expectation_backend(ctx), FactorizedJoint(map(last, inputs)))
    return -ExpectedLogDensity(nodefunction(node), prepared, slots)(nothing)
end

# Every interface from a marginal: its own, a member's of a group, or a member's of a joint
function energy_slots(node, keys)
    groups = interface_groups(node)
    names = interfaces(node)
    slots = map(names) do name
        if name in groups
            members = sort!([key[2] for key in keys if key isa Tuple{Symbol, Integer} && key[1] === name])
            (isempty(members) || members != 1:length(members)) && return nothing
            return Tuple(findfirst(isequal((name, k)), keys) for k in members)
        end
        single = findfirst(isequal(name), keys)
        isnothing(single) || return single
        for (i, key) in enumerate(keys)
            key isa Tuple && all(member -> member isa Symbol, key) || continue
            j = findfirst(isequal(name), key)
            isnothing(j) || return Component{i, j}()
        end
        return nothing
    end
    any(isnothing, slots) && return nothing
    return NamedTuple{Tuple(names)}(slots)
end
