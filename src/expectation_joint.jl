# The joints and average energies of `ExpectationRuleFallback`: where a cluster's joint has no
# rule, the joint Laplace approximation of the node's log-density times its members' messages,
# in expectation over the other clusters; where an average energy has none, -E_q[log f] under the
# clusters' marginals, joints included. Members of a joint are numbers.

function MessagePassingRulesBase.call_fallback(::ExpectationRuleFallback, node, target::ClusterTarget, algorithm, ctx, args)
    members = cluster_members(target)
    sdtype(node) isa Deterministic && return deterministic_joint(node, members, algorithm, ctx, args)
    return stochastic_joint(node, members, algorithm, ctx, args)
end

function stochastic_joint(node, members, algorithm, ctx, args)
    applicable(nodefunction, node) || return nothing
    all(member -> member isa Symbol, members) || return nothing
    messages = Dict(rule_inputs(node, args.m))
    all(member -> haskey(messages, member), members) || return nothing
    incoming = map(member -> messages[member], members)
    is_cluster_input(incoming) || return nothing
    others = rule_inputs(node, args.q)
    slots = joint_slots(node, members, map(first, others))
    isnothing(slots) && return nothing
    prepared = isempty(others) ? nothing : prepare_expectation(expectation_backend(ctx), FactorizedJoint(map(last, others)))
    logf = ExpectedLogDensity(nodefunction(node), prepared, slots)
    # the node's own message towards a member, from the others' messages and the marginals
    own(i) = own_message(node, members[i], algorithm, ctx, pairs_except(members, incoming, i), Tuple(others))
    return cluster_laplace(u -> logf(u) + incoming_logpdf(incoming, u), members, incoming, own)
end

# The joint over a deterministic node's inputs: the message on out at the function, times the
# inputs' messages
function deterministic_joint(node, members, algorithm, ctx, args)
    instance = ctx.node
    (isnothing(instance) || :out in members) && return nothing
    f = getnodefn(instance, Target(:out))
    inputs = rule_inputs(node, args.m)
    messages = Dict(inputs)
    haskey(messages, :out) || return nothing
    order = input_order(node, Tuple(key for key in map(first, inputs) if key !== :out))
    isnothing(order) && return nothing
    incoming = map(key -> messages[key], order)
    outbound = messages[:out]
    # a single input that is a vector: its marginal, by Laplace over its components
    if length(incoming) == 1 && only(incoming) isa Distribution{Multivariate}
        density = only(incoming)
        return vector_laplace(u -> message_logdensity(outbound, f(u)) + logpdf(density, u), density)
    end
    is_cluster_input(incoming) || return nothing
    own(i) = isempty(interface_groups(node)) ?
        own_message(node, order[i], algorithm, ctx, ((:out => outbound), pairs_except(Tuple(order), incoming, i)...), ()) : nothing
    return cluster_laplace(u -> message_logdensity(outbound, f(u...)) + incoming_logpdf(incoming, u), Tuple(order), incoming, own)
end

is_number_density(density) = density isa Distribution{Univariate}

# A member's message is a density on numbers, or a number, a point mass such as a constant or
# data, and not every member is a number
is_point(density) = density isa PointMass{<:Real}
is_cluster_input(incoming) = all(d -> is_number_density(d) || is_point(d), incoming) && !all(is_point, incoming)

# A point mass's log-density is zero at its point, which the joint takes
incoming_logpdf(incoming, u) = sum(i -> is_point(incoming[i]) ? zero(eltype(u)) : logpdf(incoming[i], u[i]), eachindex(incoming))

# The joint over the members that are not numbers, with the numbers plugged in; each number is a
# block of its own in a `FactorizedCluster`, as the rules of a node with a known input give it.
# One member left is its own marginal: its message times the node's message towards it, `own(i)`,
# in closed form where the node has that rule and the product is closed, so that it agrees with
# the member's marginal; the Laplace approximation otherwise, as that marginal takes it.
function cluster_laplace(logp, members, incoming, own)
    fixed = map(is_point, incoming)
    any(fixed) || return joint_laplace(logp, incoming)
    free = Tuple(i for i in eachindex(incoming) if !fixed[i])
    point(v) = [fixed[i] ? oftype(first(v), mean(incoming[i])) : v[findfirst(isequal(i), free)] for i in eachindex(incoming)]
    density = length(free) == 1 ? marginal_of_one(x -> logp(point([x])), incoming[only(free)], own(only(free))) :
        joint_laplace(v -> logp(point(v)), map(i -> incoming[i], free))
    joint, outcome = MessagePassingRulesBase.unwrap_outcome(density)
    blocks = Pair{Tuple{Vararg{Symbol}}, Any}[]
    for i in eachindex(incoming)
        if fixed[i]
            push!(blocks, (members[i],) => incoming[i])
        elseif i == first(free)
            push!(blocks, Tuple(members[j] for j in free) => joint)
        end
    end
    cluster = FactorizedCluster(Val(Tuple(map(first, blocks))), Tuple(map(last, blocks)))
    return with_outcome(cluster, outcome)
end

# A single member's marginal: its message times the node's, in closed form, or by Laplace in its
# domain
function marginal_of_one(logp, density, message)
    if !isnothing(message)
        product = prod(BayesBase.GenericProd(), density, message)
        product isa Distribution{Univariate} && return with_outcome(product, (method = :closed_form,))
    end
    domain = something(EA.domain(density), EA.RealLine())
    result = EA.laplace(logp, mean(density), domain)
    outcome = (method = :joint_laplace, converged = result.converged, iterations = result.iterations)
    return with_outcome(exponential_family_gaussian(result.distribution), outcome)
end

# Laplace in the product of the members' domains, from their messages' means: on the real line
# ExponentialFamily's Gaussian, as `LaplaceForm` gives, with how the iterations went
function joint_laplace(logp, incoming)
    domains = map(density -> something(EA.domain(density), EA.RealLine()), incoming)
    domain = all(d -> d isa EA.RealLine, domains) ? EA.RealLine() : EA.Componentwise(domains...)
    x0 = [mean(density) for density in incoming]
    result = EA.laplace(logp, x0, domain)
    outcome = (method = :joint_laplace, converged = result.converged, iterations = result.iterations)
    return with_outcome(exponential_family_gaussian(result.distribution), outcome)
end

# Laplace for a vector, from its message's mean, in the message's domain
function vector_laplace(logp, density)
    domain = something(EA.domain(density), EA.RealLine())
    result = EA.laplace(logp, mean(density), domain)
    outcome = (method = :joint_laplace, converged = result.converged, iterations = result.iterations)
    return with_outcome(exponential_family_gaussian(result.distribution), outcome)
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

# The average energy -E_q[log f] of a stochastic node under its clusters' marginals; under cavity
# propagation with the cavity energy, with the node's inbound messages given, EP's term instead
function MessagePassingRulesBase.call_energy_fallback(fallback::ExpectationRuleFallback, node, algorithm, ctx, args)
    (isnothing(fallback.propagation) || fallback.energy === :average || isempty(rule_inputs(node, args.m))) ||
        return cavity_energy(fallback.propagation, node, ctx, args)
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

# The pairs `member => message` of every member but the `i`-th
pairs_except(members, incoming, i) = Tuple(members[j] => incoming[j] for j in eachindex(incoming) if j != i)

# The node's message towards `member` by its own rule, from the messages and marginals given as
# `name => distribution` pairs; `nothing` where it has no such rule
function own_message(node, member, algorithm, ctx, messages, marginals)
    member isa Symbol || return nothing
    args = RuleArgs(m = NamedTuple(messages), q = NamedTuple(marginals))
    found = find_message_rule(node, Target(member), algorithm, args)
    found isa RuleNotFound && return nothing
    return getresult(message_passing_rule(node, Target(member), algorithm, args, ctx))
end
