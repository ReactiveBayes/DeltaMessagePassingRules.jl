# The rules of the Gaussian methods, most of whose routines come from ForneyLab.jl. The methods
# differ only in `approximate_normal` and `forward_statistics`, which MessagePassingRulesApproximations
# computes for each. The generalized unscented transform shares the joint over the inputs; its
# messages towards `out` and through an inverse take any univariate family (`rules/moments.jl`).

const GaussianApproximation = DeltaApproximation{<:GaussianMethod}
const NormalOnly = Union{Unscented, Linearization, GaussHermiteCubature}

# Each rule keeps the method's workspace in its scratch, so a call allocates only its message.
@define_message_update_rule(
    node = DeltaFn, target = :out, algorithm = DeltaApproximation{<:NormalOnly}, ctx = (:node,),
    args = (m[:in...]::NormalDistributionsFamily,),
    scratch = (algo, ctx, args) -> gaussian_scratch(algo.method, getnodefn(ctx.node, Target(:out)), args.m[:in]),
    body = (scratch, algo, ctx, args) -> approximate_normal!(scratch, algo.method, getnodefn(ctx.node, Target(:out)), args.m[:in]),
)

# The joint over the inputs: the forward statistics through the function, smoothed with the
# message from `out` (RTS, Petersen et al. 2018).
@define_marginal_update_rule(
    node = DeltaFn, target = (:in,), algorithm = GaussianApproximation, ctx = (:node,),
    args = (m[:out]::NormalDistributionsFamily, m[:in...]::NormalDistributionsFamily),
    scratch = (algo, ctx, args) -> gaussian_scratch(algo.method, getnodefn(ctx.node, Target(:out)), args.m[:in]),
    body = (scratch, algo, ctx, args) -> begin
        inputs = args.m[:in]
        μs_fw_in, Σs_fw_in = input_moments(inputs)
        μ_fw_in, Σ_fw_in = mean_cov(convert(JointNormal, μs_fw_in, Σs_fw_in))
        μ_tilde, Σ_tilde, C_tilde = scratch_statistics!(scratch, algo.method, getnodefn(ctx.node, Target(:out)), μs_fw_in, Σs_fw_in)
        μ_bw_out, Σ_bw_out = mean_cov(args.m[:out])
        μ_in, Σ_in = smoothRTS(μ_tilde, Σ_tilde, C_tilde, μ_fw_in, Σ_fw_in, μ_bw_out, Σ_bw_out)
        JointNormal(convert(promote_variate_type(typeof(μ_in), NormalMeanVariance), μ_in, Σ_in), size.(inputs))
    end,
)

# Without an inverse: the input's share of the joint, divided by its own message.
@define_message_update_rule(
    node = DeltaFn, target = (:in, k), algorithm = DeltaApproximation{<:GaussianMethod, Nothing},
    args = (m[:in][k]::NormalDistributionsFamily, q[(:in,)]::JointNormal),
    body = (args) -> begin
        ξ_in, Λ_in = weightedmean_precision(component(args.q[(:in,)], k))
        ξ_fw, Λ_fw = weightedmean_precision(args.m[:in][k])
        # The subtraction may leave a precision that is not positive definite.
        convert(promote_variate_type(typeof(ξ_in), NormalWeightedMeanPrecision), ξ_in - ξ_fw, Λ_in - Λ_fw)
    end,
)

# With an inverse: the message from `out` and the other inputs' messages, through it.
@define_message_update_rule(
    node = DeltaFn, target = (:in, k), algorithm = DeltaApproximation{<:NormalOnly, <:Union{Function, Tuple{Vararg{Function}}}},
    args = (m[:out]::NormalDistributionsFamily, m[:in][!k]::NormalDistributionsFamily),
    body = (algo, args) -> approximate_normal(algo.method, inverse_towards(algo, k), (args.m[:out], filter(!isnothing, args.m[:in])...)),
)
