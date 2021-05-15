include("QuadVine2.jl")
include("Quad_vis.jl")

using SparseArrays

model = QuadVine(2)
nq, nv, nc = mc_dims(model)
n,m = size(model)
N = 2
dt = 5e-3
x0 = [generate_config(model, rand(model.nb)); .1*rand(nv)]
u0 = .1*rand(m)
x⁺, λ = Altro.discrete_dynamics_MC(PassThrough, model, x0, u0, 0., dt)

## JACOBIAN
n,m = size(model)
n̄ = RD.state_diff_size(model)

DExp = TO.DynamicsExpansionMC(model)
diff1 = SizedMatrix{n,n̄}(zeros(n,n̄))
RD.state_diff_jacobian!(diff1, RD.LieState(model), SVector{n}(x0))
diff2 = SizedMatrix{n,n̄}(zeros(n,n̄))
RD.state_diff_jacobian!(diff2, RD.LieState(model), SVector{n}(x0))
z = KnotPoint(x0,u0,dt)
Altro.discrete_jacobian_MC!(PassThrough, DExp, model, z, x⁺, λ)
# TO.save_tmp!(DExp)
# TO.error_expansion!(DExp, diff1, diff2)
# A,B,C,G = TO.error_expansion(DExp, model)

function f_imp(z)
    # Unpack
    _x⁺ = z[1:(nq+nv)]
    _x = z[(nq+nv) .+ (1:nq+nv)]
    _u = z[2*(nq+nv) .+ (1:m)]
    _λ = z[2*(nq+nv)+m .+ (1:nc)]
    return [f_pos(model, _x⁺, _x, _u, _λ, dt); f_vel(model,  _x⁺, _x, _u, _λ, dt)]
end
auto_partials = ForwardDiff.jacobian(f_imp, [x⁺;z.z;λ])
DExp.all_partials ≈ auto_partials
display(spy(sparse(round.(DExp.all_partials - auto_partials,digits=5)), marker=2, legend=nothing, c=palette([:black], 2)))

#-----------------------------------------------
using SparseArrays

function jl_aug(s)
    # Unpack
    _x⁺ = convert(Array{eltype(s)}, x⁺)
    _x⁺[nq .+ (1:nv)] = s[1:nv]

    propagate_config!(model, _x⁺, x0, dt)
    -max_constraints_jacobian(model, _x⁺)'λ
end
# jl_jac= ForwardDiff.jacobian(jl_aug, x⁺[nq .+ (1:nv)])

function fc_aug(s)
    # Unpack
    _x⁺ = convert(Array{eltype(s)}, x⁺)
    _x⁺[nq .+ (1:nv)] = s[1:nv]
    _λ = s[nv .+ (1:nc)]

    propagate_config!(model, _x⁺, x0, dt)
    fc(model, _x⁺, x0, u0, _λ, dt)
end
fc_jac = ForwardDiff.jacobian(fc_aug, [x⁺[nq .+ (1:nv)];λ])

F = zeros(nc+nv,nc+nv)
fc_jacobian!(F, model, x⁺, x0, u0, λ, dt)
extrema(F-fc_jac)
# display(spy(sparse(round.(F-fc_jac,digits=5)), marker=2, legend=nothing, c=palette([:black], 2)))

#-----------------------------------------------
ilqr=altro
n,m,N = size(ilqr)
J = Inf
_J = TO.get_J(ilqr.obj)
J_prev = sum(_J)
grad_only = false
to = ilqr.stats.to
init = ilqr.opts.reuse_jacobians  # force recalculation if not reusing
@timeit_debug to "diff jac"     TO.state_diff_jacobian!(ilqr.G, ilqr.model, ilqr.Z)
@timeit_debug to "dynamics jac" TO.dynamics_expansion!(Altro.integration(ilqr), ilqr.D, ilqr.model, ilqr.Z, ilqr.Λ)
@timeit_debug to "err jac"      TO.error_expansion!(ilqr.D, ilqr.model, ilqr.G)
@timeit_debug to "cost exp"     TO.cost_expansion!(ilqr.quad_obj, ilqr.obj, ilqr.Z, init, true)
@timeit_debug to "cost err"     TO.error_expansion!(ilqr.E, ilqr.quad_obj, ilqr.model, ilqr.Z, ilqr.G)
@timeit_debug to "backward pass" ΔV = Altro.backwardpass!(ilqr)
display(plot(hcat(Vector.(Vec.(ilqr.K[1:end]))...)',legend=false))
display(plot(hcat(Vector.(Vec.(ilqr.d[1:end]))...)',legend=false))

@timeit_debug to "forward pass" Altro.forwardpass!(ilqr, ΔV, J_prev)

U = controls(ilqr)
display(plot(hcat(Vector.(U)...)',xlabel="timestep",ylabel="controls"))

# timing -----------------------

using STATProfilerHTML
[f_vel(model, x0, x0, u0, rand(model.p), dt) for i=1:300]
@profilehtml [f_vel(model, x0, x0, u0, rand(model.p), dt) for i=1:300]
F = zeros(n+model.p,n+model.p)
[fc_jacobian!(F, model, x⁺, x, u, λ, dt) for i=1:300]
@profilehtml [fc_jacobian!(F, model, x⁺, x, u, λ, dt) for i=1:300]