using RobotZoo, RobotDynamics
using TrajectoryOptimization
using Rotations
using ForwardDiff
using StaticArrays, LinearAlgebra
using BenchmarkTools
using Altro
using Plots
using SparseArrays

const TO = TrajectoryOptimization
const RD = RobotDynamics
const RS = Rotations

include("helper.jl")

struct QuadVine{R,T} <: LieGroupModelMC{R}
    quadrotor::RobotZoo.Quadrotor{R}
    masses::Array{T,1}
    lengths::Array{T,1}
    radii::Array{T,1} 
    inertias::Array{Diagonal{T,Array{T,1}},1}

    g::T # gravity
    nb::Int # number of rigid bodies
    p::Int # constraint force dimension
    liestate::RD.LieState
 
    function QuadVine{R,T}(masses, lengths, radii, inertias) where {R<:Rotation, T<:Real} 
        @assert length(masses) == length(lengths) == length(radii) == length(inertias)
        nb = length(masses)
        quadrotor = RobotZoo.Quadrotor{R}(;
            mass=masses[1],
            J=Diagonal(SVector{3}(inertias[1].diag)),
            gravity=SVector(0,0,-9.81),
            motor_dist=0.1750,
            kf=1.0,
            km=0.0245,
            bodyframe=false)
        liestate = RD.LieState(R,(fill(3, nb)..., Int(6*nb)))
        new(quadrotor, masses, lengths, radii, inertias, 9.81, nb, 3*(nb-1),liestate)
    end 
end
QuadVine(n) = QuadVine{UnitQuaternion{Float64},Float64}(ones(n+1), [.2; ones(n)], [1;fill(.1,n)], fill(Diagonal(ones(3)),n+1))
QuadVine() = QuadVine(2)

Altro.config_size(model::QuadVine) = 7*model.nb
Lie_P(model::QuadVine) = (fill(3, model.nb)..., Int(6*model.nb))
RD.LieState(model::QuadVine{R}) where R = model.liestate
RD.control_dim(model::QuadVine) = 7

function trim_controls(model)
    _,m = size(model)
    total_mass = sum(model.masses)
    g, kf = model.g, model.quadrotor.kf
    [fill(total_mass*g/(4kf), 4);zeros(m-4)]
end

function shift_pos!(model::QuadVine{R}, x, shift) where R
    P = Lie_P(model)
    for i=1:model.nb
        inds = RD.vec_inds(R,P,i)
        x[inds] += shift
    end
end

function shift_pos(model::QuadVine{R}, x, shift) where R
    _x = copy(x)
    shift_pos!(model,_x, shift)
    return _x
end

function max_constraints(model::QuadVine{R}, x) where R
    nq = config_size(model)
    P = Lie_P(model)
    # lie = RD.LieState(UnitQuaternion{eltype(x)}, (P[1:end-1]..., 0))

    # pos = RD.vec_states(model.liestate, x) 
    # rot = RD.rot_states(model.liestate, x) 
    pos, rot = get_configs(model, x)

    l = model.lengths
    c = zeros(eltype(x), model.p)

    # other links endpoint
    for i=2:model.nb
        d1 = UnitQuaternion(rot[i-1], false) * [0;0;-l[i-1]/2]
        d2 = UnitQuaternion(rot[i], false) * [0;0;-l[i]/2]
        c[3*(i-2) .+ (1:3)] = (pos[i-1] + d1) - (pos[i] - d2)
    end
    
    return c
end

function Altro.is_converged(model::QuadVine, x)
    c = max_constraints(model, x)
    return norm(c) < 1e-6
end

# function max_constraints_jacobian(model::QuadVine, x⁺::Vector{T}) where T
#     nq, nv, _ = mc_dims(model)
#     c_aug(x) = max_constraints(model, x)
#     J_big = ForwardDiff.jacobian(c_aug, x⁺[1:nq])

#     links = length(model.masses)
#     n,m = size(model)
#     n̄ = state_diff_size(model)
#     G = SizedMatrix{n,n̄}(zeros(T,n,n̄))
#     RD.state_diff_jacobian!(G, RD.LieState(UnitQuaternion{T}, Lie_P(model)) , SVector{n}(x⁺))
#     return J_big*G[1:nq,1:nv]
# end

function max_constraints_jacobian(model::QuadVine{R}, x) where R
    T = eltype(x)
    nb = model.nb
    nq, nv, _ = mc_dims(model)
    P = Lie_P(model)
    l = model.lengths
    d = zeros(T, 3)
    # rot = RD.rot_states(RD.LieState(UnitQuaternion{T}, Lie_P(model)), x) # use this if using ForwardDiff
    rot = RD.rot_states(model.liestate, x) # use this if not using ForwardDiff
    J = zeros(T, nc, nv)
    for i=1:nb-1
        # shift vals
        row = 3*(i-1)
        col = 6*(i-1)

        # ∂c∂qa
        qa = rot[i]
        d[3] = -l[i]/2
        J[row .+ (1:3), col .+ (4:6)] = RS.∇rotate(qa, d)*RS.∇differential(qa)

        # ∂c∂qb
        qb = rot[i+1]
        d[3] = -l[i+1]/2
        J[row .+ (1:3), col .+ (10:12)] = RS.∇rotate(qb, d)*RS.∇differential(qb)
        
        for j=1:3
            J[row+j, col+j] = 1 # ∂c∂xa = I
            J[row+j, col+6+j] = -1 # ∂c∂xb = -I
        end
    end
    return J
end

import RobotZoo: forces, moments
function RobotZoo.forces(model::RobotZoo.Quadrotor, x, u)
    q = orientation(model, x)
    kf = model.kf
    g = model.gravity
    m = model.mass

    w1 = u[1]
    w2 = u[2]
    w3 = u[3]
    w4 = u[4]

    # F1 = max(0,kf*w1);
    # F2 = max(0,kf*w2);
    # F3 = max(0,kf*w3);
    # F4 = max(0,kf*w4);
    F1 = kf*w1;
    F2 = kf*w2;
    F3 = kf*w3;
    F4 = kf*w4;
    F = @SVector [0., 0., F1+F2+F3+F4] #total rotor force in body frame

    m*g + q*F # forces in world frame
end

function RobotZoo.moments(model::RobotZoo.Quadrotor, x, u)

    kf, km = model.kf, model.km
    L = model.motor_dist

    w1 = u[1]
    w2 = u[2]
    w3 = u[3]
    w4 = u[4]

    # F1 = max(0,kf*w1);
    # F2 = max(0,kf*w2);
    # F3 = max(0,kf*w3);
    # F4 = max(0,kf*w4);
    F1 = kf*w1;
    F2 = kf*w2;
    F3 = kf*w3;
    F4 = kf*w4;

    M1 = km*w1;
    M2 = km*w2;
    M3 = km*w3;
    M4 = km*w4;
    tau = @SVector [L*(F2-F4), L*(F3-F1), (M1-M2+M3-M4)] #total rotor torque in body frame
end

function forces(model::QuadVine, x⁺, x, u)
    nb = model.nb
    g = model.g
    return [RobotZoo.forces(model.quadrotor, x, u), # quad
            [[0;0;-model.masses[i]*g] for i=2:nb]...] # splat vine
end

function torques(model::QuadVine, x⁺, x, u)
    nb = model.nb
    return [RobotZoo.moments(model.quadrotor, x, u)-u[5:7],
            [u[5:7] for i=1:nb-1]...]
end

function get_configs(model::QuadVine, x)
    nb = model.nb
    vec = [x[7*(i-1) .+ (1:3)] for i=1:nb]
    rot = [SVector{4}(x[7*(i-1) .+ (4:7)]) for i=1:nb]
    return vec, rot
end

function get_vels(model::QuadVine, x)
    nb = model.nb
    nq = config_size(model)   
    vs = [x[nq+6*(i-1) .+ (1:3)] for i=1:nb]
    ωs = [x[nq+6*(i-1) .+ (4:6)] for i=1:nb]
    return vs, ωs
end

function propagate_config!(model::QuadVine{R}, x⁺::Vector{T}, x, dt) where {R, T}
    nb = model.nb
    nq, nv, _ = mc_dims(model)
    P = Lie_P(model)

    # vec = RD.vec_states(model, x)
    # rot = RD.rot_states(RD.LieState(UnitQuaternion{T}, P), x)
    # rot = RD.rot_states(model.liestate, x)  
    vec, rot = get_configs(model,x)
    vs⁺, ωs⁺ = get_vels(model, x⁺)
    for i=1:nb
        vind = RD.vec_inds(R, P, i)
        x⁺[vind] = vec[i] + vs⁺[i]*dt
        rind = RD.rot_inds(R, P, i)
        x⁺[rind] = dt/2 * RS.lmult(rot[i]) * [sqrt(4/dt^2 - ωs⁺[i]'ωs⁺[i]); ωs⁺[i]]
    end

    return 
end

function propagate_config(model::QuadVine, x⁺, x, dt)
    x⁺ = copy(x⁺)
    propagate_config!(model, x⁺, x, dt)
    nq = config_size(model)
    return x⁺[1:nq]
end

function f_pos(model::QuadVine, x⁺, x, u, λ, dt)
    nq = config_size(model)
    return x⁺[1:nq] - propagate_config(model, x⁺, x, dt)
end

function f_vel(model::QuadVine, x⁺, x, u, λ, dt)
    J = max_constraints_jacobian(model, x⁺)
    
    nb = model.nb
    nq, nv, nc = mc_dims(model)

    ms = model.masses
    Is = model.inertias

    vs, ωs = get_vels(model, x)
    vs⁺, ωs⁺ = get_vels(model, x⁺)

    fs = forces(model,x⁺,x,u) 
    τs = torques(model,x⁺,x,u)

    f_vels = zeros(eltype(x⁺), 6*nb)
    for i=1:nb
        # translational
        f_vels[6*(i-1) .+ (1:3)] = ms[i]*(vs⁺[i]-vs[i])/dt - fs[i]

        # rotational
        ω⁺ = ωs⁺[i]
        ω = ωs[i]
        sq⁺ = sqrt(4/dt^2-ω⁺'ω⁺)
        sq = sqrt(4/dt^2-ω'ω)
        f_vels[6*(i-1) .+ (4:6)] = sq⁺*Is[i]*ω⁺ - sq*Is[i]*ω + cross(ω⁺,Is[i]*ω⁺) + cross(ω,Is[i]*ω) - 2*τs[i]
    end

    return f_vels - J'λ
end

function fc(model::QuadVine, x⁺, x, u, λ, dt)
    f = f_vel(model, x⁺, x, u, λ, dt)
    c = max_constraints(model, x⁺)
    return [f;c]
end

function fc_jacobian!(F, model::QuadVine, x⁺, x, u, λ, dt)
    nq, nv, nc = mc_dims(model)
    nb = model.nb

    # function fc_aug(s)
    #     # Unpack
    #     _x⁺ = convert(Array{eltype(s)}, x⁺)
    #     _x⁺[nq .+ (1:nv)] = s[1:nv]
    #     _λ = s[nv .+ (1:nc)]

    #     propagate_config!(model, _x⁺, x, dt)
    #     fc(model, _x⁺, x, u, _λ, dt)
    # end
    # ForwardDiff.jacobian(fc_aug, [x⁺[nq .+ (1:nv)];λ])

    ls = model.lengths
    ms = model.masses
    Is = model.inertias
    d = zeros(3)
    # rot = RD.rot_states(RD.LieState(UnitQuaternion{eltype(x)}, Lie_P(model)), x)
    rot = RD.rot_states(model.liestate, x)

    vs⁺, ωs⁺ = get_vels(model, x⁺)

    for i=1:nb
        shift = 6*(i-1)

        # df_vel/dv⁺
        F[shift .+ (1:3), shift .+ (1:3)] = ms[i]/dt*I(3)

        # df_vel/dω⁺
        f_vel1_aug(_ω⁺) = sqrt(4/dt^2-_ω⁺'_ω⁺)*Is[i]*_ω⁺ + cross(_ω⁺,Is[i]*_ω⁺) 
        if i==1
            F[shift .+ (4:6), shift .+ (4:6)] = ForwardDiff.jacobian(f_vel1_aug, ωs⁺[i])
        else
            F[shift .+ (4:6), shift .+ (4:6)] += ForwardDiff.jacobian(f_vel1_aug, ωs⁺[i])
        end

        i == nb && continue

        r_shift = 3*(i-1)
        λt = λ[r_shift .+ (1:3)]

        # dg/dv⁺
        F[nv + r_shift .+ (1:3), shift .+ (1:3)] = dt*I(3) # body 1
        F[nv + r_shift .+ (1:3), shift + 6 .+ (1:3)] = -dt*I(3) # body 2

        ## rotation derivatives with body i
        d[3] = -ls[i]/2
        rot⁺, ∇rot, dqdw = dq⁺dw⁺(ωs⁺[i], rot[i], d, dt) 

        # dg/dω⁺
        F[nv + r_shift .+ (1:3), shift .+ (4:6)] = ∇rot*dqdw
        
        # dJ'λ/dω
        F[shift .+ (4:6), shift .+ (4:6)] -= ∂Gqbᵀλ∂qb(rot⁺,λt,d)*dqdw
        
        ## rotation derivatives with body i+1
        d[3] = -ls[i+1]/2
        rot⁺, ∇rot, dqdw = dq⁺dw⁺(ωs⁺[i+1], rot[i+1], d, dt) 

        # dg/dω⁺
        F[nv + r_shift .+ (1:3), shift + 6 .+ (4:6)] = ∇rot*dqdw

        # dJ'λ/dω
        F[shift+6 .+ (4:6), shift+6 .+ (4:6)] = -∂Gqbᵀλ∂qb(rot⁺,λt,d)*dqdw
        
    end

    F[1:nv, nv+1:end] = -max_constraints_jacobian(model, x⁺)'
end

function line_step!(model::QuadVine, x⁺_new, λ_new, x⁺, λ, Δs, x, dt)
    nq, nv, nc = mc_dims(model)
    
    # update lambda
    Δλ = Δs[nv .+ (1:nc)]
    λ_new .= λ - Δλ

    # update v⁺
    Δv⁺ = Δs[1:nv]
    x⁺_new[nq .+ (1:nv)] .= x⁺[nq .+ (1:nv)] - Δv⁺    

    # compute configuration from v⁺
    propagate_config!(model, x⁺_new, x, dt)
    return    
end

# function adjust_step_size!(model::QuadVine, x⁺, Δs)
#     return
#     Δs ./= maximum(abs.(Δs[1:6]))/100
#     _, ωs = get_vels(model, x⁺)
#     Δω1, Δω2 = Δs[1:3], Δs[4:6]
#     γ = 1.0
#     while (ω1-γ*Δω1)'*(ω1-γ*Δω1) > (1/dt^2)
#         # print("%")
#         γ /= 2
#     end
    
#     while (ω2-γ*Δω2)'*(ω2-γ*Δω2) > (1/dt^2)
#         # print("%")
#         γ /= 2
#     end
#     Δs .= γ*Δs
# end

function Altro.discrete_dynamics_MC(::Type{Q}, model::QuadVine, 
    x, u, t, dt) where {Q<:RobotDynamics.Explicit}
  
    nq, nv, nc = mc_dims(model)

    # initial guess
    λ = zeros(nc)
    x⁺ = Vector(x)
    Δs = zeros(nc+nv)
    F = zeros(nc+nv, nc+nv)
    
    x⁺_new, λ_new = copy(x⁺), copy(λ)

    max_iters, line_iters, ϵ = 100, 20, 1e-6
    for i=1:max_iters  
        # print("iter ", i, ": ")

        # Newton step    
        err_vec = fc(model, x⁺, x, u, λ, dt)
        err = norm(err_vec)
        fc_jacobian!(F, model, x⁺, x, u, λ, dt)
        Δs .= sparse(F)\err_vec
        # Δs .= F\err_vec
        # ldiv!(Δs, factorize(F), err_vec)

        # line search
        j=0
        err_new = err + 1        
        while (err_new > err) && (j < line_iters)
            # print("-")
            # adjust_step_size!(model, x⁺, Δs)
            line_step!(model, x⁺_new, λ_new, x⁺, λ, Δs, x, dt)
            _, ωs⁺ = get_vels(model, x⁺_new)
            if all(1/dt^2 .>= dot(ωs⁺,ωs⁺))
                # print("!")
                err_new = norm(fc(model, x⁺_new, x, u, λ_new, dt))
            end
            Δs /= 2
            j += 1
        end
        # println(" steps: ", j)
        x⁺ .= x⁺_new
        λ .= λ_new

        # convergence check
        if err_new < ϵ
            return x⁺, λ
        end
    end

    # throw("Newton did not converge. ")
    println("Newton did not converge. ")
    return x⁺, λ
end

function RD.discrete_dynamics(::Type{Q}, model::QuadVine, x, u, t, dt) where Q
    x, λ = Altro.discrete_dynamics_MC(Q, model,  x, u, t, dt)
    return x
end

# function Altro.discrete_jacobian_MC!(::Type{Q}, Dexp, model::QuadVine,
#     z::AbstractKnotPoint{T,N,M′}, x⁺, λ) where {T,N,M′,Q<:RobotDynamics.Explicit}
    
#     all_partials, ∇f, G = Dexp.all_partials, Dexp.∇f, Dexp.G

#     n,m = size(model)
#     n̄ = state_diff_size(model)
#     nq, nv, nc = mc_dims(model)

#     x = state(z) 
#     u = control(z)
#     dt = z.dt
#     @assert dt != 0

#     # compute next state and lagrange multiplier
#     x⁺, λ = Altro.discrete_dynamics_MC(Q, model, x, u, z.t, dt)

#     function f_imp(z)
#         # Unpack
#         _x⁺ = z[1:(nq+nv)]
#         _x = z[(nq+nv) .+ (1:nq+nv)]
#         _u = z[2*(nq+nv) .+ (1:m)]
#         _λ = z[2*(nq+nv)+m .+ (1:nc)]
#         return [f_pos(model, _x⁺, _x, _u, _λ, dt); f_vel(model,  _x⁺, _x, _u, _λ, dt)]
#     end

#     all_partials = ForwardDiff.jacobian(f_imp, [x⁺;x;u;λ])
#     ∇f .= -all_partials[:,1:n]\all_partials[:,n+1:end]

#     G[:,1:n̄-nv] .= max_constraints_jacobian(model, x⁺)
# end

function f_vel_jacobian!(model::QuadVine, a_p, x⁺, x, u, λ, dt)    
    nb = model.nb
    nq, nv, nc = mc_dims(model)

    ms = model.masses
    Is = model.inertias

    vs, ωs = get_vels(model, x)
    vs⁺, ωs⁺ = get_vels(model, x⁺)

    fs = forces(model,x⁺,x,u) 
    τs = torques(model,x⁺,x,u)

    # f_vel_aug(_x⁺) = -max_constraints_jacobian(model, _x⁺)'*λ
    # a_p[nq .+ (1:nv),1:nq] = ForwardDiff.jacobian(f_vel_aug, x⁺[1:nq])
    rot⁺ = [SVector{4}(x⁺[7*(i-1) .+ (4:7)]) for i=1:nb]
    d = zeros(3)
    for i=1:nb-1
        r_shift = nq + 6*(i-1)
        c_shift = 7*(i-1)
        λt = λ[3*(i-1) .+ (1:3)]

        # link i
        d[3] = -model.lengths[i]/2
        if i == 1
            a_p[r_shift .+ (4:6), c_shift .+ (4:7)] = -∂Gqbᵀλ∂qb(rot⁺[i], λt, d)
        else
            a_p[r_shift .+ (4:6), c_shift .+ (4:7)] -= ∂Gqbᵀλ∂qb(rot⁺[i], λt, d)
        end

        # link i+1
        d[3] = -model.lengths[i+1]/2
        a_p[r_shift+6 .+ (4:6), c_shift+7 .+ (4:7)] = -∂Gqbᵀλ∂qb(rot⁺[i+1], λt, d) 
    end

    for i=1:nb
        r_shift = nq + 6*(i-1)

        # df_vel/dv⁺ = m/dt*I
        c_shift = 6*(i-1) + nq
        a_p[r_shift+1, c_shift+1] = a_p[r_shift+2, c_shift+2] = a_p[r_shift+3, c_shift+3] = ms[i]/dt

        # df_vel/dv = -m/dt*I
        c_shift = 6*(i-1) + n + nq
        a_p[r_shift+1, c_shift+1] = a_p[r_shift+2, c_shift+2] = a_p[r_shift+3, c_shift+3] = -ms[i]/dt

        # df_vel/dw⁺ 
        f_vel1_aug(_ω⁺) = sqrt(4/dt^2-_ω⁺'_ω⁺)*Is[i]*_ω⁺ + cross(_ω⁺,Is[i]*_ω⁺) 
        c_shift = 6*(i-1) + nq
        a_p[r_shift .+ (4:6), c_shift .+ (4:6)] = ForwardDiff.jacobian(f_vel1_aug, ωs⁺[i])

        # df_vel/dw 
        f_vel2_aug(_ω) = - sqrt(4/dt^2-_ω'_ω)*Is[i]*_ω  + cross(_ω,Is[i]*_ω)
        c_shift = 6*(i-1) + n + nq
        a_p[r_shift .+ (4:6), c_shift .+ (4:6)] = ForwardDiff.jacobian(f_vel2_aug, ωs[i])

        # df_vel/du
        if i == 1 # drone
            fτ_aug(z) = [-1*RobotZoo.forces(model.quadrotor, z[1:7], z[8:11]); 
                        -2*RobotZoo.moments(model.quadrotor, z[1:7], z[8:11])]
            jac = ForwardDiff.jacobian(fτ_aug, [x[1:7]; u[1:4]])
            a_p[r_shift .+ (1:6), n .+ (1:7)] = jac[1:6, 1:7] # df_dx
            a_p[r_shift .+ (1:6), 2n .+ (1:4)] = jac[1:6, 8:11] # df_du 1:4
            a_p[r_shift .+ (4:6), 2n .+ (5:7)] = 2*Matrix(I,3,3) # df_du 5:7
        else # vine
            a_p[r_shift .+ (4:6), 2n .+ (5:7)] = -2*Matrix(I,3,3) 
        end
    end
end

function Altro.discrete_jacobian_MC!(::Type{Q}, Dexp, model::QuadVine,
    z::AbstractKnotPoint{T,N,M′}, x⁺, λ) where {T,N,M′,Q<:RobotDynamics.Explicit}
    
    all_partials, ∇f, G = Dexp.all_partials, Dexp.∇f, Dexp.G

    n,m = size(model)
    n̄ = state_diff_size(model)
    nq, nv, nc = mc_dims(model)

    x = state(z) 
    u = control(z)
    dt = z.dt
    @assert dt != 0

    # compute next state and lagrange multiplier
    # x⁺, λ = discrete_dynamics_MC(Q, model, x, u, z.t, dt)

    # top half of all partials
    # d(f_pos)d(x⁺)
    all_partials[diagind(all_partials)[1:nq]] .= 1

    # d(f_pos)d([vel⁺; pos])
    tmp_zeros = zeros(nv)
    function prop_config_aug(z) 
        # Unpack
        _x⁺ = [x⁺[1:nq];z[1:nv]]
        _x = [z[nv .+ (1:nq)];tmp_zeros]
        return -1*propagate_config(model, _x⁺, _x, dt)
    end
    f_pos_view = view(all_partials, 1:nq, nq .+ (1:n))
    vel⁺ = x⁺[nq .+ (1:nv)]
    pos = x[1:nq]
    ForwardDiff.jacobian!(f_pos_view, prop_config_aug, [vel⁺; pos])

    # bottom half of all_partials
    # function f_vel_aug(z)
    #     # Unpack
    #     _x⁺ = z[1:(nq+nv)]
    #     _x = z[(nq+nv) .+ (1:nq+nv)]
    #     _u = z[2*(nq+nv) .+ (1:m)]
    #     _λ = z[2*(nq+nv)+m .+ (1:nc)]
    #     return f_vel(model,  _x⁺, _x, _u, _λ, dt)
    # end
    # f_vel_view = view(all_partials, nq .+ (1:nv), 1:(2n+m+nc))
    # ForwardDiff.jacobian!(f_vel_view, f_vel_aug, [x⁺;x;u;λ])

    G[:,1:n̄-nv] .= max_constraints_jacobian(model, x⁺)

    # bottom half of all_partials
    f_vel_jacobian!(model, all_partials, x⁺, x, u, λ, dt) 
    all_partials[nq .+ (1:nv), (2n+m) .+ (1:nc)] = -G[:, 1:nv]'

    # implicit function theorem
    ∇f .= -sparse(all_partials[:,1:n])\all_partials[:,n+1:end]
    # ∇f .= -all_partials[:,1:n]\all_partials[:,n+1:end]
end

# compute maximal coordinate configuration given body rotations
function generate_config(model, rotations)
    @assert model.nb == length(rotations)
    pin = zeros(3)
    q = zeros(0)   
    l = model.lengths
    for i = 1:model.nb
        r = UnitQuaternion(rotations[i]...)
        delta = r * [0,0,-l[i]/2]
        q = [q; pin+delta; Rotations.params(r)]
        pin += 2*delta
    end
    return q
end

function generate_config(model, θ::Vector{<:Number})
    rotations = UnitQuaternion.(RotX.(θ))
    return generate_config(model, rotations)
end

function interpolate_config(model, shift, dx, aa::AngleAxis, k, N)
    # shift: xyz shift for all configs
    # dx: final xyz shift of drone
    # aa: final rotation of vine links
    # k: start index for rotating vine
    # N: length of trajectory
    N -= 1
    nq, nv, nc = mc_dims(model)
    X_track = map(0:N) do i 
        angle = i < k ? 0 : aa.theta*(i-k)/(N-k)
        rots = fill(AngleAxis(angle, aa.axis_x, aa.axis_y, aa.axis_z, false), model.nb)
        rots[1] = one(UnitQuaternion)       
        x = [generate_config(model, rots); zeros(nv)]
        shift_pos!(model, x, shift+dx*i/N)
        x
    end
    return X_track    
end

## ROLLOUT
function quick_rollout(model, x0, u, dt, N)
    X = [x0]
    xnext = x0
    for i=1:N
        z = KnotPoint(xnext, u, dt)
        xnext = RD.discrete_dynamics(PassThrough, model, z)
        push!(X, xnext)
    end
    return X
end

function plot_traj(X, U)
    N = length(X)
    quats1 = [X[i][4:7] for i=1:N]
    quats2 = [X[i][7 .+ (4:7)] for i=1:N]

    q1mat = hcat(quats1...)'
    q2mat = hcat(quats2...)'
    Umat = hcat(Vector.(U)...)'

    display(plot(q1mat))
    display(plot(q2mat))
    display(plot(Umat))

    return q1mat, q2mat, Umat
end

function plot_angles(X, U)
    N = length(X)
    quats1 = [UnitQuaternion(X[i][4:7]...) for i=1:N]
    quats2 = [UnitQuaternion(X[i][7 .+ (4:7)]...) for i=1:N]
    angles1 = [rotation_angle(quats1[i])*rotation_axis(quats1[i])[1] for i=1:N]
    angles2 = [rotation_angle(quats2[i])*rotation_axis(quats2[i])[1] for i=1:N]

    plot(angles1, label = "θ1",xlabel="time step",ylabel="state")
    plt = plot!(angles2,  label = "θ2")
    display(plt)

    Umat = hcat(Vector.(U)...)'
    display(plot(Umat))
end
