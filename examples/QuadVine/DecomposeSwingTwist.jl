using LinearAlgebra, Rotations, StaticArrays

# swing after twist decompostion
# Algo 1 from https://arxiv.org/pdf/1506.05481.pdf
# angle axis from https://github.com/JuliaGeometry/Rotations.jl/blob/master/src/angleaxis_types.jl
function bend_twist_decomp(q)    
    # twist
    q_t = ((q.w!=0) || (q.z!=0)) ? 
        UnitQuaternion(q.w,0,0,q.z) : one(UnitQuaternion)
    θ_t = 2 * atan(q.z, q.w) 

    # swing
    aa = AngleAxis(q*conj(q_t))
    axis_b = rotation_axis(aa)
    θ_b = rotation_angle(aa)

    return θ_t, axis_b, θ_b
end

# positive twist
θ_t, axis_b, θ_b = bend_twist_decomp(UnitQuaternion(RotZ(pi)))
@assert θ_t ≈ pi
@assert θ_b ≈ 0

# negative twist
θ_t, axis_b, θ_b = bend_twist_decomp(UnitQuaternion(RotZ(-.2)))
@assert θ_t ≈ -.2
@assert θ_b ≈ 0

# positive bend
θ_t, axis_b, θ_b = bend_twist_decomp(UnitQuaternion(RotX(pi)))
@assert θ_t ≈ 0
@assert axis_b ≈ SA[1,0,0]
@assert θ_b ≈ pi

# negative bend
θ_t, axis_b, θ_b = bend_twist_decomp(UnitQuaternion(RotX(-.2)))
@assert θ_t ≈ 0
@assert axis_b ≈ SA[-1,0,0]
@assert θ_b ≈ .2

# combined
θ_t, axis_b, θ_b = bend_twist_decomp(UnitQuaternion(RotX(-.2)*RotZ(.2)))
@assert θ_t ≈ .2
@assert axis_b ≈ SA[-1,0,0]
@assert θ_b ≈ .2

# combined
θ_t, axis_b, θ_b = bend_twist_decomp(UnitQuaternion(RotX(.2)*RotZ(-.2)))
@assert θ_t ≈ -.2
@assert axis_b ≈ SA[1,0,0]
@assert θ_b ≈ .2

# consider two links
qa = rand(UnitQuaternion)
qb = rand(UnitQuaternion)
θ_ta, axis_ba, θ_ba = bend_twist_decomp(conj(qa)*qb) # qb in body frame a
θ_tb, axis_bb, θ_bb = bend_twist_decomp(conj(qb)*qa) # qa in body grame b
@assert θ_ta == -θ_tb
@assert conj(qb)*qa*axis_ba ≈ -axis_bb
@assert θ_ba == θ_bb

@assert AngleAxis(θ_bb, axis_bb...) ≈ AngleAxis(-θ_ba, (conj(qb)*qa*axis_ba)...)

function torques!(τa, τb, qa, qb, ωa, ωb, K, C)
    # decompostion in body a frame
    θ_ta, axis_ba, θ_ba = bend_twist_decomp(conj(qa)*qb)

    # twist stiffness
    τa[3] += K[3,3]*θ_ta
    τb[3] += -K[3,3]*θ_ta

    # bend stiffness
    τa .+= K[1,1]*θ_ba*axis_ba
    τb .+= -K[1,1]*θ_ba*(conj(qb)*qa*axis_ba)

    # damping
    τa .-= C*(ωa - conj(qa)*qb*ωb) # in body frame a
    τb .-= C*(ωb - conj(qb)*qa*ωa) # in body frame b
end

ωa = zeros(3)
ωb = zeros(3)
K = I(3)
C = I(3)

# twist only
qa = UnitQuaternion(RotZ(pi/8))
qb = UnitQuaternion(RotZ(pi/4))
τa = zeros(3)
τb = zeros(3)
torques!(τa, τb, qa, qb, ωa, ωb, K, C)
@assert τa ≈ [0,0,pi/8]
@assert τb ≈ -[0,0,pi/8]

# bending only
qa = UnitQuaternion(AngleAxis(pi/8, 1, 1, 0))
qb = UnitQuaternion(AngleAxis(pi/4, 1, 1, 0))
τa = zeros(3)
τb = zeros(3)
torques!(τa, τb, qa, qb, ωa, ωb, K, C)
@assert τa ≈ [pi/8/sqrt(2), pi/8/sqrt(2), 0]
@assert τb ≈ -[pi/8/sqrt(2), pi/8/sqrt(2), 0]

# damping only
qa = one(UnitQuaternion)
qb = one(UnitQuaternion)
ωa = rand(3)
ωb = rand(3)
τa = zeros(3)
τb = zeros(3)
torques!(τa, τb, qa, qb, ωa, ωb, K, C)
@assert τa ≈ ωb-ωa
@assert τb ≈ ωa-ωb

# combined
qa = one(UnitQuaternion)
qb = UnitQuaternion(RotX(pi/2))
ωa = zeros(3)
ωb = [1.,0,0]
τa = zeros(3)
τb = zeros(3)
torques!(τa, τb, qa, qb, ωa, ωb, K, C)
@assert τa ≈ [pi/2,1,0] # fix
@assert τb ≈ [-1,pi/2,0] # fix