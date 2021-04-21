using RobotDynamics
using Rotations
using ForwardDiff
using StaticArrays, LinearAlgebra
using SparseArrays
using BenchmarkTools
using Test
using Altro
using TrajectoryOptimization
using Plots
using ConstrainedDynamics
using ConstrainedDynamicsVis
using ConstrainedControl

const TO = TrajectoryOptimization
const RD = RobotDynamics
const RS = Rotations
const CD = ConstrainedDynamics
const CDV = ConstrainedDynamicsVis
const CC = ConstrainedControl

# the robot is body_link     =>      arm_1     ==>     arm_2    ...   ==>      arm_nb 
# t                        joint1             joint2       ...     joint_nb
# the arm extends along positive x direction
struct FloatingSpace{R,T} <: LieGroupModelMC{R}
    body_mass::T
    body_size::T
    arm_mass::T
    arm_width::T
    arm_length::T 
    body_inertias::Diagonal{T,Array{T,1}}
    arm_inertias::Diagonal{T,Array{T,1}}
    joint_directions::Array{Array{T,1},1}
    joint_vertices::Array{Array{T,1},1}   # this is a term used by Jan 
                                          # each joint has two vertices, so joint_vertices[i] is a 6x1 vector

    g::T

    nb::Integer     # number of arm links, total rigid bodies in the system will be nb+1
    p::Integer      # the body has no constraint, then each one more link brings in 5 more constraints because all joints are rotational
    ns::Integer     # total state size 13*(nb+1)

    function FloatingSpace{R,T}(nb,_joint_directions) where {R<:Rotation, T<:Real}
        m0 = 10.0
        m1 = 1.0
        g = 0      # in space, no gravity!
        body_size = 0.5
        arm_length = 1.0
        joint_vertices = [[body_size/2, 0, 0, -arm_length/2, 0, 0]]
        for i=1:nb
            push!(joint_vertices,[arm_length/2, 0, 0, -arm_length/2, 0, 0])
        end


        new(m0, body_size, m1, 0.1, arm_length, 
            Diagonal(1 / 12 * m0 * diagm([0.5^2 + 0.5^2;0.5^2 + 0.5^2;0.5^2 + 0.5^2])),  # body inertia
            Diagonal(1 / 12 * m1 * diagm([0.1^2 + 0.1^2;0.1^2 + 1.0^2;1.0^2 + 0.1^2])),   # arm inertia
            _joint_directions,
            joint_vertices,
            0, # space robot, no gravity
            nb, 5*nb,       # 5 because we assume all joints are revolute joints
            13*(nb+1)
        )
    end
    function FloatingSpace()
        _joint_directions = [[0.0;0.0;1.0]]
        FloatingSpace{UnitQuaternion{Float64},Float64}(1,_joint_directions)
    end

    # all z axis rotation
    function FloatingSpace(nb::Integer)
        @assert nb >= 1
        _joint_directions = [[0.0;0.0;1.0] for i=1:nb]
        FloatingSpace{UnitQuaternion{Float64},Float64}(nb,_joint_directions)
    end
end 

# joint axes are orthogonal to each other #space snake?
# Z -> Y -> Z -> Y ...
function FloatingSpaceOrth(nb::Integer)
    @assert nb >= 1
    _joint_directions = [ i % 2 == 0 ? [0.0;1.0;0.0] : [0.0;0.0;1.0] for i=1:nb]
    FloatingSpace{UnitQuaternion{Float64},Float64}(nb,_joint_directions)
end


# arrange state as Jan
# x v q w, x v q w,...
#1,2,3, 4,5,6, 7,8,9,10, 11,12,13
Altro.config_size(model::FloatingSpace) = 7*(model.nb+1)
Lie_P(model::FloatingSpace) = (6,fill(9, model.nb)..., 3)
RD.LieState(model::FloatingSpace{R}) where R = RD.LieState(R,Lie_P(model))
RD.control_dim(model::FloatingSpace) = model.nb + 6  # 6 because we assume the body is fully actuated

# extract the x v q w indices of ith link
function fullargsinds(model::FloatingSpace, i)
    # x, v, q, ω
    return 13*(i-1) .+ (1:3), 
            13*(i-1) .+ (4:6), 
            13*(i-1) .+ (7:10), 
            13*(i-1) .+ (11:13)
end

begin
    # basic construct test
    a = FloatingSpace()
    @test Altro.config_size(a) == 14
    println(Lie_P(a))
end

# state config x v q w 
# the body pose is the position and z orientation of the body link, then rotations are rotation matrices of joint angles
function generate_config(model::FloatingSpace, body_pose, rotations)
    # com position of the body link 
    pin = zeros(3)
    pin[1] = body_pose[1]
    pin[2] = body_pose[2]
    pin[3] = body_pose[3]
    prev_q = UnitQuaternion(RotZ(body_pose[4]))   # TODO: improve body_pose to contain full rotations?

    state = [pin;zeros(3);RS.params(prev_q);zeros(3)]
    pin = pin+prev_q * [model.body_size/2,0,0]   
    for i = 1:length(rotations)
        r = UnitQuaternion(rotations[i])
        link_q = prev_q * r
        delta = link_q * [model.arm_length/2,0,0] # assume all arms have equal length
        link_x = pin+delta
        state = [state; link_x;zeros(3);RS.params(link_q);zeros(3)]

        prev_q = link_q
        pin += 2*delta
    end
    return state
end

# body pose is x y z thetaz, θ is just angle 
function generate_config(model::FloatingSpace, body_pose::Vector{<:Number}, θ::Vector{<:Number})
    @assert length(θ) == model.nb
    rotations = []
    for i=1:length(θ)
        axis = model.joint_directions[i]
        push!(rotations, 
            UnitQuaternion(AngleAxis(θ[i], axis[1], axis[2], axis[3]))
        )
    end
    return generate_config(model, body_pose, rotations)
end

### add random velocity to generated state
# state config x v q w 
# the body pose is the position and z orientation of the body link, then rotations are rotation matrices of joint angles
function generate_config_with_rand_vel(model::FloatingSpace, body_pose, rotations)
    # com position of the body link 
    pin = zeros(3)
    pin[1] = body_pose[1]
    pin[2] = body_pose[2]
    pin[3] = body_pose[3]
    prev_q = UnitQuaternion(RotZ(body_pose[4]))   # TODO: improve body_pose to contain full rotations?

    state = [pin;0.01*randn(3);RS.params(prev_q);0.01*randn(3)]
    pin = pin+prev_q * [model.body_size/2,0,0]   
    for i = 1:length(rotations)
        r = UnitQuaternion(rotations[i])
        link_q = prev_q * r
        delta = link_q * [model.arm_length/2,0,0] # assume all arms have equal length
        link_x = pin+delta
        state = [state; link_x;0.01*randn(3);RS.params(link_q);0.01*randn(3)]

        prev_q = link_q
        pin += 2*delta
    end
    return state
end

# body pose is x y z thetaz, θ is just angle 
function generate_config_with_rand_vel(model::FloatingSpace, body_pose::Vector{<:Number}, θ::Vector{<:Number})
    @assert length(θ) == model.nb
    rotations = []
    for i=1:length(θ)
        axis = model.joint_directions[i]
        push!(rotations, 
            UnitQuaternion(AngleAxis(θ[i], axis[1], axis[2], axis[3]))
        )
    end
    return generate_config_with_rand_vel(model, body_pose, rotations)
end

begin
    # basic state genereation test
    model = FloatingSpace()
    x0 = generate_config(model, [2.0;2.0;1.0;0], [0])
    println(x0)
end

# this function returns a mech object, which is a constrainedDynamics object so that we can visualize the robot using 
# constraineddynamics viz
function vis_mech_generation(model::FloatingSpace)
    origin = CD.Origin{Float64}()
    link0 = CD.Box(model.body_size, model.body_size, model.body_size, 1., color = CD.RGBA(1., 1., 0.))
    link0.m = model.body_mass # set base mass
    world2base = CD.EqualityConstraint(Floating(origin, link0)) # free floating
    
    arm_links = [CD.Box(model.arm_length, model.arm_width, model.arm_width, 1.,color = CD.RGBA(0.1*i, 0.2*i, 1.0/i)) for i = 1:model.nb]
    # joint 1 
    vert01 = model.joint_vertices[1][1:3] # connection offset from body to joint1
    vert11 = model.joint_vertices[1][4:6] # connection offset from arm_link1 to joint1

    joint1 = CD.EqualityConstraint(CD.Revolute(link0, arm_links[1], model.joint_directions[1]; p1=vert01,p2=vert11)) # joint1 : body to link1

    links = [link0; arm_links]
    constraints = [world2base; joint1]
    if model.nb > 1
        for i=2:model.nb
            vert01 = model.joint_vertices[i][1:3] # connection offset from armi-1 to jointi
            vert11 = model.joint_vertices[i][4:6] # connection offset from armi to jointi
        
            jointi = CD.EqualityConstraint(CD.Revolute(arm_links[i-1], arm_links[i], model.joint_directions[i]; p1=vert01,p2=vert11)) # joint1 : armi-1 to larmi
            constraints = [constraints;jointi]
        end
    end
    mech = CD.Mechanism(origin, links, constraints, g=-model.g)
    return mech
end

function setStates!(model::FloatingSpace, mech, z)
    for (i, body) in enumerate(mech.bodies)   
        xinds, vinds, qinds, ωinds = fullargsinds(model,i)   
        setPosition!(body; x = SVector{3}(z[xinds]), q = UnitQuaternion(z[qinds]...))
        setVelocity!(body; v = SVector{3}(z[vinds]), ω = SVector{3}(z[ωinds]))
    end
end

# test: visualize 
begin
    model = FloatingSpaceOrth(2)
    x0 = generate_config(model, [2.0;2.0;1.0;pi/2], [pi/4,pi/4]);
    println(reshape(x0,(13,model.nb+1))')
    mech = vis_mech_generation(model)
    setStates!(model,mech,x0)
    steps = Base.OneTo(1)
    storage = CD.Storage{Float64}(steps,length(mech.bodies))
    for i=1:model.nb+1
        storage.x[i][1] = mech.bodies[i].state.xc
        storage.v[i][1] = mech.bodies[i].state.vc
        storage.q[i][1] = mech.bodies[i].state.qc
        storage.ω[i][1] = mech.bodies[i].state.ωc
    end
    visualize(mech,storage, env = "editor")
end

# the position constraint g
function g(model::FloatingSpace, x)
    # we have nb joints, so the dimension of constraint is p=5*nb
    g_val = zeros(eltype(x),model.p)
    for i=2:model.nb+1   # i is the rigidbody index
        r_ainds, v_ainds, q_ainds, w_ainds = fullargsinds(model, i-1) # a is the previous rigid body
        r_binds, v_binds, q_binds, w_binds = fullargsinds(model, i)   # b is the next rigid body
        r_a = SVector{3}(x[r_ainds])
        r_b = SVector{3}(x[r_binds])
        q_a = SVector{4}(x[q_ainds])
        q_b = SVector{4}(x[q_binds])

        val = view(g_val, (5*(i-2)).+(1:5))
        vertex1 = model.joint_vertices[i-1][1:3]
        vertex2 = model.joint_vertices[i-1][4:6]

        val[1:3] = (r_b + RS.vmat()*RS.rmult(q_b)'*RS.lmult(q_b)*RS.hmat()*vertex2) - 
        (r_a + RS.vmat()*RS.rmult(q_a)'*RS.lmult(q_a)*RS.hmat()*vertex1)
        tmp = RS.vmat()*RS.lmult(q_a)'*q_b
        # the joint constraint map, it depends on the joint rotation direction 
        cmat = [0 0 1; 
                1 0 0]
        if model.joint_directions[i-1] == [0,0,1]
            cmat = [0 1 0; 
                   1 0 0]
        else
            cmat = [0 0 1; 
                    1 0 0]
        end
        val[4:5] = cmat*tmp  
    end
    return g_val
end

# jacobian of g, treat quaternion as normal 4 vectors
function Dg(model::FloatingSpace, x)
    Dgmtx = spzeros(model.p,model.ns)
    for i=2:model.nb+1   # i is the rigidbody index
        r_ainds, v_ainds, q_ainds, w_ainds = fullargsinds(model, i-1) # a is the previous rigid body
        r_binds, v_binds, q_binds, w_binds = fullargsinds(model, i)   # b is the next rigid body

        vertex1 = model.joint_vertices[i-1][1:3]
        vertex2 = model.joint_vertices[i-1][4:6]
        cmat = [0 0 1; 
                1 0 0]
        if model.joint_directions[i-1] == [0,0,1]
            cmat = [0 1 0; 
                   1 0 0]
        else
            cmat = [0 0 1; 
                    1 0 0]
        end

        Dgblock = view(Dgmtx, (5*(i-2)).+(1:5),:)

        q_a = SVector{4}(x[q_ainds])
        q_b = SVector{4}(x[q_binds])
        Dgblock[:,r_ainds] = [-I;zeros(2,3)]  # dg/dra
        Dgblock[:,r_binds]  = [I;zeros(2,3)] # dg/drb
        Dgblock[:,q_ainds] = [-2*RS.vmat()*RS.rmult(q_a)'*RS.rmult(RS.hmat()*vertex1);
                                -cmat*RS.vmat()*RS.lmult(q_b)'
                               ]
        Dgblock[:,q_binds] = [2*RS.vmat()*RS.rmult(q_b)'*RS.rmult(RS.hmat()*vertex2);
                                cmat*RS.vmat()*RS.lmult(q_a)'
                               ]
    end
    return Dgmtx
end

# This is similar to g, but we need to propogate state
function gp1(model::FloatingSpace, x, dt)
    g_val = zeros(eltype(x),model.p)
    for i=2:model.nb+1   # i is the rigidbody index
        r_ainds, v_ainds, q_ainds, w_ainds = fullargsinds(model, i-1) # a is the previous rigid body
        r_binds, v_binds, q_binds, w_binds = fullargsinds(model, i)   # b is the next rigid body
        r_a = SVector{3}(x[r_ainds])
        v_a = SVector{3}(x[v_ainds])
        r_b = SVector{3}(x[r_binds])
        v_b = SVector{3}(x[v_binds])
        q_a = SVector{4}(x[q_ainds])
        w_a = SVector{3}(x[w_ainds])
        q_b = SVector{4}(x[q_binds])
        w_b = SVector{3}(x[w_binds])
        # propagate states 
        r_a1 = r_a + v_a*dt
        r_b1 = r_b + v_b*dt
    
        q_a1 = dt/2*RS.lmult(q_a)*SVector{4}([sqrt(4/dt^2 -w_a'*w_a);w_a])
        q_b1 = dt/2*RS.lmult(q_b)*SVector{4}([sqrt(4/dt^2 -w_b'*w_b);w_b])

        # impose constraint on r_a1, r_b1, q_a1, q_b1
        val = view(g_val, (5*(i-2)).+(1:5))
        vertex1 = model.joint_vertices[i-1][1:3]
        vertex2 = model.joint_vertices[i-1][4:6]

        val[1:3] = (r_b1 + RS.vmat()*RS.rmult(q_b1)'*RS.lmult(q_b1)*RS.hmat()*vertex2) - 
                   (r_a1 + RS.vmat()*RS.rmult(q_a1)'*RS.lmult(q_a1)*RS.hmat()*vertex1)
        tmp = RS.vmat()*RS.lmult(q_a1)'*q_b1
        # the joint constraint map, it depends on the joint rotation direction 
        cmat = [0 0 1; 
                1 0 0]
        if model.joint_directions[i-1] == [0,0,1]
            cmat = [0 1 0; 
                   1 0 0]
        else
            cmat = [0 0 1; 
                    1 0 0]
        end
        val[4:5] = cmat*tmp  
    end
    return g_val
end
# function Dgp1, the jacobian of gp1
# jacobian of gp1, treat quaternion as normal 4 vectors
function Dgp1(model::FloatingSpace, x, dt)
    Dgmtx = spzeros(model.p,model.ns)
    for i=2:model.nb+1   # i is the rigidbody index
        r_ainds, v_ainds, q_ainds, w_ainds = fullargsinds(model, i-1) # a is the previous rigid body
        r_binds, v_binds, q_binds, w_binds = fullargsinds(model, i)   # b is the next rigid body
        r_a = SVector{3}(x[r_ainds])
        v_a = SVector{3}(x[v_ainds])
        r_b = SVector{3}(x[r_binds])
        v_b = SVector{3}(x[v_binds])
        q_a = SVector{4}(x[q_ainds])
        w_a = SVector{3}(x[w_ainds])
        q_b = SVector{4}(x[q_binds])
        w_b = SVector{3}(x[w_binds])
        # propagate states 
        r_a1 = r_a + v_a*dt
        r_b1 = r_b + v_b*dt
    
        q_a1 = dt/2*RS.lmult(q_a)*SVector{4}([sqrt(4/dt^2 -w_a'*w_a);w_a])
        q_b1 = dt/2*RS.lmult(q_b)*SVector{4}([sqrt(4/dt^2 -w_b'*w_b);w_b])


        vertex1 = model.joint_vertices[i-1][1:3]
        vertex2 = model.joint_vertices[i-1][4:6]
        cmat = [0 0 1; 
                1 0 0]
        if model.joint_directions[i-1] == [0,0,1]
            cmat = [0 1 0; 
                   1 0 0]
        else
            cmat = [0 0 1; 
                    1 0 0]
        end

        Dgblock = view(Dgmtx, (5*(i-2)).+(1:5),:)

        ∂dgp1∂dra1 = [-I;zeros(2,3)]
        ∂dgp1∂drb1 = [ I;zeros(2,3)]
        ∂dgp1∂dqa1 = [-2*RS.vmat()*RS.rmult(q_a1)'*RS.rmult(RS.hmat()*vertex1);
                      -cmat*RS.vmat()*RS.lmult(q_b)'
                    ]
        ∂dgp1∂dqb1 =[2*RS.vmat()*RS.rmult(q_b1)'*RS.rmult(RS.hmat()*vertex2);
                        cmat*RS.vmat()*RS.lmult(q_a)'
                    ]
        ∂dra1∂dva = I(3)*dt
        ∂drb1∂dvb = I(3)*dt   
        ∂dqa1∂dqa = dt/2*RS.rmult(SVector{4}([sqrt(4/dt^2 -w_a'*w_a);w_a]))      
        ∂dqa1∂dwa = dt/2*(-q_a*w_a'/sqrt(4/dt^2 -w_a'*w_a) + RS.lmult(q_a)*RS.hmat())    

        ∂dqb1∂dqb = dt/2*RS.rmult(SVector{4}([sqrt(4/dt^2 -w_b'*w_b);w_b]))      
        ∂dqb1∂dwb = dt/2*(-q_b*w_b'/sqrt(4/dt^2 -w_b'*w_b) + RS.lmult(q_b)*RS.hmat())  

        Dgblock[:,13*0 .+ (1:3)] =  ∂dgp1∂dra1 # dg/dra
        Dgblock[:,13*0 .+ (4:6)] =  ∂dgp1∂dra1*∂dra1∂dva# dg/dva

        Dgblock[:,13*1 .+ (1:3)]  = ∂dgp1∂drb1  # dg/drb
        Dgblock[:,13*1 .+ (4:6)]  =  ∂dgp1∂drb1*∂drb1∂dvb# dg/dvb

        Dgblock[:,13*0 .+ (7:10)] = ∂dgp1∂dqa1*∂dqa1∂dqa# dg/dqa
        Dgblock[:,13*0 .+ (11:13)] = ∂dgp1∂dqa1*∂dqa1∂dwa# dg/dwa
        Dgblock[:,13*1 .+ (7:10)] =  ∂dgp1∂dqb1*∂dqb1∂dqb# dg/dqb
        Dgblock[:,13*1 .+ (11:13)] =  ∂dgp1∂dqb1*∂dqb1∂dwb# dg/dwb
    end
    return Dgmtx
end

# this calculates a part of Dg*attiG, only related to G_qa , dim is 5x3
function Gqa(q_a::SArray{Tuple{4},Float64,1,4},q_b::SArray{Tuple{4},Float64,1,4},vertices, joint_direction)  
    vertex1 = vertices[1:3]
    vertex2 = vertices[4:6]
    cmat = [0 1.0 0; 
            1 0 0]
    if joint_direction == [0,0,1]
        cmat = [0 1.0 0; 
               1 0 0]
    else
        cmat = [0 0 1.0; 
                1 0 0]
    end
    Dgmtx = [-2*RS.vmat()*RS.rmult(q_a)'*RS.rmult(RS.hmat()*vertex1);
             -cmat*RS.vmat()*RS.lmult(q_b)'
            ]
    return Dgmtx*RS.lmult(q_a)*RS.hmat()
end

# this calculates a part of Dg*attiG, only related to G_qb, dim is 5x3
function Gqb(q_a::SArray{Tuple{4},Float64,1,4},q_b::SArray{Tuple{4},Float64,1,4},vertices, joint_direction)  
    vertex1 = vertices[1:3]
    vertex2 = vertices[4:6]
    cmat = [0 0 1; 
            1 0 0]
    if joint_direction == [0,0,1]
        cmat = [0 1 0; 
               1 0 0]
    else
        cmat = [0 0 1; 
                1 0 0]
    end
    Dgmtx = [2*RS.vmat()*RS.rmult(q_b)'*RS.rmult(RS.hmat()*vertex2);
             cmat*RS.vmat()*RS.lmult(q_a)'
            ]
    return Dgmtx*RS.lmult(q_b)*RS.hmat()
end

# test: constraint g
begin
    model = FloatingSpace()
    x0 = generate_config(model, [2.0;2.0;1.0;pi/2], [pi/2]);
    # gval = g(model,x0)
    # println(gval)
    # Dgmtx = Dg(model,x0)
    # println(Dgmtx)

    # TODO, test gp1 and Dgp1
    gval = gp1(model,x0,0.01)
    println(gval)
    Dp1gmtx = Dgp1(model,x0,0.01)
    println(Dgmtx)
    gp1aug(z) = gp1(model,z,0.01)
    Dgp1forward = ForwardDiff.jacobian(gp1aug,x0)
    @test Dgp1forward ≈ Dp1gmtx

    q_a = UnitQuaternion(RotX(0.03))
    q_b = UnitQuaternion(RotY(0.03))
    vertices = [1,2,3,4,5,6]
    joint_direction = [0,0,1]
    @show joint_direction == [0,0,1]
    Gqa(RS.params(q_a),RS.params(q_b),vertices, joint_direction) 
    Gqb(RS.params(q_a),RS.params(q_b),vertices, joint_direction) 
end

function state_diff_jac(model::FloatingSpace,x::Vector{T}) where T
    n,m = size(model)
    n̄ = state_diff_size(model)

    G = SizedMatrix{n,n̄}(zeros(T,n,n̄))
    RD.state_diff_jacobian!(G, RD.LieState(UnitQuaternion{T}, Lie_P(model)) , SVector{n}(x))
    
    return G
end

# test state_diff_jac
begin
    model = FloatingSpace()
    n,m = size(model)
    n̄ = state_diff_size(model)
    @show n
    @show n̄

    x0 = generate_config(model, [2.0;2.0;1.0;pi/2], [pi/2]);
    sparse(state_diff_jac(model, x0))
end

# implicity dynamics function fdyn
# return f(x_t1, x_t, u_t, λt) = 0
# TODO: 5 and 13 are sort of magic number that should be put in constraint
# TODO: what if all masses of links are different
function fdyn(model::FloatingSpace,xt1, xt, ut, λt, dt)
    fdyn_vec = zeros(eltype(xt1),model.ns)
    u_joint = ut[7:end]
    for link_id=1:model.nb+1
        fdyn_vec_block = view(fdyn_vec, (13*(link_id-1)).+(1:13))
        joint_before_id = link_id-1
        joint_after_id  = link_id
        # iterate through all rigid bodies
        r_ainds, v_ainds, q_ainds, w_ainds = fullargsinds(model, link_id) # a is the current link

        # get state from xt1
        rat1 = xt1[r_ainds]
        vat1 = xt1[v_ainds]
        qat1 = SVector{4}(xt1[q_ainds])
        wat1 = xt1[w_ainds]
        # get state from xt
        rat = xt[r_ainds]
        vat = xt[v_ainds]
        qat = SVector{4}(xt[q_ainds])
        wat = xt[w_ainds]

        # link_id==1 (the body) need special attention 
        # link_id==nb+1 (the last arm link)
        if (link_id == 1)  #the body link
            # get next link state from xt1
            r_binds, v_binds, q_binds, w_binds = fullargsinds(model, link_id+1) # b is the next link
            rbt1 = xt1[r_binds]
            qbt1 = SVector{4}(xt1[q_binds])

            # only the body link use these forces and torques
            Ft = ut[1:3]
            taut = ut[4:6]
            tau_joint = u_joint[joint_after_id]
            λt_block = λt[(5*(link_id-1)).+(1:5)]
            # position
            fdyn_vec_block[1:3] = rat1 - (rat + vat*dt)

            # velocity
            Ma = diagm([model.body_mass,model.body_mass,model.body_mass])
            aa = Ma*(vat1-vat) + Ma*[0;0;model.g]*dt
            fdyn_vec_block[4:6] =  aa - Ft*dt - [-I(3);zeros(2,3)]'*λt_block*dt   # Gra'λ

            # orientation
            fdyn_vec_block[7:10] = qat1 - dt/2*RS.lmult(qat)*SVector{4}([sqrt(4/dt^2 -wat'*wat);wat])

            # angular velocity
            vertices = model.joint_vertices[joint_after_id] # notice joint_vertices is 6x1
            joint_direction = model.joint_directions[joint_after_id]
            Gqamtx = Gqa(qat1,qbt1,vertices, joint_direction)  
            Ja = model.body_inertias
            a = Ja * wat1 * sqrt(4/dt^2 -wat1'*wat1) + cross(wat1, (Ja * wat1)) - Ja * wat  * sqrt(4/dt^2 - wat'*wat) + cross(wat,(Ja * wat))
            k = - 2*taut + 2*tau_joint*joint_direction - Gqamtx'*λt_block
            fdyn_vec_block[11:13] = a+k

        elseif (link_id >= 2 && link_id < model.nb+1) # normal arm link
            # get next link state from xt1
            r_binds, v_binds, q_binds, w_binds = fullargsinds(model, link_id+1) # b is the next link
            rbt1 = xt1[r_binds]
            qbt1 = SVector{4}(xt1[q_binds])
            # get previous link state from xt1
            r_zinds, v_zinds, q_zinds, w_zinds = fullargsinds(model, link_id-1) # z is the previous link
            rzt1 = xt1[r_zinds]
            qzt1 = SVector{4}(xt1[q_zinds])

            next_tau_joint = u_joint[joint_after_id]   # next == after
            prev_tau_joint = u_joint[joint_before_id]  # perv == before

            next_λt_block = λt[(5*(link_id-1)).+(1:5)]
            prev_λt_block = λt[(5*(joint_before_id-1)).+(1:5)]

            # position
            fdyn_vec_block[1:3] = rat1 - (rat + vat*dt)
            # velocity 
            Ma = diagm([model.arm_mass,model.arm_mass,model.arm_mass])
            aa = Ma*(vat1-vat) + Ma*[0;0;model.g]*dt
            fdyn_vec_block[4:6] =  aa - [-I(3);zeros(2,3)]'*next_λt_block*dt   
                                -  [I(3);zeros(2,3)]'*prev_λt_block*dt
            # orientation
            fdyn_vec_block[7:10] = qat1 - dt/2*RS.lmult(qat)*SVector{4}([sqrt(4/dt^2 -wat'*wat);wat])
            # angular velocity (need to add previous joint constraint)
            # joint between a and b # use Gra
            next_vertices = model.joint_vertices[joint_after_id] # notice joint_vertices is 6x1
            next_joint_direction = model.joint_directions[joint_after_id]
            Gqamtx = Gqa(qat1,qbt1,next_vertices, next_joint_direction)  
            # joint between z and a  # use Grb
            prev_vertices = model.joint_vertices[joint_before_id] # notice joint_vertices is 6x1
            prev_joint_direction = model.joint_directions[joint_before_id]
            Gqzmtx = Gqb(qzt1,qat1,prev_vertices, prev_joint_direction)  


            Ja = model.arm_inertias
            a = Ja * wat1 * sqrt(4/dt^2 -wat1'*wat1) + cross(wat1, (Ja * wat1)) - Ja * wat  * sqrt(4/dt^2 - wat'*wat) + cross(wat,(Ja * wat))
            k =  - 2*prev_tau_joint*prev_joint_direction
                 + 2*next_tau_joint*next_joint_direction - Gqamtx'*next_λt_block - Gqzmtx'*prev_λt_block
                 fdyn_vec_block[11:13] = a+k

        else # the last link 
            # get previous link state from xt1
            r_zinds, v_zinds, q_zinds, w_zinds = fullargsinds(model, link_id-1) # z is the previous link
            rzt1 = xt1[r_zinds]
            qzt1 = SVector{4}(xt1[q_zinds])
            prev_tau_joint = u_joint[joint_before_id]  # perv == before
            prev_λt_block = λt[(5*(joint_before_id-1)).+(1:5)]
            # position
            fdyn_vec_block[1:3] = rat1 - (rat + vat*dt)
            # velocity (only different from link_id == 1 is no force, and different mass)
            Ma = diagm([model.arm_mass,model.arm_mass,model.arm_mass])
            aa = Ma*(vat1-vat) + Ma*[0;0;model.g]*dt
            fdyn_vec_block[4:6] =  aa -  [I(3);zeros(2,3)]'*prev_λt_block*dt
            # orientation
            fdyn_vec_block[7:10] = qat1 - dt/2*RS.lmult(qat)*SVector{4}([sqrt(4/dt^2 -wat'*wat);wat])
            # angular velocity (need to add previous joint constraint)
            # joint between z and a 
            prev_vertices = model.joint_vertices[joint_before_id] # notice joint_vertices is 6x1
            prev_joint_direction = model.joint_directions[joint_before_id]
            Gqzmtx = Gqb(qzt1,qat1,prev_vertices, prev_joint_direction) 

            Ja = model.arm_inertias
            a = Ja * wat1 * sqrt(4/dt^2 -wat1'*wat1) + cross(wat1, (Ja * wat1)) - Ja * wat  * sqrt(4/dt^2 - wat'*wat) + cross(wat,(Ja * wat))
            k =  - 2*prev_tau_joint*prev_joint_direction
                 - Gqzmtx'*prev_λt_block
            fdyn_vec_block[11:13] = a+k

        end
    end
    return fdyn_vec
end
# TODO: function Dfdyn
# TODO: function attiG_f
# TODO: discrete_dynamics!

# test dynamics

begin
    using Random
    Random.seed!(123)
    model = FloatingSpace()
    x0 = generate_config_with_rand_vel(model, [2.0;2.0;1.0;pi/4], [pi/4])
    dr = pi/14
    x1 = generate_config_with_rand_vel(model, [2.0;2.0;1.0;pi/4+dr], [pi/4+dr]);
    u = 2*randn(6+model.nb)
    du = 0.01*randn(6+model.nb)
    λ = randn(5*model.nb)
    dλ = 0.001*randn(5*model.nb)
    dxv = zeros(model.ns)
    dxv[(13*0).+(4:6)] = randn(3)
    dxv[(13*0).+(11:13)] = randn(3)
    dxv[(13*1).+(4:6)] = randn(3)
    dxv[(13*1).+(11:13)] = randn(3)
    dt = 0.01
    f1 = fdyn(model,x1, x0, u, λ, dt)
    @show f1
    f2 = fdyn(model,x1+dxv, x0+dxv, u+du, λ+dλ, dt)
end