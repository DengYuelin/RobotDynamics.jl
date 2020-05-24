using RobotDynamics
using Test
using StaticArrays


model = Cartpole()
x,u = zeros(model)
@test sum(x) == 0
@test sum(u) == 0
x,u = rand(model)
@test sum(x) != 0
@test sum(u) != 0
xdot = dynamics(model, x, u)
@test sum(x) != 0

n,m = size(model)
dt = 0.1
F = zeros(n,n+m)
z = KnotPoint(x,u,dt)
jacobian!(F, model, z)
@test sum(F) != 0

D = RobotDynamics.DynamicsJacobian(n,m)
jacobian!(D, model, z)
@test D.A == F[:,1:n]
@test D.B ≈ F[:,n .+ (1:m)]

@test discrete_dynamics(RK3, model, x, u, 0.0, dt) ≈
    discrete_dynamics(RK3, model, z)
@test discrete_dynamics(RK3, model, z) ≈ discrete_dynamics(model, z)

F = zeros(n,n+m)
discrete_jacobian!(RK3, F, model, z)

tmp = [RobotDynamics.DynamicsJacobian(n,m) for k = 1:3]
jacobian!(RK3, D, model, z, tmp)
@test D.A ≈ F[1:n,1:n]
@test D.B ≈ F[1:n,n .+ (1:m)]
@test sum(F) != 0
@test F[1] == 1

# @btime discrete_jacobian!($RK3, $F, $model, $z)
# @btime jacobian!($RK3, $D, $model, $z, $tmp)
