include("QuadVine.jl")
include("Quad_vis.jl")

# model and timing
model = QuadVine(2) 
nq, nv, nc = mc_dims(model)
n,m = size(model)
N = 500
dt = 5e-3
tf = (N-1)*dt     

# initial and final conditions
x0 = [generate_config(model, zeros(model.nb)); zeros(nv)]
shift_pos!(model, x0, [0,0,0])
@assert norm(max_constraints(model, x0)) < 1e-6

xf = [generate_config(model, RotX.([0, -.4, -.8])); zeros(nv)]
shift_pos!(model, xf, [.5,.5,0])
@assert norm(max_constraints(model, xf)) < 1e-6

# objective
Qf = Diagonal(SVector{n}([fill(250., nq); fill(250., nv)]))
Q = Diagonal(SVector{n}([fill(1e-4, nq); fill(1e-4, nv)]))
R = Diagonal(@SVector fill(2e-2/dt, m))
costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf), SVector{m}(trim_controls(model)); w=1e-4) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf), SVector{m}(trim_controls(model)); w=250.0)
obj = Objective(costfuns);

# problem
prob = Problem(model, obj, xf, tf, x0=x0);

# initial controls
u0 = trim_controls(model)
U0 = [SVector{m}(u0) for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);
plot_traj(states(prob), controls(prob))
# visualize!(model, states(prob), dt)

# ILQR
# using TimerOutputs
# TimerOutputs.enable_debug_timings(Altro)
opts = SolverOptions(verbose=7, static_bp=0, iterations=3, cost_tolerance=1e-4)
ilqr = Altro.iLQRSolver(prob, opts);
set_options!(ilqr, iterations=10, cost_tolerance=1e-6)

# solve
# reset_timer!(ilqr.stats.to)
solve!(ilqr);
# show(ilqr.stats.to)

# results
# X,U = states(ilqr), controls(ilqr)
plot_traj(states(ilqr), controls(ilqr))
visualize!(model, states(ilqr), dt)

# # display(plot(hcat(Vector.(Vec.(ilqr.K[1:end]))...)',legend=false))
# # display(plot(hcat(Vector.(Vec.(ilqr.d[1:end]))...)',legend=false))
