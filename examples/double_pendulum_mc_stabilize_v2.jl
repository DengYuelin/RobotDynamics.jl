# NOTE: run double_pendulum_mc.jl first

#goal 
th1 = -pi/4
th2 = -pi/4
d1 = .5*model.l1*[cos(th1);sin(th1)]
d2 = .5*model.l2*[cos(th2);sin(th2)]
xf = [d1; th1; 2*d1 + d2; th2; zeros(6)]

#costs
Q = zeros(n,n)
Q[3,3] = 1e-3
Q[6,6] = 1e-3
Q[9,9] = 1e-3
Q[12,12] = 1e-3
Qf = zeros(n,n)
Qf[3,3] = 250
Qf[6,6] = 250
Qf[9,9] = 250
Qf[12,12] = 250
R = 1e-4*Matrix(I,m,m)

function f(x,u,dt)
    z = KnotPoint(x,u,dt)
    return discrete_dynamics_MC(PassThrough, model, z)
end

function getABCG(x⁺,x,u,λ,dt)
    z = KnotPoint(x,u,dt)
    return discrete_jacobian_MC(PassThrough, model, z)
end

function backwardpass(X,Lam,U,F,Q,R,Qf,xf)
    n, N = size(X)
    m = size(U,1)
    nc = 4
    
    S = zeros(n,n,N)
    s = zeros(n,N)    
    K_list = zeros(m,n,N-1)
    d_list = zeros(m,N-1)
    
    S[:,:,N] = Qf
    s[:,N] = Qf*(X[:,N] - xf)
    v1 = 0.0
    v2 = 0.0

    mu = 0.0
    k = N-1
    
    while k >= 1
        q = Q*(X[:,k] - xf)
        r = R*(U[:,k])
        S⁺ = S[:,:,k+1]
        s⁺ = s[:,k+1]
        
        A,B,C,G = F(X[:,k+1],X[:,k],U[:,k],Lam[:,k],dt)
        
        D = B - C/(G*C)*G*B

        Qx = q + A'*s⁺
        Qu = r + B'*s⁺
        Qλ = C'*s⁺
        Qux = B'*S⁺*A
        Quu = R + B'*S⁺*B
        Quλ = B'*S⁺*C
        Qxx = Q + A'*S⁺*A
        Qxu = A'*S⁺*B
        Qxλ = A'*S⁺*C
        Qλx = C'*S⁺*A
        Qλu = C'*S⁺*B
        Qλλ = C'*S⁺*C
        
        M = [Quu Quλ; G*B G*C]
        b = [-Qux;-G*A]
        l = [-Qu;zeros(nc)]
        K_all = M\b
        K = K_all[1:m,:]
        Kλ = K_all[m+1:m+nc,:]
        K_list[:,:,k] = K
        l_all = M\l
        d = l_all[1:m,:]
        lλ = l_all[m+1:m+nc,:]
        d_list[:,k] = d

        S[:,:,k] = Qxx + 2*Qxλ*Kλ + Kλ'*Qλλ*Kλ + K'*Quu*K + 2*Qxu*K + 2*Kλ'*Qλu*K
        s[:,k] = Qx + K'*Qu + Kλ'*Qλ + K'*Quu*d + Qxu*d + Kλ'*Qλu*d


        # M11 = R + D'*S⁺*B
        # M12 = D'*S⁺*C
        # M21 = G*B
        # M22 = G*C

        # M = [M11 M12;M21 M22]
        # b = [D'*S⁺;G]*A

        # K_all = M\b
        # Ku = K_all[1:m,:]
        # Kλ = K_all[m+1:m+nc,:]
        # K[:,:,k] = Ku

        # l_all = M\[r + D'*s⁺; zeros(nc)]
        # lu = l_all[1:m,:]
        # lλ = l_all[m+1:m+nc,:]
        # l[:,k] = lu

        # Abar = A-B*Ku-C*Kλ
        # bbar = -B*lu - C*lλ
        # S[:,:,k] = Q + Ku'*R*Ku + Abar'*S⁺*Abar
        # s[:,k] = q - Ku'*r + Ku'*R*lu + Abar'*S⁺*bbar + Abar'*s⁺

        k = k - 1;
    end
    return K_list, d_list, v1, v2
end

function stable_rollout(Ku,x0,u0,f,dt,tf)
    N = convert(Int64,floor(tf/dt))
    X = zeros(size(x0,1),N)
    U = zeros(m,N-1)
    Lam = zeros(4,N-1)
    X[:,1] = x0
    for k = 1:N-1
        U[:,k] = u0+Ku*(X[:,k]-xf)
        X[:,k+1], Lam[:,k] = f(X[:,k],U[:,k],dt)
    end
    return X, Lam, U
end

#simulation
dt = 0.01
tf = 6.0

# compute and verify nominal torques
m1, m2, g = model.m1, model.m2, model.g
uf = [(m1*xf[1] + m2*xf[4])*g; (xf[4] - 2*xf[1])*m2*g]
xf′, λf = f(xf,uf,dt) # check xf′ = xf

# compute stabilizing gains
timesteps = 300
X = repeat(xf,outer=(1,timesteps+1))
Lam = repeat(λf,outer=(1,timesteps))
U = repeat(uf,outer=(1,timesteps))
K, l, v1, v2 = backwardpass(X,Lam,U,getABCG,Q,R,Q,xf)
K6 = [K[1,6,i] for i=1:timesteps]
K3 = [K[1,3,i] for i=1:timesteps]
plot([K3 K6])

# run stablizing controller
Ku = K[:,:,1]
x1, _ = f(xf,[5., 0],dt) # perturbance
X, Lam, U=stable_rollout(Ku,x1,U[:,1],f,dt,tf)
plot(X[3,:])
plot!(X[6,:])
println(norm(X[:,end]-xf))

plot(1:2*timesteps,[X[3,:] X[6,:]],linewidth=4,xtickfontsize=38,ytickfontsize=38,legendfontsize=38, label = ["joint 1 angle" "joint 2 angle"],xlabel="time steps",xguidefontsize=38,ylabel="angle (rad)",yguidefontsize=38)