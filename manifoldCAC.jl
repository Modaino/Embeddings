
# Right hand side of the dif
function basic_rule3!(du, u, params, t)
    D = params[1]
    T = params[2]
    steps = params[3]
    beta = params[4]
    lambda = params[5]
    gamma = params[6]
    b = params[7]
    
    N = size(D, 1)  # Dimension of the matrix

    a = 1.0
    #a = t./steps+1.1
    
    # Reshape the vector u to get the matrix P
    x = u[1:N^2]
    e = u[(N^2+1):(2*N^2)]
    f = u[(2*N^2+1):end]
    X = reshape(x, N, N)  #orthogonal -b, 1-b
    E = reshape(e, N, N)
    F = repeat(f, 1, N)  # Repeat along the rows

    P = X .+ b # 0,1 variable
    Y = 2.0 .* P .- 1 # -1, 1 variable
    
    # Euclidian gradient (descent)
    grad = 2.0 .* T * X * D + (2*b) .* T * D .* F
    dx_dt = - E.* grad * beta
    dx_dt_M = X * (X' * dx_dt - dx_dt' * X)

    de_dt = - lambda .* (Y.^2 .- a) .* E
    df_dt = gamma .* (sum(P,dims=2) .-1)

    du[1:N^2] = vec(dx_dt_M)  # Flatten the matrix into a vector
    du[(N^2+1):(2*N^2)] = vec(de_dt)
    du[(2*N^2+1):end] = df_dt
  
end

function run_ODE(N,params,tspan,M₀,cost_fnc_X,H0)

    e = ones(N,N)
    f = ones(N)
    P0 = rand(M₀) # This is P(t=0)
    u0 = [vec(P0);vec(e);f]
    
    # 3) run
    
    prob = ODEProblem(basic_rule3!, u0, tspan, params)
    #sol = solve(prob, Vern9(), reltol=1e-9, abstol=1e-12);
    tCPU = @elapsed sol = solve(prob, Vern9(), reltol=1e-6, abstol=1e-6);
    
    # 4) collect results
    H = [ cost_fnc_X(reshape(x_t[1:N^2], N, N), false) for x_t in sol.u]
    H_perm = [ cost_fnc_X(reshape(x_t[1:N^2], N, N), true) for x_t in sol.u]
    
    iopt = findfirst(x -> x == H0, H_perm)
    
    if isnothing(iopt)
        isolved = false
        t0 = steps
    else
        isolved = true
        t0 = sol.t[iopt]
    end

    return isolved, t0, tCPU, H, H_perm, sol

end