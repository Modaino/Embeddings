
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
