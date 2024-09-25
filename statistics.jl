using CSV, Distributed, DataFrames
using Manopt, Manifolds, Random, LinearAlgebra, ManifoldDiff
using ManifoldDiff: grad_distance, prox_distance
@everywhere using DifferentialEquations, DiffEqCallbacks
using DelimitedFiles

# Note, run this as 'julia -p {nbr_procs} statistics.jl' for multithreadding, replace {nbr_procs} with the desired number of processes

@everywhere function fGRADCTDS_rule!(du, u, p, t)
    (c, k, β, A, B, C, M₀) = p

    M, N² = size(c)
    N = size(A, 1)

    s = u[1:N²]
    P = 0.5 * (reshape(s, N, N) .+ 1)  # Add 1 directly, instead of creating an entire ones array
    dP_dt = similar(P)
    a = u[N²+1:end]
  
    K = ones(M)  # Pre-allocate K array
    for m in 1:M
        for j in 1:N²
            K[m] *= (1 - c[m, j] * s[j])
        end
    end

    temp_prod = ones(M, N²)  # Pre-allocate temporary array for inner products
    for m in 1:M
        for i in 1:N²
            temp_prod[m, i] = prod([j != i ? (1 - c[m, j] * s[j]) : 1 for j in 1:N²])
        end
    end

    for i in 1:N²
        sum_term = 0.0
        for m in 1:M
            sum_term += 2 * a[m] * c[m, i] * 2.0^(-k[m]) * temp_prod[m, i] * K[m]
        end
        du[i] = sum_term
    end

    for m in 1:M
        du[N² + m] = a[m] * K[m]
    end

    # Matrix multiplication optimization
    dP_dt .= (B * P * A + B' * P * A)
    dP_dt .*= β * exp(-t)

    # Avoid creating the ones(N, N) matrix, and optimize matrix flattening
    du[1:N²] .+= -vec(2 * dP_dt .- 1)

    return nothing
end

@everywhere function GRADCTDS_rule!(du, u, p, t)
    c = p[1]
    k = p[2]
    β = p[3]
    A = p[4]
    B = p[5]
    C = p[6]
    M₀ = p[7]

    M, N² = size(c)
    N = size(A)[1]

    s = u[1:N²]
    #P = reshape(s, N, N)
    P = 0.5*(reshape(s, N, N)+ones(N, N))
    dP_dt = similar(P)
    a = u[N²+1:end]

    #
    K = [prod([ (1-c[m, j]*s[j]) for j in 1:length(s)]) for m in 1:M]

    # Calculating the constraint vector field
    for i in 1:length(s)
        du[i] = sum([2*a[m]*c[m,i]*2.0^(-k[m])*prod([ j!=i ? (1-c[m, j]*s[j]) : 1 for j in 1:length(s)])*K[m] for m in 1:M])
        #du[i] = sum([2*a[m]*c[m,i]*(2.0^(-k[m])*prod([ j!=i ? (1-c[m, j]*s[j]) : 1 for j in 1:length(s)]))^2*(1-c[m, i]*s[i]) for m in 1:M])
    end
    for m in 1:M
        du[N²+m] = a[m]*K[m]
        #du[N²+m] = a[m]*2.0^(-k[m])*prod([ (1-c[m, j]*s[j]) for j in 1:length(s)])
    end
    
    #Time weighted Eucledian gradient
    #dP_dt = β * (B*P*A + B'*P*A) *exp(-t) # + C is not added
    dP_dt .= (B * P * A + B' * P * A) # Avoid C if unnecessary
    #β = 1/maximum(dP_dt)
    dP_dt .*= β * exp(-t)  

    # Adding the Riemannian gradient, by converting the matrix dP_dt to a vector
    du[1:N²] .+= -vec(2*dP_dt - ones(N, N))  # Flatten the matrix into a vector
    #du[1:N²] .+= -vec(dP_dt)  # Flatten the matrix into a vector

    return nothing
end

function load_QAP_data(file_path::String)
    open(file_path, "r") do io
        # Read the integer (first line)
        n = parse(Int, readline(io))
        
        # Skip the empty line
        readline(io)
        
        # Read the first matrix (nxn)
        matrix1 = [parse.(Int, split(strip(readline(io)))) for _ in 1:n] |> x -> hcat(x...)
        
        # Skip the empty line
        readline(io)

        # Read the second matrix (nxn)
        matrix2 = [parse.(Int, split(strip(readline(io)))) for _ in 1:n] |> x -> hcat(x...)
        
        # Check if there is an empty line before the third matrix
        third_matrix = zeros(Int64, n, n)
        next_line = readline(io, keep=false)
        if next_line != ""
            # If the next line is not empty, it's part of the third matrix
            third_matrix = [parse.(Int, split(strip(next_line))) for _ in 1:n] |> x -> hcat(x...)
        end
        
        return (n, matrix1, matrix2, third_matrix)
    end
end

function load_cnf(file_name)
    c = Nothing
    open(file_name) do file
        for (idx, line) in enumerate(eachline(file))
            if idx == 1
                N = parse(Int32, split(line, " ")[3])
                M = parse(Int32, split(line, " ")[4])
                c = zeros(M,N)
            else
                variables = split(line, " ")
                for var_str in variables
                    var = parse(Int32, var_str)
                    if var != 0
                        if var > 0
                            c[idx-1, var] = 1
                        elseif var < 0
                            c[idx-1, -var] = -1
                        end
                    end
                end
            end
        end
    end
    return c
end

@everywhere function satisfied(spin_config, c)
    function check_clause(row, state)
        for (index,elem) in enumerate(row)
            if elem == state[index]
                return true
            end
        end
    end

    incorrect_flag = false
    for clause in eachrow(c)
        if check_clause(clause, spin_config) != true
            incorrect_flag = true
            break
        end
    end

    if incorrect_flag
        return false
    end
    return true
end

# Define your simulate function
@everywhere function simulate(params, initial_condition, tspan)
    prob = ODEProblem(fGRADCTDS_rule!, initial_condition, tspan, params)
    condition(u, t, integrator) = satisfied([s>0 ? 1 : -1 for s in u], params[1])
    affect!(integrator) = terminate!(integrator)
    CTDS_cb = DiscreteCallback(condition, affect!);

    # Measure wallclock time for solving the problem
    elapsed_wtime = @elapsed begin
        sol = solve(prob, Tsit5(), callback = CTDS_cb)  # Choose your solver here
    end

    M, N² = size(params[1])
    aTTS = sol.t[end]
    final_state = sol.u[end][1:N²]
    return (initial_condition[1:N²], final_state, aTTS, elapsed_wtime)
end

@everywhere function ssimulate(params, initial_condition, tspan; dt=1e-3)
    # Extract time span and define number of steps
    t0, tf = tspan
    Nsteps = Int((tf - t0) / dt)
    
    # Initialize state and time
    u = initial_condition
    t = t0
    
    # Prepare to store solution (optional, depends if you need all steps or just the final state)
    # trajectory = Vector{typeof(u)}(undef, Nsteps)
    
    # Euler step loop
    for step in 1:Nsteps
        # Call your right-hand side function `fGRADCTDS_rule!`
        du = similar(u)  # Create a vector to store the derivative
        fGRADCTDS_rule!(du, u, params)  # Compute derivative
        
        # Update the state using Euler's method
        u .= u .+ dt .* du
        
        # Update time
        t += dt
        
        # Optionally store the state
        # trajectory[step] = copy(u)
    end
    
    # Extract final state and time-to-solution (TTS)
    M, N² = size(params[1])
    TTS = t
    
    # Return initial condition, final state, and any other values
    initial_state = initial_condition[1:N²]
    final_state = u[1:N²]  # Final state vector
    return (initial_state, final_state, TTS, 0.1)  # Modify return values if necessary
end

# Simulate multiple cases in parallel
function run_simulations(param_list, init_conditions_list, problem_name_list, tspan)
    results = pmap((p, ic, pname) -> (pname, simulate(p, ic, tspan)...), param_list, init_conditions_list, problem_name_list)
    #results = pmap((p, ic, pname) -> (pname, ssimulate(p, ic, tspan)...), param_list, init_conditions_list, problem_name_list)
    return results
end

function write_results_to_file(file_name, results)
    open(file_name, "w") do io  # Open the file for writing
        for (description, u0, s_final, WCT, ATTS) in results
            println(description)
            # Write the string (first element)
            write(io, description * "\n")
            
            # Write the double vector u0 (second element)
            write(io, "[" * join(u0, ", ") * "]\n")
            
            # Write the integer vector s_final (third element)
            write(io, "[" * join(s_final, ", ") * "]\n")
            
            # Write the two double values WCT and ATTS (fourth and fifth elements)
            write(io, string(WCT) * "\n")
            write(io, string(ATTS) * "\n")
        end
    end
end

function generate_params_and_initial_conditions(indexes, nbr_init)
    param_list = []
    init_conditions_list = []
    problem_name_list = []
    
    for (N, m) in indexes
        # Generate nbr_init different initial conditions
        for i in 1:nbr_init
            # Load data for the specific problem
            N, B, A, C = load_QAP_data("QAPs/randomQAP_N$(N)_$(m).dat")
            c = load_cnf("permutation_cnf/N$(N)s.cnf")
            M, N² = size(c)
            k = map(row -> count(x -> x != 0, row), eachrow(c))
            M₀ = OrthogonalMatrices(N)
            
            # Generate β
            P0 = rand(M₀)  # Random permutation matrix as initial condition
            β = 1 / maximum(B * P0 * A + B' * P0 * A)
            p = (c, 2.0 .^ (-k), β, A, B, C, M₀)
            
            # Store parameters
            push!(param_list, p)
            P0 = rand(M₀)  # Create a new random permutation matrix
            u0 = vcat(vec(P0), ones(M))  # Vectorize and concatenate
            push!(init_conditions_list, u0)
            push!(problem_name_list, "QAPs/randomQAP_N$(N)_$(m).dat")
        end
    end
    
    return param_list, init_conditions_list, problem_name_list
end


# Define your problem indices as (N, m)
indexes = []  # Add more as needed
for N ∈ 6:7
    for m ∈ 1:5
        push!(indexes, (N, m))
    end
end
param_list, init_conditions_list, problem_name_list = generate_params_and_initial_conditions(indexes, 2)
tspan = (0.0,0.35);

# Run simulations in parallel
results = run_simulations(param_list, init_conditions_list, problem_name_list, tspan)


# Write results to file
write_results_to_file("results/test_run3.dat", results)
