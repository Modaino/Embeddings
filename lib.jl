using Random
using DelimitedFiles

using Combinatorics  # Import Combinatorics for generating permutations
using LinearAlgebra  # Import LinearAlgebra for identity matrix


## Create random TSP instances

function generate_tsp_distance_matrix(num_cities::Int, symmetric::Bool=true)
    """
    Generates a distance matrix for the TSP.
    
    Args:
        num_cities (Int): Number of cities (nodes).
        symmetric (Bool): If true, generates a symmetric matrix. If false, generates a non-symmetric matrix.
        
    Returns:
        Array{Int, 2}: Distance matrix.
    """
    # Generate a random distance matrix
    distance_matrix = rand(1:99, num_cities, num_cities)

    if symmetric
        # Make the matrix symmetric
        for i in 2:num_cities, j in 1:i-1
            distance_matrix[i, j] = distance_matrix[j, i]
        end
    end
    
    # Set the diagonal to zero (no self-loops)
    for i in 1:num_cities
        distance_matrix[i, i] = 0
    end

    return distance_matrix
end

# create cycle adjacency graph

function cycle_graph_adjacency_matrix(n::Int, directed::Bool=true)
    """
    Generates the adjacency matrix of a cycle graph of length n.

    Args:
        n (Int): The number of nodes in the cycle graph.
        directed (Bool): If true, generates a directed cycle graph.
                         If false, generates an undirected cycle graph.

    Returns:
        Array{Int64, 2}: The adjacency matrix of the cycle graph.
    """
    # Initialize an n x n matrix with zeros
    adj_matrix = zeros(Int, n, n)
    
    if directed
        # Directed cycle graph
        for i in 1:(n-1)
            adj_matrix[i, i+1] = 1
        end
        adj_matrix[n, 1] = 1  # Last node connects to the first node
    else
        # Undirected cycle graph
        for i in 1:(n-1)
            adj_matrix[i, i+1] = 1
            adj_matrix[i+1, i] = 1
        end
        adj_matrix[n, 1] = 1
        adj_matrix[1, n] = 1  # Last node connects to the first node and vice versa
    end
    
    return adj_matrix
end

#find ground-state by brute force

function generate_permutation_matrices(n)
    perms = collect(permutations(1:n))  # Generate all permutations of 1:n
    matrices = []

    for p in perms
        perm_matrix = I(n)[p, :]  # Generate permutation matrix from permutation
        push!(matrices, perm_matrix)  # Store the matrix
    end

    return matrices
end

function is_permutation_matrix(X)
    # Check if the matrix is square
    if size(X, 1) != size(X, 2)
        return false
    end
    
    # Check if the matrix contains only 0s and 1s
    if any(x -> !(x in (0, 1)), X)
        return false
    end
    
    # Check if each row contains exactly one 1
    for row in eachrow(X)
        if sum(row) != 1
            return false
        end
    end
    
    # Check if each column contains exactly one 1
    for col in eachcol(X)
        if sum(col) != 1
            return false
        end
    end
    
    return true
end

function brute_force(N,D)

    directed = false  # Set to false for undirected
    T = cycle_graph_adjacency_matrix(N, directed);
        
    function cost_fnc_P(P)
        return tr(transpose(D)*transpose(P)*T*P)
    end

    # Example usage:
    perm_matrices = generate_permutation_matrices(N)
    
    H = []
    
    for (i, mat) in enumerate(perm_matrices)
    
        cH = cost_fnc_P(Float64.(mat))
            
        if false
            println("Permutation Matrix $i:")
            println(mat)
            println(cH)
            print()
            println()
        end
        push!(H, cH)
    end
    
    H = Float64.(H)
    H0 = minimum(H)
    println("Optimal cost $H0")

    return H, H0

end

function load_TSP_instance(N,i)

    file_path = joinpath(folder, "rTSP_D_$(N)_$(i).dat")
    file_path_H0 = joinpath(folder, "rTSP_H0_$(N)_$(i).dat")
    
    # Read the matrix file and convert it to Int8
    if isfile(file_path)
        matrix = readdlm(file_path)  # Read the matrix
        matrix_int8 = Int8.(matrix)  # Convert to Int8
        #println("Loaded matrix from: $file_path")
        #println(matrix_int8)
    else
        println("File not found: $file_path")
    end
    
    # Read the scalar file
    if isfile(file_path_H0)
        scalar = readdlm(file_path_H0)[1]  # Assuming the scalar is the first value in the file
        #println("Loaded scalar from: $file_path_H0")
        #println(scalar)
    else
        println("File not found: $file_path_H0")
    end

    return matrix, scalar

end

function initialize_solver(N,D)

    T = cycle_graph_adjacency_matrix(N, false);

    b = 2/N
    
    function cost_fnc_P(P)
        return tr(transpose(D)*transpose(P)*T*P)
    end
        
    function cost_fnc_X(X, checkPerm)
        P = (sign.(2.0 .* (X .+ b) .- 1) .+ 1)/2 #but is not necessarily a permutation matrix
        if !checkPerm
            return tr(transpose(D)*transpose(P)*T*P)
        else
            if is_permutation_matrix(P)
                return tr(transpose(D)*transpose(P)*T*P)
            else
                return NaN
            end
        end
    end

    return cost_fnc_P, cost_fnc_X, T, b

end