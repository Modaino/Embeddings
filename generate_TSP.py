import numpy as np
import sys

def generate_tsp_distance_matrix(num_cities, symmetric=True):
    """
    Generates a distance matrix for the TSP.
    
    Args:
        num_cities (int): Number of cities (nodes).
        symmetric (bool): If True, generates a symmetric matrix. If False, generates a non-symmetric matrix.
        
    Returns:
        np.ndarray: Distance matrix.
    """
    # Generate a random distance matrix
    distance_matrix = np.random.randint(1, 100, size=(num_cities, num_cities))
    
    if symmetric:
        # Make the matrix symmetric
        i_lower = np.tril_indices(num_cities, -1)
        distance_matrix[i_lower] = distance_matrix.T[i_lower]
    
    # Set the diagonal to zero (no self-loops)
    np.fill_diagonal(distance_matrix, 0)
    
    return distance_matrix

def write_distance_matrix_to_file(filename, distance_matrix):
    """
    Writes the distance matrix to a file.
    
    Args:
        filename (str): Name of the file to write the matrix to.
        distance_matrix (np.ndarray): The distance matrix to write.
    """
    np.savetxt(filename, distance_matrix, fmt='%d')

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_tsp.py <num_cities> <symmetric> <filename>")
        sys.exit(1)
    
    num_cities = int(sys.argv[1])
    symmetric = int(sys.argv[2])
    filename = sys.argv[3]
    
    matrix = generate_tsp_distance_matrix(num_cities, symmetric)
    write_distance_matrix_to_file(filename, matrix)
    print(f"Distance matrix saved to {filename}\n{matrix}")
