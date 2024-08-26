import sys

def generate_cnf(N, filename):
    # Open file for writing
    with open(filename, 'w') as f:
        # Calculate the number of variables and clauses
        num_vars = N * N
        num_clauses = (N * (N - 1)) + N * (N - 1)  # Clauses for no two '1's in a row and at least one '1' in a row
        
        # Write the problem line
        f.write(f"p cnf {num_vars} {num_clauses}\n")

        # Clauses to ensure at least one '1' in each row
        for i in range(N):
            row_clause = [i * N + j + 1 for j in range(N)]
            f.write(" ".join(map(str, row_clause)) + " 0\n")

        # Clauses to ensure no two '1's in the same row
        for i in range(N):
            for j in range(N):
                for k in range(j + 1, N):
                    clause = [- (i * N + j + 1), - (i * N + k + 1)]
                    f.write(" ".join(map(str, clause)) + " 0\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <N> <filename>")
        sys.exit(1)

    N = int(sys.argv[1])
    filename = sys.argv[2]
    generate_cnf(N, filename)
