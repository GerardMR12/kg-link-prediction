import torch

# Input vector
vector = [8, 2, 0, 5]

# Determine the size of the matrix (rows = len(vector), columns = max value + 1)
rows = len(vector)
cols = max(vector) + 1
matrix = torch.zeros((rows, cols), dtype=int)

# Fill the matrix with 1s at the specified positions
for i, pos in enumerate(vector):
    matrix[i, pos] = 1

print(matrix)