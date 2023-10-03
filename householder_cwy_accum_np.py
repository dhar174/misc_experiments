import time
import numpy as np
import numba
from numba import cuda


def householder(v, device="cpu"):
    """Create a Householder reflection matrix for vector v."""
    n = v.shape[0]
    e1 = np.zeros_like(v)
    e1[0] = 1
    H = np.eye(n) - 2 * np.outer(v, v) / np.dot(v, v)
    return H


@cuda.jit
def cwy_transform(vectors, device="cpu"):
    """Compute the CWY transform for a list of Householder vectors."""
    i = cuda.grid(1)  # Get the index of the current thread on the GPU
    if i < vectors.size:
        L = len(vectors)
        n = vectors[0].shape[0]
        U = np.column_stack(vectors)
        S = 0.5 * np.eye(L) + np.triu(U.T @ U, k=1)
        composition = np.eye(n) - U @ np.linalg.inv(S) @ U.T
    return composition


def cwy_transform(vectors, device="cpu"):
    """Compute the CWY transform for a list of Householder vectors."""
    L = len(vectors)
    n = vectors[0].shape[0]
    U = np.column_stack(vectors)
    S = 0.5 * np.eye(L) + np.triu(U.T @ U, k=1)
    composition = np.eye(n) - U @ np.linalg.inv(S) @ U.T
    return composition


def test(device="cpu"):
    print(f"Using device: {device}")
    # Example vectors defining Householder reflections
    # Example vectors defining Householder reflections
    # Number of features and dimensionality
    num_features = 3
    dimension = 3

    # Generate random vectors
    random_vectors = [np.random.randn(dimension) for _ in range(num_features)]
    if device != "cpu":
        # Transfer the arrays to the GPU device
        x_device = cuda.to_device(random_vectors)
        threads_per_block = 32
        blocks_per_grid = (
            random_vectors.size + (threads_per_block - 1)
        ) // threads_per_block
        cwy_transform[blocks_per_grid, threads_per_block](x_device)
    # Compute CWY transform
    composition = cwy_transform(random_vectors, device=device)

    # Check orthogonality
    for i in range(num_features):
        for j in range(i + 1, num_features):
            assert np.allclose(
                np.dot(composition[:, i], composition[:, j]),
                np.array(0.0),
            )

    print("Orthogonality test passed!")

    # Verify against sequential Householder multiplication
    # Compute sequential Householder multiplication
    sequential_composition = np.eye(dimension, device=device)
    for v in random_vectors:
        H = householder(v)
        sequential_composition = H @ sequential_composition

    # Check if both methods give the same result
    assert np.allclose(composition, sequential_composition)
    print("CWY transform and sequential Householder calculations match!")
    return composition


start0 = time.time()  # Get current time in seconds
out0 = test("cpu")  # Call func1 with input data
end0 = time.time()  # Get current time in seconds
time0 = end0 - start0  # Compute execution time of func1 in seconds

# Test func1
start1 = time.time()  # Get current time in seconds
out1 = test()  # Call func1 with input data
end1 = time.time()  # Get current time in seconds
time1 = end1 - start1

print(f"Execution time of NP GPU: {time0:.6f} seconds")
print(f"Execution time of NP CPU: {time1:.6f} seconds")
out1 = out1.to(out0.device)
print(f"Output of func1CPU and func2 are equal: {np.equal(out0, out1)}")
