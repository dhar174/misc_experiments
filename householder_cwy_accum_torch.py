import time
import torch
import triton
import triton.language as tl

# import householder_cwy_accum_np


def householder(v, device=torch.device("cuda")):
    n = v.shape[0]
    e1 = torch.zeros_like(v)
    e1[0] = 1
    v = v - torch.norm(v) * e1
    H = torch.eye(n, device=device) - 2 * torch.outer(v, v) / torch.dot(v, v)
    return H


def householder_reflections(A, device=torch.device("cuda")):
    n = A.shape[0]
    householder_vectors = []
    for i in range(n):
        target_vector = torch.zeros(n, device=device)
        target_vector[i] = 1
        u = A[:, i] - target_vector
        u = u / torch.norm(u)
        H = torch.eye(n, device=device) - 2 * torch.outer(u, u)
        A = H @ A
        householder_vectors.append(u)

        A[:, i] = A[:, i] / torch.norm(A[:, i])
    return A, householder_vectors


def cwy_transform(vectors, device=torch.device("cuda")):
    """Compute the CWY transform for a list of Householder vectors."""
    L = len(vectors)
    n = vectors[0].shape[0]
    U = torch.column_stack(vectors)
    S = 0.5 * torch.eye(L, device=device) + torch.triu(U.T @ U, diagonal=1)
    composition = torch.eye(n, device=device) - U @ torch.inverse(S) @ U.T
    return composition


def generate_random_orthogonal_matrix(n):
    # Start with a random matrix
    A = torch.randn((n, n), device=device)

    # Iterate through the columns
    for i in range(n):
        # Choose a target vector (e.g., a unit vector along the i-th coordinate axis)
        target_vector = torch.zeros(n, device=device)
        target_vector[i] = 1

        # Compute the difference vector and normalize it
        u = A[:, i] - target_vector
        u = u / torch.norm(u)

        # Compute the Householder matrix
        H = torch.eye(n, device=device) - 2 * torch.outer(u, u)

        # Apply the Householder matrix to A
        A = H @ A

    # Normalize each column to ensure orthonormality
    for i in range(n):
        A[:, i] = A[:, i] / torch.norm(A[:, i])

    return A


def test(device=torch.device("cuda")):
    print(f"Using device: {device}")
    # Example vectors defining Householder reflections
    # Number of features and dimensionality
    num_features = 3
    dimension = 3
    A = torch.randn((dimension, num_features), device=device)
    Q, householder_vectors = householder_reflections(A, device=device)

    # # Generate random vectors
    # random_vectors = [
    #     torch.randn(dimension, device=device) for _ in range(num_features)
    # ]
    # normalized_vectors = [v / torch.norm(v) for v in random_vectors]

    # Compute CWY transform
    composition = cwy_transform(householder_vectors, device=device)
    # Check that the resulting matrix is orthogonal (i.e., Q^T Q = I)
    assert torch.allclose(Q.T @ Q, torch.eye(dimension, device=device))

    # Check that the CWY transform represents the composition of the Householder reflections
    assert torch.allclose(composition, Q)
    # Apply Householder reflections to create an orthogonal matrix

    print("Orthogonality test 1 passed!")

    # Check orthogonality
    for i in range(num_features):
        for j in range(i + 1, num_features):
            assert torch.allclose(
                torch.dot(composition[:, i], composition[:, j]),
                torch.tensor(0.0, device=device),
            )

    print("Orthogonality test 2 passed!")

    # Verify against sequential Householder multiplication
    # Compute sequential Householder multiplication
    sequential_composition = torch.eye(dimension, device=device)
    for v in random_vectors:
        H = householder(v)
        sequential_composition = H @ sequential_composition

    # Check if both methods give the same result
    assert torch.allclose(composition, sequential_composition)
    print("CWY transform and sequential Householder calculations match!")

    return composition


# # Implement householder and cwy_transform in Triton
# @triton.jit
# def householder_kernel(v, out):
#     """Create a Householder reflection matrix for vector v."""
#     n = v.shape[0]
#     e1 = tl.zeros_like(v)
#     e1[0] = 1
#     H = tl.eye(n) - 2 * tl.outer(v, v) / tl.dot(v, v)
#     tl.copy(H, out)


# @triton.jit
# def cwy_transform_kernel(vectors, out):
#     """Compute the CWY transform for a list of Householder vectors."""
#     L = vectors.shape[1]
#     n = vectors.shape[0]
#     U = tl.column_stack(vectors)
#     S = 0.5 * tl.eye(L) + tl.triu(U.T @ U, diagonal=1)
#     composition = tl.eye(n) - U @ tl.inv(S) @ U.T
#     tl.copy(composition, out)


# # Example vectors defining Householder reflections
# v1 = torch.tensor([1.0, 0.0, 0.0])
# v2 = torch.tensor([0.0, 1.0, 0.0])

# # Allocate output buffers
# H1 = torch.empty((3, 3))
# H2 = torch.empty((3, 3))
# composition = torch.empty((3, 3))

# BLOCK_SIZE = 1

# # Invoke the kernels
# householder_kernel[
# householder_kernel[1](v2, H2)
# cwy_transform_kernel[1](torch.stack([v1, v2]), composition)

# # Check if both methods give the same result
# assert torch.allclose(composition, H1 @ H2)
# print("Success for Triton")
func1 = test
# func2 = householder_cwy_accum_np.test
# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# x = torch.randn(32, 10)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# x = x.to(device)

# Test func1
torch.cuda.synchronize()  # Synchronize code before starting timer
start0 = time.time()  # Get current time in seconds
out0 = func1("cpu")  # Call func1 with input data
torch.cuda.synchronize()  # Synchronize code before stopping timer
end0 = time.time()  # Get current time in seconds
time0 = end0 - start0  # Compute execution time of func1 in seconds

# Test func1
torch.cuda.synchronize()  # Synchronize code before starting timer
start1 = time.time()  # Get current time in seconds
out1 = func1()  # Call func1 with input data
torch.cuda.synchronize()  # Synchronize code before stopping timer
end1 = time.time()  # Get current time in seconds
time1 = end1 - start1  # Compute execution time of func1 in seconds

# # Test func2
# torch.cuda.synchronize()  # Synchronize code before starting timer
# start2 = time.time()  # Get current time in seconds
# out2 = func2()  # Call func2 with input data
# torch.cuda.synchronize()  # Synchronize code before stopping timer
# end2 = time.time()  # Get current time in seconds
# time2 = end2 - start2  # Compute execution time of func2 in seconds
# out2 = torch.tensor(out2)  # Convert output of func2 to torch.Tensor


# Compare execution times and outputs
print(f"Execution time of Torch CPU: {time0:.6f} seconds")
print(f"Execution time of Torch CUDA: {time1:.6f} seconds")
# print(f"Execution time of NP: {time2:.6f} seconds")
# print(f"Output of func1CPU and func2 are equal: {torch.equal(out0, out2)}")
# out2 = out2.to(out1.device)
# print(f"Output of func1CUDA and func2 are equal: {torch.equal(out1, out2)}")
# out1 = out1.to(out0.device)
print(f"Output of func1CPU and func1CUDA are equal: {torch.equal(out0, out1)}")
