# Import Triton
import math
import triton
import torch
import triton.testing as tt


@triton.jit
def smreg_plus_kernel(
    W,
    Q,
    K,
    omega,
    features,
    q_stride,
    k_stride,
    w_stride,
    new_q,
    new_k,
    BLOCK_SIZE: triton.language.constexpr,
):
    row = triton.language.program_id(0)
    col = triton.language.program_id(1)
    row_start_ptr_q = Q + row * q_stride
    col_start_ptr_k = K + col * k_stride
    col_offsets = triton.language.arange(0, BLOCK_SIZE)
    input_ptrs_q = row_start_ptr_q + col_offsets
    input_ptrs_k = col_start_ptr_k + col_offsets

    # Load rows into SRAM, using a mask since BLOCK_SIZE may be greater than dim
    q_row = triton.language.load(input_ptrs_q, mask=col_offsets < Q.shape[-1])
    k_col = triton.language.load(input_ptrs_k, mask=col_offsets < K.shape[-1])

    qk = q_row * k_col

    # Calculate omega pointers for each row and column
    omega_row_start_ptr = omega + row * w_stride
    omega_col_start_ptr = omega + col * w_stride
    omega_row_ptr = omega_row_start_ptr + col_offsets
    omega_col_ptr = omega_col_start_ptr + col_offsets

    # Load omega rows and columns into SRAM
    omega_row = triton.language.load(omega_row_ptr, mask=col_offsets < Q.shape[-1])
    omega_col = triton.language.load(omega_col_ptr, mask=col_offsets < K.shape[-1])

    # Compute the cosine term
    cos = triton.language.cos(qk - omega_row - omega_col)

    # Load features into SRAM
    features_row_start_ptr = features + row * w_stride
    features_col_start_ptr = features + col * w_stride
    features_row_ptr = features_row_start_ptr + col_offsets
    features_col_ptr = features_col_start_ptr + col_offsets

    features_row = triton.language.load(
        features_row_ptr, mask=col_offsets < Q.shape[-1]
    )
    features_col = triton.language.load(
        features_col_ptr, mask=col_offsets < K.shape[-1]
    )

    # # Compute the exponential term
    # exp = triton.language.exp(features_row * cos * features_col)
    # # triton.language.device_print("exp: %f\n", exp)

    # # Synchronize threads within a warp
    # # Compute the sum of the exponential terms along the last dimension
    # exp_sum = triton.language.sum(exp, axis=2)
    # Normalize the output matrix by dividing by the sum along the last dimension
    # triton.language.device_print("exp_sum: %f\n", exp_sum)
    # triton.language.store(W, exp_sum)
    # Write back output to DRAM
    # output_ptr = W + row * w_stride + col * w_stride
    # new_q_ptr = new_q + row * q_stride + col * q_stride
    # new_k_ptr = new_k + row * k_stride + col * k_stride
    # triton.store(output_ptr, W[row, col])
    # # Also store new_q and new_k
    # triton.store(new_q_ptr, q_row)
    # triton.store(new_k_ptr, k_col)


@triton.jit
def einsum_kernel(Gps, q_prime, output, Gps_shape, q_prime_shape, output_shape, **META):
    # get thread index for each dimension
    pid_b = triton.language.ProgramID(0)
    pid_c = triton.language.ProgramID(1)
    pid_e = triton.language.ProgramID(2)

    # check if thread index is within bounds
    if pid_b >= output_shape[0] or pid_c >= output_shape[1] or pid_e >= output_shape[2]:
        return

    # initialize accumulator
    acc = 0.0

    # loop over dimension f
    for f in range(Gps_shape[3]):
        # get a block of Gps[b,c,:,e] and q_prime[b,c,f]
        Gps_block = triton.language.Block(
            Gps
            + pid_b * Gps_shape[1] * Gps_shape[2] * Gps_shape[3]
            + pid_c * Gps_shape[2] * Gps_shape[3]
            + f * Gps_shape[3]
            + pid_e,
            offsets=[0],
            shape=[Gps_shape[2]],
            strides=[Gps_shape[3]],
        )
        q_prime_block = triton.language.Block(
            q_prime
            + pid_b * q_prime_shape[1] * q_prime_shape[2]
            + pid_c * q_prime_shape[2]
            + f,
            offsets=[0],
            shape=[1],
            strides=[1],
        )
        # compute dot product of blocks
        dp = triton.language.dot(Gps_block, q_prime_block)
        # accumulate result
        acc += dp

    # write result to output[b,c,e]
    output_ptr = (
        output
        + pid_b * output_shape[1] * output_shape[2]
        + pid_c * output_shape[2]
        + pid_e
    )
    triton.language.atomic.add(output_ptr, acc)
    triton.language.store(output_ptr, acc)


def smreg_plus(Q, K, omega, v, dim_features=None, dim_head=None):
    if dim_features is None:
        assert dim_head is not None, "dim_features or dim_head needs to be passed"
        dim_features = math.ceil(dim_head * (1 + torch.log2(dim_head)))
        dim_features = 2 * (dim_features // 2)  # needs to be even for some variants

    else:
        dim_features = dim_features

    n_rows, dim = Q.shape
    n_cols = K.shape[1]
    blocks = math.ceil(dim / dim_features)
    # The block size is the smallest power of two greater than the number of columns
    BLOCK_SIZE = triton.next_power_of_2(dim)
    print(BLOCK_SIZE)
    # Allocate output
    print(n_rows)
    print(n_cols)
    print(dim)
    print(dim_features)
    h = torch.randn((blocks, dim, dim), device="cuda", requires_grad=False)
    print(h.shape)
    # #  # Import torch library
    # import torch

    # # Define some parameters
    # dim_features = 512 # Number of random features
    # dim = 781 # Dimension of input vectors
    # n_rows = 1823 # Number of query vectors
    # n_cols = 1823 # Number of key vectors

    # # Define some random input tensors
    # Q = torch.randn(n_rows, dim, device="cuda") # Query tensor
    # K = torch.randn(dim, n_cols, device="cuda") # Key tensor
    # v = torch.randn(dim, device="cuda") # Value vector
    # h = torch.tensor(0.5, device="cuda") # Kernel bandwidth parameter

    # # Method 1: Random Fourier features sampled from a Gaussian distribution

    # # 0 Compute the inverse kernel bandwidth using Silverman's rule of thumb
    # sigma = (4 / (dim + 2)) ** (1 / (dim + 4)) * n_rows ** (-1 / (dim + 4))

    # # 1 Sample W from a Gaussian distribution with mean zero and covariance matrix proportional to sigma^2
    # W = torch.real(torch.normal(0, sigma, size=(dim_features, dim), device="cuda", requires_grad=False)) / math.sqrt(2)

    # # 2 Project Q and K using W
    # Q_proj = Q @ W.T # Shape: (n_rows, dim_features)
    # K_proj = K.T @ W.T # Shape: (n_cols, dim_features)

    # # 3 Compute the approximate attention matrix using softmax
    # A_approx_1 = torch.softmax(Q_proj @ K_proj.T / h, dim=-1) # Shape: (n_rows, n_cols)

    # # 4 Compute the approximate output vector using v
    # y_approx_1 = A_approx_1 @ v # Shape: (n_rows)

    # # Method 2: Random orthogonal matrix obtained by QR decomposition

    # # 1 Sample a random Gaussian matrix H
    # H = torch.randn((dim, dim), device="cuda", requires_grad=False)

    # # 1.5 Apply QR decomposition to H and obtain an orthogonal matrix Q_orth
    # Q_orth, R = torch.linalg.qr(H) # Shape: (dim, dim)

    # # 2 Project Q and K using Q_orth
    # Q_proj_orth = Q @ Q_orth.T # Shape: (n_rows, dim)
    # K_proj_orth = K.T @ Q_orth.T # Shape: (n_cols, dim)

    # # 3 Compute the approximate attention matrix using softmax
    # A_approx_2 = torch.softmax(Q_proj_orth @ K_proj_orth.T / h, dim=-1) # Shape: (n_rows, n_cols)

    # # 4 Compute the approximate output vector using v
    # y_approx_2 = A_approx_2 @ v # Shape: (n_rows)

    # method 1
    sigma = (4 / (dim + 2)) ** (1 / (dim + 4)) * n_rows ** (-1 / (dim + 4))
    W = torch.real(
        torch.normal(
            0, sigma, size=(dim_features, dim), device="cuda", requires_grad=False
        )
    ) / math.sqrt(2)

    print(W.shape)
    # method 2
    features, R = torch.linalg.qr(h)
    print(features.shape)
    features = (
        torch.diag_embed(torch.sign(torch.diagonal(R, dim1=1, dim2=2))) @ features
    )
    print(features.shape)
    new_q = torch.empty_like(Q)
    new_k = torch.empty_like(K)
    print(new_q.shape)
    print(new_k.shape)
    print(W.shape)
    print(Q.shape)
    print(K.shape)
    print(omega.shape)

    # Project Q and K using W
    Q_proj = Q @ W.T  # Shape: (n_rows, dim_features)
    K_proj = K.T @ W.T  # Shape: (n_cols, dim_features)

    # Enqueue kernel
    smreg_plus_kernel[(n_rows, n_cols)](
        W,
        Q_proj,
        K_proj,
        omega,
        features,
        Q.stride(0),
        K.stride(1),
        W.stride(0),
        new_q,
        new_k,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # # einsum kernel
    # ref_v = torch.ones_like(v.unsqueeze(2))  # BATCH x SEQ x 1 x EMB
    # Gps = new_k.unsqueeze(3) * v.unsqueeze(2)
    # Grenorm = new_k.unsqueeze(3) * ref_v

    # output = torch.empty((n_rows, n_cols, 1), device="cuda")
    # einsum_kernel[(n_rows, n_cols, 1)](
    #     Gps,
    #     new_q,
    #     output,
    #     Gps.shape,
    #     new_q.shape,
    #     output.shape,
    # )

    # Normalize the output matrix by dividing by the sum along the last dimension

    W @ features
    W = W / W.sum(dim=-1, keepdim=True)

    return W


# Define a reference implementation using PyTorch
def smreg_plus_torch(Q, K, h, omega):
    # Compute the dot product of Q and K
    qk = torch.einsum("bld,bmd->blm", Q, K)
    # Compute the cosine term
    cos = torch.cos(qk - omega.unsqueeze(1) - omega.unsqueeze(2))
    # Compute the exponential term
    exp = torch.exp(h * cos)
    # Normalize the output matrix by dividing by the sum along the last dimension
    W = exp / exp.sum(dim=-1, keepdim=True)
    return W


# @triton.jit
# def qr_decomposition(
#     A,
#     Q,
#     R,
#     n: triton.language.constexpr,
#     M: triton.language.constexpr,
#     N: triton.language.constexpr,
# ):
#     # Transpose A
#     A_ = triton.language.zeros((n, n), dtype=triton.language.float32)
#     for i in range(n):
#         for j in range(n):
#             A_[i, j] = A[j, i]

#     # Initialize Q and R
#     for i in range(n):
#         for j in range(n):
#             Q[i, j] = 0.0
#             R[i, j] = 0.0

#     # Begin the Gram-Schmidt process
#     for i in range(n):
#         v = A_[i]
#         for j in range(i):
#             # Compute dot product
#             R[j, i] = triton.language.sum(Q[j, k] * A_[i, k] for k in range(n))
#             # Compute vector subtraction and scalar multiplication
#             v = v - Q[j] * R[j, i]

#         # Compute vector norm
#         R[i, i] = triton.language.sqrt(triton.language.sum(vi**2 for vi in v))
#         # Scalar multiplication
#         Q[i] = v / R[i, i]

#     # Transpose Q
#     Q1 = triton.language.zeros((n, n), dtype=triton.language.float32)
#     for i in range(n):
#         for j in range(n):
#             Q1[i, j] = Q[j, i]

#     return Q1, R


# # Define the input matrix A (for example, a 3x3 matrix)
# A = torch.tensor(
#     [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
#     device="cuda",
#     dtype=torch.float32,
# )

# # Get the size of the matrix
# n = A.size(0)

# # Allocate space for the output matrices Q and R
# Q = torch.zeros((n, n), device="cuda", dtype=torch.float32)
# R = torch.zeros((n, n), device="cuda", dtype=torch.float32)

# # Define the grid size for the Triton kernel
# M = 128
# N = 128

# # Call the Triton kernel
# qr_decomposition[(M, N), (n, n)](A, Q, R, n, M, N)

# The matrices Q and R now contain the QR decomposition of A
# print("Q:", Q.cpu())
# print("R:", R.cpu())


torch.manual_seed(0)
Q = torch.randn(1823, 781, device="cuda")
K = torch.randn(781, 1823, device="cuda")
v = torch.randn(781, device="cuda")
h = torch.tensor(0.5, device="cuda")
omega = torch.randn(1823, device="cuda")

W_triton = smreg_plus(Q, K, omega, v, dim_features=512)

# You can then compare W_triton with expected output using any desired method


# Create some random input tensors
B = 16  # batch size
L = 32  # sequence length
D = 64  # hidden dimension
Q = torch.randn(B, L, D).cuda()
K = torch.randn(B, L, D).cuda()
omega = torch.randn(B, L).cuda()
# Create an output tensor for the Triton kernel
W = torch.empty(B, L, L).cuda()

# Define some launch parameters for the Triton kernel
grid = lambda meta: (
    triton.cdiv(meta.size(0), meta.block[0]),
    triton.cdiv(meta.size(1), meta.block[1]),
)
block = (32, 32)

# Call the Triton kernel with the input tensors and launch parameters
smreg_plus_kernelgrid = smreg_plus_kernel[grid, block](Q, K, W, h, omega)

# Call the reference implementation with the same input tensors
W_ref = smreg_plus_torch(Q, K, h, omega)

# Check the correctness and performance of the Triton kernel using triton.testing module
tt.assert_almost_equal(W_ref, W)  # check that the output tensors are almost equal
benchmark = tt.Benchmark(W.numel(), warmup=10, rep=100)  # create a benchmark object
ms_torch = benchmark(
    lambda: smreg_plus_torch(Q, K, h, omega)
)  # measure the execution time of the reference implementation in milliseconds
ms_triton = benchmark(
    lambda: smreg_plus_kernelgrid
)  # measure the execution time of the Triton kernel in milliseconds
ms_t2 = benchmark(lambda: smreg_plus)
print(f"PyTorch: {ms_torch} ms")
print(f"Triton: {ms_triton} ms")
print(f"Triton2: {ms_t2} ms")


# device = torch.device("cuda")

# # Define input matrices Q and K
# Q = torch.randn(64, 16, device=device)
# # batch_size=64, dim=16
# K = torch.randn(64, 16, device=device)
# print(Q.shape)
# # Define output matrix W
# W = torch.empty(64, 64, device=device)
# # batch_size=64, seq_len=64

# # Define scaling factor h
# h = 1.0

# # Define random vector omega
# omega = torch.randn(64, device=device)

# # Launch SMREG+ kernel on a given stream
# stream = torch.cuda.current_stream()
# grid_size = lambda opt: (64, 64)  # grid size is equal to output matrix shape
# block_size = lambda opt: (1,)  # block size is one thread per row/column pair
# smreg_plus_kernel[grid_size, block_size](
#     Q, K, W, h, omega, num_warps=4, stream=stream, device=device
# )
