import torch
import triton
import triton.language as tl
import math


BLOCK_SIZE = 128


@triton.jit
def smreg_feature_map_kernel(
    x,
    features,
    result,
    N: tl.constexpr,
    D: tl.constexpr,
    F: tl.constexpr,
):
    row_idx = tl.program_id(0)
    # Calculate the corresponding row index in x
    offset = tl.zeros((F,), dtype=tl.float32)
    idx = row_idx * BLOCK_SIZE
    if idx < N:
        # Matrix-vector multiplication logic
        # Step 1: Load Features and Compute x_scaled
        x_scaled = tl.zeros((F,), dtype=tl.float32)
        for col in range(D):
            a_values = tl.load(features + col * F, F)
            x_value_ptr = x + idx * D + col
            x_value = tl.load(x_value_ptr, 1)  # Load a single value from x
            product = a_values * x_value
            x_scaled += product

        # Step 2: Compute Exponential Values
        exp_val = tl.exp(x_scaled + offset)
        neg_exp_val = tl.exp(-x_scaled + offset)

        # Step 3: Store Results
        result_ptr = result + idx * 2 * F
        for i in range(F):
            tl.store(result_ptr + i, exp_val[i])
            tl.store(result_ptr + F + i, neg_exp_val[i])


@triton.jit
def causal_attention_kernel(k_prime, q_prime, v, att_raw, att_norm, N, E):
    # Initialize tensors for Gps and Grenorm
    Gps = tl.zeros((N, N, E))
    Grenorm = tl.zeros((N, N, E))

    # Compute Gps and Grenorm
    for i in range(N):
        for j in range(i + 1):
            Gps[i, j, :] = k_prime[i, :] * v[j, :]
            Grenorm[i, j, :] = k_prime[i, :] * 1.0  # assuming v has ones

    # Compute att_raw and att_norm using einsum and cumulative sum
    att_raw = tl.einsum("bce,bcf->bcf", Gps, q_prime).cumsum(2)
    att_norm = tl.einsum("bce,bcf->bcf", Grenorm, q_prime).cumsum(2)


@triton.jit
def feature_mapping_kernel(
    x, features, result, N: tl.constexpr, D: tl.constexpr, F: tl.constexpr
):
    pid = tl.program_id(0)
    np = tl.num_programs(0) * BLOCK_SIZE
    for i in range(0, N, np):
        idx = pid * BLOCK_SIZE + i
        if idx < N:
            # Load the row of x into a local variable
            x_row = tl.load(x + idx * D, D)
            # Apply the feature mapping
            x_transformed = 0.0
            for col in range(D):
                a_values = tl.load(features + col * F, F)
                x_value = x_row[col]
                x_transformed += a_values * x_value
            # Store the result
            tl.store(result + idx * F, x_transformed)


def _get_random_ortho_matrix(blocks, dim, device):
    H = torch.randn((blocks, dim, dim), device=device, requires_grad=False)
    Q, R = torch.linalg.qr(H)
    Q = torch.diag_embed(torch.sign(torch.diagonal(R, dim1=1, dim2=2))) @ Q
    return Q


def favor_attention_forward(q, k, v):
    N, D = q.shape
    F = 128  # Feature dimension, set according to your needs
    dim_features = F
    BLOCK_SIZE = triton.next_power_of_2(D)

    # Define a function to get random orthogonal matrices
    def _get_random_ortho_matrix(blocks, dim, device):
        H = torch.randn((blocks, dim, dim), device=device, requires_grad=False)
        Q, R = torch.linalg.qr(H)
        Q = torch.diag_embed(torch.sign(torch.diagonal(R, dim1=1, dim2=2))) @ Q
        return Q

    # Get per block random unitary matrices and flatten them
    blocks = math.ceil(D / dim_features)
    features = _get_random_ortho_matrix(blocks, dim_features, device=q.device).view(
        -1, dim_features
    )

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Offset, set according to the SMReg formulation
    # x_scaled = torch.empty_like(x)
    # Apply SMReg feature map
    q_prime = torch.empty_like(q)
    k_prime = torch.empty_like(k)
    smreg_feature_map_kernel[N, BLOCK_SIZE](q, features, q_prime, N, D, F)
    smreg_feature_map_kernel[N, BLOCK_SIZE](k, features, k_prime, N, D, F)

    # Compute causal attention
    att_raw = torch.empty_like(q)
    att_norm = torch.empty_like(q)
    causal_attention_kernel[N, BLOCK_SIZE](k_prime, q_prime, v, att_raw, att_norm, N, D)

    # Normalize
    att = att_raw / att_norm

    return att


# Parameters
N = 1823
D = 781
F = 128
BLOCK_SIZE = 128

# Device
device = torch.device("cuda")

# Create random input for x
x = torch.randn(N, D, device=device)

# Create random orthogonal features
blocks = math.ceil(D / F)
features = _get_random_ortho_matrix(blocks, F, device=device)

# Flatten the features
features = features.view(-1, F)

# Allocate output buffer
result = torch.empty(N, F, device=device)

# Invoke the kernel
feature_mapping_kernel[N, BLOCK_SIZE](x, features, result, N, D, F)

# Optionally: Verify results with a reference implementation
# ... Implement a reference version using standard PyTorch operations ...
# assert torch.allclose(result, reference_result, atol=1e-5)

# Print result
print(result)


# # Create random test data for queries, keys, and values
# torch.manual_seed(0)
# x = torch.randn(1823, 781, device="cuda")
# N = x.shape[0]
# D = x.shape[1]
# E = 512
# # Example dimensions
# q = torch.randn(N, D, device="cuda")
# k = torch.randn(N, D, device="cuda")
# v = torch.randn(N, E, device="cuda")

# # Run the Triton kernel implementation
# triton_result = favor_attention_forward(q, k, v)
# print(triton_result)
