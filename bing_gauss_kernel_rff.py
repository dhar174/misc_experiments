# Import libraries
import torch
import math


# Define Gaussian kernel function
def gaussian_kernel(
    q: torch.Tensor, k: torch.Tensor, gamma: float = 1.0
) -> torch.Tensor:
    """
    Args:
        q: query matrix of shape (batch_size, num_heads, seq_len, dim_head)
        k: key matrix of shape (batch_size, num_heads, seq_len, dim_head)
        gamma: width parameter for Gaussian kernel

    Returns:
        w: attention weights matrix of shape (batch_size, num_heads, seq_len, seq_len)
    """
    # Compute squared Euclidean distance between query and key matrices
    qk = torch.cdist(q, k) ** 2

    # Apply Gaussian kernel function
    w = torch.exp(-gamma * qk)

    # Normalize attention weights by row sum
    w = w / torch.sum(w, dim=-1, keepdim=True)

    return w


# Define random Fourier features for Gaussian kernel
def rff_gaussian(
    q: torch.Tensor, k: torch.Tensor, gamma: float = 1.0, m: int = 256
) -> torch.Tensor:
    """
    Args:
        q: query matrix of shape (batch_size, num_heads, seq_len, dim_head)
        k: key matrix of shape (batch_size, num_heads, seq_len, dim_head)
        gamma: width parameter for Gaussian kernel
        m: number of random Fourier features

    Returns:
        w: attention weights matrix of shape (batch_size, num_heads, seq_len, seq_len)
    """
    # Get batch size, number of heads, sequence length and head dimension
    bsz, n_heads, seq_len, dim_head = q.shape

    # Generate random vectors for random Fourier features
    omega = torch.randn((bsz * n_heads * dim_head), device=q.device) * math.sqrt(
        2 * gamma
    )

    # Reshape query and key matrices
    q = q.view(bsz * n_heads * seq_len, dim_head)
    k = k.view(bsz * n_heads * seq_len, dim_head)

    # Compute random Fourier features for query and key matrices
    q_rff = torch.einsum("bd,d->bd", q, omega).view(bsz * n_heads * m)
    k_rff = torch.einsum("bd,d->bd", k, omega).view(bsz * n_heads * m)

    # Compute cosine function for query and key matrices
    q_cos = torch.cos(q_rff)
    k_cos = torch.cos(k_rff)

    # Compute dot product of query and key matrices
    qk = torch.einsum("bld,bhd->blh", q_cos / math.sqrt(m), k_cos / math.sqrt(m))

    # Normalize attention weights by row sum
    w = qk / torch.sum(qk, dim=-1, keepdim=True)

    # Reshape attention weights matrix
    w = w.view(bsz * n_heads * seq_len * seq_len)

    return w


# Define input matrices Q and K
Q = torch.randn(2 * 4 * 8 * 16)  # batch_size=2 * num_heads=4 * seq_len=8 * dim_head=16
K = torch.randn(2 * 4 * 8 * 16)

# Define output matrices W
W_gaussian = torch.empty(
    2 * 4 * 8 * 8
)  # batch_size=2 * num_heads=4 * seq_len=8 * seq_len=8
W_rff = torch.empty(2 * 4 * 8 * 8)

# Compute attention weights using Gaussian kernel
W_gaussian = gaussian_kernel(Q, K)

# Compute attention weights using random Fourier features
W_rff = rff_gaussian(Q, K)

# Compare the results
print("Gaussian kernel:", W_gaussian)
print("Random Fourier features:", W_rff)
print("Relative error:", torch.norm(W_rff - W_gaussian) / torch.norm(W_gaussian))
