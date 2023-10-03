import numpy as np

from scipy.special import softmax
from scipy.stats import ortho_group


def generate_random_orthogonal_matrix(d, M):
    # Generate an M x d orthogonal matrix
    return ortho_group.rvs(dim=d)[:M]


def generate_OPFR_params(x, y, d):
    rho_star = calculate_rho_star(x, y, d)
    A = (1 - 1 / rho_star) / 8
    B = np.sqrt(1 - 4 * A)
    C = -(1 + 1) / 2
    D = np.sqrt(4 / (1 - 4 * A)) ** d
    return A, B, C, D


def f_GE_1(omega, x, A, B, C, D):
    return D * np.exp(
        A * np.linalg.norm(omega) ** 2
        + B * np.dot(omega, x)
        + C * np.linalg.norm(x) ** 2
    )


def f_GE_2(omega, y, A, B, C, D):
    return D * np.exp(
        A * np.linalg.norm(omega) ** 2
        + B * np.dot(omega, y)
        + C * np.linalg.norm(y) ** 2
    )


def calculate_rho_star(x, y, d):
    term1 = (2 * np.linalg.norm(x + y) ** 2 + d) ** 2
    term2 = 8 * d * np.linalg.norm(x + y) ** 2
    term3 = 2 * np.linalg.norm(x + y) ** 2 - d
    rho_star = np.sqrt(term1 + term2) - term3
    rho_star /= 4 * np.linalg.norm(x + y) ** 2
    return rho_star


def batched_self_attention(Q, K, V, omega, A, B, C, D, M):
    N, T, d = Q.shape  # Batch size, sequence length, feature dimension
    OPRFs = np.zeros((N, T, T, M))

    for i in range(N):
        for j in range(T):
            for k in range(T):
                for l in range(M):
                    x = Q[i, j]
                    y = K[i, k]
                    OPRFs[i, j, k, l] = f_GE_1(omega[l], x, A, B, C, D) * f_GE_2(
                        omega[l], y, A, B, C, D
                    )

    attention_weights = np.sum(OPRFs, axis=-1)
    attention_output = np.matmul(attention_weights, V)
    return attention_output


# Initialize some parameters
d = 64  # Feature dimension
M = 8  # Number of random projections
N = 32  # Batch size
T = 10  # Sequence length

# Initialize Q, K, V matrices
Q = np.random.rand(N, T, d)
K = np.random.rand(N, T, d)
V = np.random.rand(N, T, d)

# Initialize omega
omega = np.random.randn(M, d)

# Calculate A, B, C, D using a sample x and y
x_sample = Q[0, 0]
y_sample = K[0, 0]

# Calculate rho* using x and y
rho_star = calculate_rho_star(x_sample, y_sample, d)

# Calculate A using rho*
A, B, C, D = generate_OPFR_params(x_sample, y_sample, d)

# Compute the self-attention output
attention_output = batched_self_attention(Q, K, V, omega, A, B, C, D, M)

print("Attention Output Shape:", attention_output.shape)


# Different values for M
M_values = [8, 128, 256]

# 1. Check the mean and variance of the generated OPRFs
for M in M_values:
    omega = np.random.randn(M, d)
    OPRFs_sample = np.array(
        [
            f_GE_1(omega[i, :], x_sample, A, B, C, D)
            * f_GE_2(omega[i, :], y_sample, A, B, C, D)
            for i in range(M)
        ]
    )
    mean_OPRFs = np.mean(OPRFs_sample)
    var_OPRFs = np.var(OPRFs_sample)
    print(f"For M = {M}, Mean of OPRFs: {mean_OPRFs}, Variance of OPRFs: {var_OPRFs}")

# 2. Run some sample queries and keys through the attention mechanism
M = 8  # Use the smallest M for quick testing
omega = np.random.randn(M, d)
attention_output = batched_self_attention(Q, K, V, omega, A, B, C, D, M)
print("\nSample Attention Output:", attention_output[0, 0, :])

# 3. Benchmark the computational time for different values of M
for M in M_values:
    omega = np.random.randn(M, d)
    start_time = time.time()
    attention_output = batched_self_attention(Q, K, V, omega, A, B, C, D, M)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nFor M = {M}, Time taken: {elapsed_time:.4f} seconds")
