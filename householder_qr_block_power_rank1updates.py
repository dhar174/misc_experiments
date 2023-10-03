import numpy as np


# Householder QR Factorization
def householder_qr(A):
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    for j in range(n):
        x = R[j:, j]
        e = np.zeros_like(x)
        e[0] = 1
        u = x - np.linalg.norm(x) * e
        if np.linalg.norm(u) == 0:  # Avoid division by zero
            continue
        v = u / np.linalg.norm(u)
        Q_j = np.eye(m)
        Q_j[j:, j:] -= 2.0 * np.outer(v, v)
        R = Q_j @ R
        Q = Q @ Q_j.T
    return Q, R


# Block Power Method
def block_power_method(A, num_eigenvalues=1, max_iter=100):
    m, n = A.shape
    Q = np.random.rand(m, num_eigenvalues)
    for _ in range(max_iter):
        Q, _ = householder_qr(A @ Q)
    eigenvalues = np.diag(Q.T @ A @ Q)
    eigenvectors = Q
    return eigenvalues, eigenvectors


# Rank-1 Update of QR Factorization
def rank1_update_qr(Q, R, u, v):
    A_updated = Q @ R + np.outer(u, v)
    return householder_qr(A_updated)


# Example usage
A = np.array([[4.0, 2.0], [2.0, 3.0]])
Q, R = householder_qr(A)
eigenvalues, eigenvectors = block_power_method(A)
u = np.array([1.0, 0.0])
v = np.array([0.0, 1.0])
Q_updated, R_updated = rank1_update_qr(Q, R, u, v)

print("Original A:")
print(A)
print("QR Factorization:")
print("Q:")
print(Q)
print("R:")
print(R)
print("Eigenvalues:")
print(eigenvalues)
print("Eigenvectors:")
print(eigenvectors)
print("Updated Q after Rank-1 Update:")
print(Q_updated)
print("Updated R after Rank-1 Update:")
print(R_updated)


def block_power_method_svd(Q, r, max_iter=100):
    L, d = Q.shape
    B = np.random.randn(L, r)
    B, _ = np.linalg.qr(B)  # Normalize columns to unit length

    for _ in range(max_iter):
        # Compute Z = Q^T B
        Z = Q.T @ B

        # Orthogonalize Z using QR factorization
        Q_Z, R_Z = np.linalg.qr(Z)

        # Compute B = Q Q_Z
        B = Q @ Q_Z

        # Orthogonalize B using QR factorization
        B, _ = np.linalg.qr(B)

    # Compute singular values
    sigma = np.linalg.norm(Z, axis=0)

    return B, sigma


# Example usage
Q = np.array([[4.0, 2.0], [2.0, 3.0], [1.0, 1.0]])
r = 1
B, sigma = block_power_method_svd(Q, r)

print("Approximation of the top r right singular vectors of Q:")
print(B)
print("Approximation of the top r singular values of Q:")
print(sigma)
