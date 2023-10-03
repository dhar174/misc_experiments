import numpy as np
import time
from sklearn.metrics import mean_squared_error


batch_size = 32
sequence_length = 128
embedding_dimension = 64
r = 64  # Number of random features


def test_positive(func, Q, K, V, r):
    try:
        _, por_features = func(Q, K, V, r)  # Assuming the function returns PORFs
        assert (por_features > 0).all(), "Features are not positive"
        print(f"{func.__name__}: Positive Test Passed")
    except Exception as e:
        print("Error in test_positive_orthogonality: ", e)


def test_orthogonality(func, Q, K, V, r):
    try:
        _, por_features = func(Q, K, V, r)  # Assuming the function returns PORFs
        assert np.allclose(
            por_features.T @ por_features, np.eye(r)
        ), "Features are not orthogonal"
        print(f"{func.__name__}: Orthogonality Test Passed")
    except Exception as e:
        print("Error in test_positive_orthogonality: ", e)


def test_computation_speed(func, Q, K, V, r):
    start_time = time.time()
    try:
        func(Q, K, V, r)
        elapsed_time = time.time() - start_time
        print(f"{func.__name__}: Computation Speed = {elapsed_time} seconds")
    except Exception as e:
        elapsed_time = 0
        print("Error in test_computation_speed: ", e)
    return elapsed_time


def test_accuracy(func, Q, K, V, r, true_attention_output):
    try:
        predicted_attention_output, _ = func(Q, K, V, r)
    except Exception as e:
        mse = 0
        print(f"Error in test_accuracy1: {e} in function {func.__name__}")
    try:
        mse = mean_squared_error(true_attention_output, predicted_attention_output)
        print(f"{func.__name__}: Mean Squared Error = {mse}")
    except Exception as e:
        mse = 0
        print(f"Error in test_accuracy2: {e} in function {func.__name__}")
    return mse


def householder_qr(A):
    err_string = ""
    try:
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
    except Exception as e:
        # Q = np.eye(1)
        # R = A.copy()
        print("Error in householder_qr: ", e)
        return Q, R
    return Q, R


def generate_por_features(d, r):
    # Create a random matrix
    A = np.random.randn(d, r)

    # Apply Householder QR factorization to obtain orthogonal features
    Q, _ = householder_qr(A)

    return Q


# Block Power Method
def block_power_method(A, num_eigenvalues=1, max_iter=100):
    m, n = A.shape
    print("m = ", m)
    print("n = ", n)
    Q = np.random.rand(m, num_eigenvalues)
    print("Q.shape = ", Q.shape)
    for _ in range(max_iter):
        Q, _ = householder_qr(A @ Q)
    eigenvalues = np.diag(Q.T @ A @ Q)
    eigenvectors = Q
    print("eigenvalues.shape = ", eigenvalues.shape)
    print("eigenvectors.shape = ", eigenvectors.shape)

    return eigenvalues, eigenvectors


# Rank-1 Update of QR Factorization
def rank1_update_qr(Q, R, u, v):
    A_updated = Q @ R + np.outer(u, v)
    return householder_qr(A_updated)


def favor_plus_attention(Q, K, V, r, redraw_features=True):
    d = Q.shape[-1]
    print(d)
    print(r)
    print(Q.shape)
    print(K.shape)
    print(V.shape)
    try:
        assert Q.shape[-1] == K.shape[-1] == V.shape[-1], "Dimension mismatch"
        assert Q.shape[0] == K.shape[0] == V.shape[0], "Batch size mismatch"
        assert Q.shape[1] == K.shape[1] == V.shape[1], "Sequence length mismatch"
        # Generate Positive Orthogonal Random Features
        por_features = generate_por_features(d, r)
        print(por_features.shape)
    except Exception as e:
        print("Error in favor_plus_attention: ", e)
        return None, None
    # Apply random feature mapping to queries and keys
    try:
        Q_transformed = Q @ por_features
        print("Q_transformed.shape = ", Q_transformed.shape)
        K_transformed = K @ por_features
        print("K_transformed.shape = ", K_transformed.T.shape)
    except Exception as e:
        print("Error in favor_plus_attention: ", e)
        return None, None

    # Optional feature redrawing
    try:
        if redraw_features:
            u = np.random.randn(d)
            v = np.random.randn(r)
            Q_redraw, _ = rank1_update_qr(Q_transformed, np.eye(r), u, v)
            K_redraw, _ = rank1_update_qr(K_transformed, np.eye(r), u, v)
            Q_transformed = Q_redraw
            K_transformed = K_redraw
    except Exception as e:
        print("Error in favor_plus_attention: ", e)
        return None, None

    try:
        # Transpose the last dimension of K (temporary solution for testing)
        K_transposed = np.swapaxes(K, -1, -2)
        print("K_transposed.shape = ", K_transposed.shape)
    except Exception as e:
        print("Error in favor_plus_attention: ", e)
        return None, None
    try:
        # Perform batched matrix multiplication (i.e., compute Q K^T)   (temporary solution for testing)
        attention_scores = np.einsum("bik,bkj->bij", Q_transformed, K_transposed)
        print("attention_scores.shape after einsum", attention_scores.shape)

        attention_scores = attention_scores.reshape(
            batch_size, sequence_length, sequence_length
        )
        print("attention_scores.shape after reshape", attention_scores.shape)
        # Apply attention scores to values
        attention_output = attention_scores @ V
        print("attention_output.shape after projection ", attention_output.shape)
    except Exception as e:
        print("Error in favor_plus_attention: ", e)
        return None, None

    return attention_output, por_features


def favor_plus_attention_no_redraw(Q, K, V, r):
    return favor_plus_attention(Q, K, V, r, redraw_features=False)


def favor_plus_attention_with_redraw(Q, K, V, r):
    return favor_plus_attention(Q, K, V, r, redraw_features=True)


def favor_plus_attention_with_block_power_method(Q, K, V, r):
    d = Q.shape[-1]

    try:
        por_features = generate_por_features(d, r)
    except Exception as e:
        print("Error in favor_plus_attention_with_block_power_method: ", e)
        return None, None, None
    # Apply block power method to random features
    _, eigenvectors = block_power_method(por_features, num_eigenvalues=r)
    transformed_features = por_features @ eigenvectors

    # Apply transformed random feature mapping to queries and keys
    Q_transformed = Q @ transformed_features
    K_transformed = K @ transformed_features

    # Compute attention using associative property
    attention_scores = Q_transformed @ K_transformed.T
    attention_scores = attention_scores.reshape(
        batch_size, sequence_length, sequence_length
    )
    # Apply attention scores to values
    attention_output = attention_scores @ V

    return attention_output, por_features


def main_test(Q, K, V, r, true_attention_output):
    err_string = ""
    functions = [
        favor_plus_attention_no_redraw,
        favor_plus_attention_with_redraw,
        favor_plus_attention_with_block_power_method,
    ]

    i = 0
    for func in functions:
        i += 1
        print(f"Test {i}:")
        print(f"Testing {func.__name__}...")
        # print(f"Q = {Q}")
        print(f"Q.shape = {Q.shape}")
        # print(f"K = {K}")
        print("K.shape = {K.shape}")
        # print(f"V = {V}")
        print(f"V.shape = {V.shape}")
        print(f"r = {r}")
        # print(f"true_attention_output = {true_attention_output}")
        try:
            test_orthogonality(func, Q, K, V, r)
            test_positive(func, Q, K, V, r)
            speed = test_computation_speed(func, Q, K, V, r)

            accuracy = test_accuracy(func, Q, K, V, r, true_attention_output)

            print(f"{func.__name__}: Speed = {speed}, Accuracy = {accuracy}\n")
        except Exception as e:
            print("Error in main_test: ", e)
            continue


# Define queries, keys, and values
# Q = np.array([[1, 2], [3, 4]])
# K = np.array([[5, 6], [7, 8]])
# V = np.array([[9, 10], [11, 12]])

# Define the number of random features
r = 2

# Compute the attention output using FAVOR+
# attention_output = favor_plus_attention(Q, K, V, r)

# print("Attention Output:")
# print(attention_output)
batch_size = 32
sequence_length = 128
embedding_dimension = 64
r = 64  # Number of random features

Q = np.random.randn(batch_size, sequence_length, embedding_dimension)
K = np.random.randn(batch_size, sequence_length, embedding_dimension)
V = np.random.randn(batch_size, sequence_length, embedding_dimension)

# Define weight matrices for the linear transformations
W_q = np.random.randn(embedding_dimension, embedding_dimension)
W_k = np.random.randn(embedding_dimension, embedding_dimension)
W_v = np.random.randn(embedding_dimension, embedding_dimension)

# Apply the linear transformations to Q, K, and V
Q_transformed = Q @ W_q
K_transformed = K @ W_k
V_transformed = V @ W_v


true_attention_output = np.random.randn(
    batch_size, sequence_length, embedding_dimension
)  # Example ground truth

# main_test(Q, K, V, r, true_attention_output)
# Call the test function with the transformed Q, K, and V
main_test(Q_transformed, K_transformed, V_transformed, r, true_attention_output)
