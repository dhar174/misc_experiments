import numpy as np
import math
import scipy.stats as stats


def generate_orthogonal_random_features(d, M):
    # Step 3: Calculate t and n
    t = math.ceil(M / d)
    n = M - d * t

    # Step 4: Draw t i.i.d. random orthogonal matrices
    W_list = []
    for _ in range(t):
        A = np.random.normal(0, 1, (d, d))
        Q, _ = np.linalg.qr(A)
        W_list.append(Q)

    # Step 5: Draw c in R^M
    # Assuming c(d) distribution is standard normal for demonstration purposes
    c = np.random.normal(0, 1, M)

    # Step 6: Construct the final W matrix
    W_final = np.hstack(W_list[:-1] + [W_list[-1][:, :n]])
    W_final *= np.sqrt(d)  # Adjusting for the dimension

    # Multiply with diagonal matrix of c
    W_final = W_final * c

    return W_final


# Test the function
d = 5
M = 16
W = generate_orthogonal_random_features(d, M)
print("Generated W matrix:")
print(W)
