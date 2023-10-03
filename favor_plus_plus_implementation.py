# Let's implement the revised FAVOR++ in Python
import numpy as np
from scipy.stats import ortho_group
import time
from joblib import Parallel, delayed

# import cupy as cp
import torch

# cp.cuda.Device(0).use()
import triton.language as tl
import triton


@triton.jit
def calculate_terms(
    term1,
    term2_Q,
    term2_K,
    term3_Q,
    term3_K,
    Q,
    K,
    omega,
    A: tl.float32,
    B,
    C,
    N,
    T,
    d: tl.constexpr,
    M: tl.constexpr,
    block_size: tl.constexpr,
):
    # Program ID and Thread ID

    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)
    lane_id = tl.arange(0, block_size)
    # Determine the current block using the lane_id
    this_block = tl.program_id(0) * block_size + lane_id

    # Offsets
    offset_omega = pid_m * d
    offset_Q = pid_n * T * d + pid_t * d
    offset_K = pid_n * T * d + pid_t * d

    # Mask for each dimension
    mask_n = pid_n < N
    mask_t = pid_t < T
    mask_m = pid_m < M
    mask = mask_n & mask_t & mask_m

    # Skip if any of the indices are out of bounds
    if not mask:
        return

    # Local Memory
    term2_q_val = tl.zeros([block_size], dtype=tl.float32)
    term2_k_val = tl.zeros([block_size], dtype=tl.float32)
    Q_val = tl.zeros([block_size], dtype=tl.float32)
    K_val = tl.zeros([block_size], dtype=tl.float32)
    term1_val = tl.zeros([block_size], dtype=tl.float32)

    # Create mask for omega_local
    mask_omega = tl.arange(0, d) < d

    # Load omega_local values into a temporary variable
    omega_local = tl.load(omega + offset_omega, mask=mask_m)
    # Perform the operation on each element and accumulate in term1_sum
    A_local = tl.zeros([block_size], dtype=tl.float32)
    A_local = tl.load(A + pid_m, mask=mask_m)
    term1_val = tl.exp(tl.log(omega_local) * 2)  # element-wise squaring
    # term1_val = tl.sum(term1_val, axis=0)
    term1_val = A_local * term1_val

    tl.store(term1 + pid_m, term1_val, mask=mask_m)

    Q_val = tl.load(Q + offset_Q, mask=mask_n)
    K_val = tl.load(K + offset_K, mask=mask_n)

    term2_q_val = omega_local * Q_val
    term2_k_val = omega_local * K_val
    C_local = tl.zeros([block_size], dtype=tl.float32)
    C_local = tl.load(C + pid_m, mask=mask_m)
    term3_q_val = tl.exp(tl.log(Q_val) * 2)  # equivalent to squaring
    term3_q_val = C_local * term3_q_val
    term3_k_val = C_local * tl.exp(tl.log(K_val) * 2)  # equivalent to squaring

    tl.store(term2_Q + pid_n * T * M + pid_t * M + pid_m, term2_q_val, mask=mask)
    tl.store(term2_K + pid_n * T * M + pid_t * M + pid_m, term2_k_val, mask=mask)
    tl.store(term3_Q + pid_n * T + pid_t, term3_q_val, mask=mask_n & mask_t)
    tl.store(term3_K + pid_n * T + pid_t, term3_k_val, mask=mask_n & mask_t)

    # # Load A values into a temporary variable
    # A_local = tl.zeros([block_size], dtype=tl.float32)
    # A_local = tl.load(A + pid_m, mask=mask_m)

    # term1_val = A_local * tl.sum(term1_sum, axis=0)
    # tl.store(term1 + pid_m, term1_val, mask=mask_m)
    # omega_i = tl.zeros([block_size], dtype=tl.float32)
    # # Calculate term2 and term3 values
    # for i in range(0, d):
    #     omega_i = tl.load(omega_local + pid_m * d + i, mask=mask_m)
    #     Q_val = tl.load(Q + pid_n * T * d + pid_t * d + i, mask=mask_n)
    #     K_val = tl.load(K + pid_n * T * d + pid_t * d + i, mask=mask_n)

    #     term2_q_val += omega_i * Q_val
    #     term2_k_val += omega_i * K_val

    # tl.store(term2_Q + pid_n * T * M + pid_t * M + pid_m, term2_q_val, mask=mask_n)
    # tl.store(term2_K + pid_n * T * M + pid_t * M + pid_m, term2_k_val, mask=mask_n)

    # # Calculate term3 for Q and K
    # Q_local = tl.load(Q + pid_n * T * d + pid_t * d, mask=mask_n)
    # K_local = tl.load(K + pid_n * T * d + pid_t * d, mask=mask_n)

    # term3_Q_val = C * tl.sum(Q_local**2)
    # term3_K_val = C * tl.sum(K_local**2)

    # tl.store(term3_Q + pid_n * T + pid_t, term3_Q_val, mask=mask_t)
    # tl.store(term3_K + pid_n * T + pid_t, term3_K_val, mask=mask_t)


MAX_N = 1024


from triton import next_power_of_2

BLOCK_SIZE = 256  # Choose based on your hardware


@triton.jit
def calculate_A_kernel(
    A_result,
    x,
    y,
    sum_x,
    sum_y,
    term1_result,
    term2_result,
    term3_result,
    d: tl.float32,
    N: tl.constexpr,
):
    # Initialize accumulators for sum_x2 and sum_y2
    acc_sum_x2 = tl.zeros([1], dtype=tl.float32)
    acc_sum_y2 = tl.zeros([1], dtype=tl.float32)
    
    # Program ID
    pid_n = tl.program_id(0)
    lane_id = tl.arange(0, N)
    block_start = pid_n * N
    offsets = block_start + lane_id
    # A_result = tl.load(A_result)
    # Calculate sum_x2 and sum_y2
    for i in range(0, N):
        x_val = tl.load(x + i)
        y_val = tl.load(y + i)
        
        # Square using tl.exp and tl.log (x^2 = e^(2 * log(x)))
        x2 = tl.exp(2 * tl.log(x_val))
        y2 = tl.exp(2 * tl.log(y_val))
        
        acc_sum_x2 += x2
        acc_sum_y2 += y2

    # Store the accumulated values
    tl.store(sum_x + offsets, acc_sum_x2)
    tl.store(sum_y + offsets, acc_sum_y2)
    
    # Calculate term1, term2, and term3
    term1_base = 2 * acc_sum_x2 + 2 * acc_sum_y2 + d
    term1 = tl.exp(2 * tl.log(term1_base))

    term2 = 8 * d * (acc_sum_x2 + acc_sum_y2)
    term3 = 2 * acc_sum_x2 + 2 * acc_sum_y2 - d
    
    # Calculate rho_star and A
    rho_star = tl.sqrt(term1 + term2) - term3
    rho_star /= 4 * (acc_sum_x2 + acc_sum_y2)
    A = (1 - 1 / rho_star) / 8
    
    # Store the result
    tl.store(A_result + offsets, A)
    tl.store(term1_result + offsets, term1)
    tl.store(term2_result + offsets, term2)
    tl.store(term3_result + offsets, term3)



@triton.jit
def calculate_A_intermediate_kernel(
    A_intermediate,
    x,
    y,
    sum_x,
    sum_y,
    term1_result,
    term2_result,
    term3_result,
    d: tl.float32,
    N: tl.constexpr,
):
    # Initialize accumulators for sum_x2 and sum_y2
    acc_sum_x2 = tl.zeros([1], dtype=tl.float32)
    acc_sum_y2 = tl.zeros([1], dtype=tl.float32)
    
    # Program ID
    pid_n = tl.program_id(0)
    lane_id = tl.arange(0, N)
    block_start = pid_n * N
    offsets = block_start + lane_id
    
    # Calculate sum_x2 and sum_y2
    for i in range(0, d):
        x_val = tl.load(x + i)
        y_val = tl.load(y + i)
        
        # Square using tl.exp and tl.log (x^2 = e^(2 * log(x)))
        x2 = tl.exp(2 * tl.log(x_val))
        y2 = tl.exp(2 * tl.log(y_val))
        
        acc_sum_x2 += x2
        acc_sum_y2 += y2
    
    # Store the accumulated values
    tl.store(sum_x + offsets, acc_sum_x2)
    tl.store(sum_y + offsets, acc_sum_y2)
    
    # Calculate term1, term2, and term3
    term1_base = 2 * acc_sum_x2 + 2 * acc_sum_y2 + d
    term1 = tl.exp(2 * tl.log(term1_base))

    term2 = 8 * d * (acc_sum_x2 + acc_sum_y2)
    term3 = 2 * acc_sum_x2 + 2 * acc_sum_y2 - d
    
    # Calculate rho_star and A
    rho_star = tl.sqrt(term1 + term2) - term3
    rho_star /= 4 * (acc_sum_x2 + acc_sum_y2)
    A = (1 - 1 / rho_star) / 8
    
    # Atomic add to accumulate A values
    tl.atomic_add(A_intermediate+offsets, A)
    tl.store(term1_result + offsets, term1)
    tl.store(term2_result + offsets, term2)
    tl.store(term3_result + offsets, term3)


@triton.jit
def calculate_A_final_kernel(
    A_result,
    A_intermediate,
    N: tl.constexpr,
):
    # Initialize accumulator for A
    acc_A = tl.zeros([1], dtype=tl.float32)
    
    # Program ID
    pid_n = tl.program_id(0)
    lane_id = tl.arange(0, N)
    block_start = pid_n * N
    offsets = block_start + lane_id
    
    # Accumulate A values
    for i in range(0, N):
        A_val = tl.load(A_intermediate + i)
        acc_A += A_val
    
    # Store the final result
    if pid_n == 0:
        tl.store(A_result + offsets, acc_A)




@triton.jit
def calculate_term1(
    term1,
    debug_pid_m,
    debug_lane_id,
    debug_before_acc,
    debug_after_acc,
    debug_omega_local,
    debug_omega_squared,
    debug_non_zero_pid_m,
    debug_non_zero_omega_local,
    debug_mask_d,
    debug_omega_local_check,
    debug_start_d,
    debug_this_block,
    debug_max_this_block,
    omega,
    A,
    M,
    d: tl.constexpr,
    block_size: tl.constexpr,
):
    # Program ID and Thread ID
    pid_m = tl.program_id(0)
    lane_id = tl.arange(0, block_size)

    # Initialize local accumulator
    acc_term1_val = tl.zeros([block_size], dtype=tl.float32)

    # Loop over the dimension 'd' in chunks of 'block_size'
    for start_d in range(0, d, block_size):
        this_block = start_d + lane_id + pid_m * block_size
        # Inside the loop over 'd'
        tl.store(debug_start_d + pid_m, start_d, mask=pid_m < M)
        # Debug this_block

        mask_d = this_block < d
        # After defining mask_d
        tl.store(debug_mask_d + pid_m * block_size + lane_id, mask_d, mask=mask_d)
        tl.store(
            debug_this_block + pid_m * block_size + lane_id, this_block, mask=mask_d
        )
        omega_local = tl.load(omega + pid_m * block_size + this_block, mask=mask_d)
        # After loading omega_local
        # Store the maximum value of this_block for each pid_m
        max_this_block = tl.max(this_block, axis=0)
        tl.store(debug_max_this_block + pid_m, max_this_block, mask=pid_m < M)

        # Store omega_local to debug tensor
        tl.store(
            debug_omega_local + pid_m * block_size + lane_id, omega_local, mask=mask_d
        )

        omega_local_safe = tl.where(omega_local <= 0.0, 1e-7, omega_local)
        omega_squared = tl.exp(tl.log(omega_local_safe) * 2)

        # Store omega_squared to debug tensor
        tl.store(
            debug_omega_squared + pid_m * block_size + lane_id,
            omega_squared,
            mask=mask_d,
        )
        # Debug tensor to store acc_term1_val before accumulation
        tl.store(
            debug_before_acc + pid_m * block_size + lane_id,
            acc_term1_val,
            mask=lane_id < block_size,
        )
        acc_term1_val = tl.where(mask_d, acc_term1_val + omega_squared, acc_term1_val)

        # Debug tensor to store acc_term1_val after accumulation
        tl.store(
            debug_after_acc + pid_m * block_size + lane_id,
            acc_term1_val,
            mask=lane_id < block_size,
        )
        # Conditional debug for non-zero program IDs
        if pid_m != 0:
            tl.store(debug_non_zero_pid_m + pid_m, pid_m, mask=pid_m < M)
            tl.store(
                debug_non_zero_omega_local + pid_m * block_size + lane_id,
                omega_local,
                mask=mask_d,
            )

    # Post-processing and global write-back
    acc_term1_val = tl.where(acc_term1_val != acc_term1_val, 0.0, acc_term1_val)
    acc_term1_val = tl.where(acc_term1_val > 1e30, 1e30, acc_term1_val)
    acc_term1_val = tl.where(acc_term1_val < -1e30, -1e30, acc_term1_val)

    A_local = tl.load(A)
    acc_term1_val *= A_local

    tl.store(debug_pid_m + pid_m, pid_m, mask=pid_m < M)
    tl.store(
        debug_lane_id + pid_m * block_size + lane_id, lane_id, mask=lane_id < block_size
    )

    tl.store(
        term1 + pid_m * block_size + lane_id, acc_term1_val, mask=lane_id < block_size
    )


@triton.jit
def accumulate_term1(final_term1, term1, A, M, block_size: tl.constexpr):
    pid_m = tl.program_id(0)
    lane_id = tl.arange(0, block_size)

    acc_term1_val = 0.0  # Initialize as scalar

    # Load each thread's term1 value and accumulate
    term1_val = tl.load(
        term1 + pid_m * block_size + lane_id, mask=lane_id < block_size, other=0
    )
    acc_term1_val += tl.sum(term1_val)
    A_local = tl.zeros([block_size], dtype=tl.float32)
    A_local = tl.load(A)

    acc_term1_val *= A_local  # Do this before storing

    # Store the accumulated value back into final_term1 for this pid_m
    tl.store(final_term1 + pid_m, acc_term1_val)


@triton.jit
def calculate_term2(
    term2_Q,
    Q,
    omega,
    N,
    T,
    d,
    M: tl.constexpr,
):
    # Program ID and Thread ID
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)
    lane_id = tl.arange(0, M)

    # Initialize an accumulator to hold a vector of length M
    acc_term2_val = tl.zeros([M], dtype=tl.float32)

    for start_d in range(0, d, M):
        this_block = start_d + lane_id
        mask_d = this_block < d

        # Load Q values
        Q_local = tl.load(Q + pid_n * T * d + pid_t * d + this_block, mask=mask_d)

        # Initialize a temporary variable to hold the result of Q_local * omega_local
        temp_val = tl.zeros([M], dtype=tl.float32)

        # Update the accumulation logic
        for m in range(0, M):
            omega_local = tl.load(omega + m * d + this_block, mask=mask_d)
            temp_val = Q_local * omega_local
            acc_term2_val = tl.where(
                lane_id == m, acc_term2_val + tl.sum(temp_val, axis=0), acc_term2_val
            )

        # Store the vector in the output tensor term2_Q
        tl.store(
            term2_Q + pid_n * T * M + pid_t * M + start_d + lane_id,
            acc_term2_val,
            mask=mask_d,
        )


@triton.jit
def calculate_term3(
    term3_Q,
    Q,
    N,
    T,
    d,
    block_size: tl.constexpr,
):
    # Program ID and Thread ID
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    lane_id = tl.arange(0, block_size)

    # Initialize a scalar variable to accumulate term3_Q_val
    acc_term3_val = tl.zeros([block_size], dtype=tl.float32)

    # Loop over the dimension 'd' in chunks of 'block_size'
    for start_d in range(0, d, block_size):
        this_block = start_d + lane_id
        mask_d = this_block < d

        # Load Q values
        Q_local = tl.load(Q + pid_n * T * d + pid_t * d + this_block, mask=mask_d)

        # Perform the square and accumulation
        # acc_term3_val += Q_local**2
        acc_term3_val = tl.exp(tl.log(Q_local) * 2)

    # Reduce the block into a single scalar value
    acc_term3_val = tl.sum(acc_term3_val, axis=0)

    # Multiply by constant C
    C = -(1 + 1) / 2
    acc_term3_val *= C

    # Store the result into term3_Q
    tl.store(term3_Q + pid_n * T + pid_t, acc_term3_val)


@triton.jit
def calculate_f_ge(
    f_ge_Q,
    term1,
    term2_Q,
    term3_Q,
    D,
    N,
    T,
    M,
    block_size: tl.constexpr,
):
    # Program ID and Thread ID
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)
    lane_id = tl.arange(0, block_size)

    # Load precomputed term1, term2_Q, and term3_Q values
    term1_local = tl.load(term1 + pid_n * T + pid_t)
    term2_Q_local = tl.load(term2_Q + pid_n * T * M + pid_t * M + pid_m)
    term3_Q_local = tl.load(term3_Q + pid_n * T + pid_t)

    # Compute the sum and apply the exponential function
    sum_terms = term1_local + term2_Q_local + term3_Q_local
    exp_sum_terms = tl.exp(sum_terms)

    # Multiply by constant D
    D_local = tl.load(D)
    f_ge_Q_val = D_local * exp_sum_terms

    # Store the result into f_ge_Q
    tl.store(f_ge_Q + pid_n * T * M + pid_t * M + pid_m, f_ge_Q_val)


@triton.jit
def calculate_attention_weights(
    attention_weights,
    f_ge_Q,
    f_ge_K,
    N: tl.constexpr,
    T: tl.constexpr,
    M: tl.constexpr,
    block_size: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)
    lane_id = tl.arange(0, block_size)
    # Offsets
    block_start_Q = pid_n * T * M + pid_t * M
    block_start_K = pid_n * T * M + pid_t * M

    offsets_Q = block_start_Q + lane_id
    offsets_K = block_start_K + lane_id
    mask_n = pid_n < N
    mask_t = pid_t < T
    mask_m = pid_m < M
    mask = mask_n & mask_t & mask_m
    # offset_Q = pid_n * T * M + pid_t * M + pid_m
    # offset_K = pid_n * T * M + pid_t * M + pid_m
    if not mask:
        return
    # Create masks for valid data points
    mask_Q = offsets_Q < (N * T * M)
    mask_K = offsets_K < (N * T * M)
    # Local Memory
    f_ge_Q_local = tl.zeros([block_size], dtype=tl.float32)
    f_ge_K_local = tl.zeros([block_size], dtype=tl.float32)
    attention_val = tl.zeros([block_size], dtype=tl.float32)

    # Load f_ge_Q and f_ge_K values into local memory
    f_ge_Q_local = tl.load(f_ge_Q + offsets_Q, mask=mask_Q)
    f_ge_K_local = tl.load(f_ge_K + offsets_K, mask=mask_K)

    # Compute element-wise multiplication and sum across the last dimension
    attention_val = f_ge_Q_local * f_ge_K_local
    attention_val = tl.sum(attention_val, axis=0)

    # Store the result back
    tl.store(attention_weights + pid_n * T + pid_t, attention_val, mask=mask)


def test_equivalence_A(A_triton, d, Q, K, x, y, sumx2_triton,sumy2_triton, term1_triton, term2_triton, term3_triton):
    print("Testing equivalence of A")
    x_sample = x
    y_sample = y
    A_triton = A_triton[0]
    sum_x2 = torch.sum(x**2)
    print(f"sum_x2: {sum_x2}")
    print(f"sum_x2_triton: {sumx2_triton}")
    sum_x2_diff = torch.abs(sum_x2 - sumx2_triton)
    print(f"sum_x2_diff: {sum_x2_diff}")
    sum_y2 = torch.sum(y**2)
    print(f"sum_y2: {sum_y2}")
    print(f"sum_y2_triton: {sumy2_triton}")
    sum_y2_diff = torch.abs(sum_y2 - sumy2_triton)
    print(f"sum_y2_diff: {sum_y2_diff}")
    term1 = (2 * sum_x2 + 2 * sum_y2 + d) ** 2
    print(f"term1: {term1}")
    print(f"term1_triton: {term1_triton}")
    term1_diff = torch.abs(term1 - term1_triton)
    print(f"term1_diff: {term1_diff}")
    term2 = 8 * d * (sum_x2 + sum_y2)
    print(f"term2: {term2}")
    print(f"term2_triton: {term2_triton}")
    term2_diff = torch.abs(term2 - term2_triton)
    print(f"term2_diff: {term2_diff}")
    term3 = 2 * sum_x2 + 2 * sum_y2 - d
    print(f"term3: {term3}")
    print(f"term3_triton: {term3_triton}")
    term3_diff = torch.abs(term3 - term3_triton)
    print(f"term3_diff: {term3_diff}")
    rho_star = torch.sqrt(term1 + term2) - term3
    rho_star /= 4 * (sum_x2 + sum_y2)
    A = (1 - 1 / rho_star) / 8
    print(f"A_triton: {A_triton}")
    print(f"A: {A}")
    diff = torch.abs(A_triton - A)
    max_diff = torch.max(diff)
    print(f"Maximum difference between Triton and PyTorch results: {max_diff.item()}")
    print(f"Average difference between Triton and PyTorch results: {torch.mean(diff)}")
    print(
        f"Standard deviation of difference between Triton and PyTorch results: {torch.std(diff)}"
    )
    print(f"Minimum difference between Triton and PyTorch results: {torch.min(diff)}")
    print(f"Median difference between Triton and PyTorch results: {torch.median(diff)}")
    print(f"Total number of elements: {torch.numel(diff)}")
    print(f"Number of elements that differ by more than 1e-6: {torch.sum(diff > 1e-6)}")

    assert max_diff.item() < 1e-6, "The two implementations are not equivalent"


def test_equivalence_term1(term1_triton, omega, A, M, d, BLOCK_SIZE, x_sample, y_sample):
    print("Testing equivalence of term1")
    A = A[0]    
    print(f"A: {A}")
    print(f"A.type: {type(A)}")
    print(f"A.shape: {A.shape}")
    print(f"A.numel: {A.numel()}")
    print(f"A.dtype: {A.dtype}")
    print(f"A.device: {A.device}")
    print(f"omega: {omega}")
    A_local = calculate_A_torch(x_sample, y_sample, d)
    print(f"A_local: {A_local}")
    print(f"A_local.type: {type(A_local)}")
    print(f"A_local.shape: {A_local.shape}")
    print(f"A_local.numel: {A_local.numel()}")
    print(f"A_local.dtype: {A_local.dtype}")
    assert A_local.allclose(A), "A is not equivalent"
    print("A is equivalent")
    B = torch.sqrt((1 - 4 * A))
    C = -(1 + 1) / 2
    D = torch.sqrt((1 - 4 * A)) ** (d / 2)
    term1_torch = A * torch.sum(omega**2, axis=1)


    print(f"full term1_torch: {term1_torch}")
    diff = torch.abs(term1_triton - term1_torch)
    print(f"full term1_triton: {term1_triton}")
    max_diff = torch.max(diff)

    print(f"Maximum difference between Triton and PyTorch results: {max_diff.item()}")
    print(f"Average difference between Triton and PyTorch results: {torch.mean(diff)}")
    print(
        f"Standard deviation of difference between Triton and PyTorch results: {torch.std(diff)}"
    )
    print(f"Minimum difference between Triton and PyTorch results: {torch.min(diff)}")
    print(f"Median difference between Triton and PyTorch results: {torch.median(diff)}")
    print(f"Total number of elements: {torch.numel(diff)}")
    print(f"Number of elements that differ by more than 1e-6: {torch.sum(diff > 1e-6)}")
    print(f"Number of elements that differ by more than 1e-3: {torch.sum(diff > 1e-3)}")
    print(f"Number of elements that differ by more than 1e-1: {torch.sum(diff > 1e-1)}")
    print(f"Number of elements that differ by more than 1e0: {torch.sum(diff > 1e0)}")
    print(f"Number of elements that differ by more than 1e1: {torch.sum(diff > 1e1)}")
    print(f"Number of elements that differ by more than 1e2: {torch.sum(diff > 1e2)}")
    print(f"Number of elements that differ by more than 1e3: {torch.sum(diff > 1e3)}")
    print(f"Number of elements that differ by more than 1e4: {torch.sum(diff > 1e4)}")

    assert max_diff.item() < 1e-6, "The two implementations are not equivalent"


def test_equivalence_with_precomputed_result(term2_Q_triton, Q, omega):
    print("Testing equivalence of term2")
    # Compare with PyTorch einsum
    term2_Q_torch = torch.einsum("ijk,lk->ijl", Q, omega)

    # Compare the two results
    diff = torch.abs(term2_Q_triton - term2_Q_torch)
    max_diff = torch.max(diff)
    print(f"Maximum difference between Triton and PyTorch results: {max_diff.item()}")
    print(f"Average difference between Triton and PyTorch results: {torch.mean(diff)}")
    print(
        f"Standard deviation of difference between Triton and PyTorch results: {torch.std(diff)}"
    )
    print(f"Minimum difference between Triton and PyTorch results: {torch.min(diff)}")
    print(f"Median difference between Triton and PyTorch results: {torch.median(diff)}")
    print(f"Total number of elements: {torch.numel(diff)}")
    print(f"Number of elements that differ by more than 1e-6: {torch.sum(diff > 1e-6)}")
    print(f"Number of elements that differ by more than 1e-3: {torch.sum(diff > 1e-3)}")
    print(f"Number of elements that differ by more than 1e-1: {torch.sum(diff > 1e-1)}")
    print(f"Number of elements that differ by more than 1e0: {torch.sum(diff > 1e0)}")
    print(f"Number of elements that differ by more than 1e1: {torch.sum(diff > 1e1)}")
    print(f"Number of elements that differ by more than 1e2: {torch.sum(diff > 1e2)}")
    print(f"Number of elements that differ by more than 1e3: {torch.sum(diff > 1e3)}")
    print(f"Number of elements that differ by more than 1e4: {torch.sum(diff > 1e4)}")

    assert max_diff.item() < 1e-6, "The two implementations are not equivalent"


@triton.jit
def calculate_attention_output(
    attention_output,
    attention_weights,
    V,
    N: tl.constexpr,
    T: tl.constexpr,
    d: tl.constexpr,
    block_size: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    lane_id = tl.arange(0, block_size)

    mask_n = pid_n < N
    mask_t = pid_t < T
    mask = mask_n & mask_t

    if not mask:
        return

    # Local Memory
    attention_weights_local = tl.zeros([block_size], dtype=tl.float32)
    V_local = tl.zeros([block_size], dtype=tl.float32)

    # Load attention_weights and V values into local memory
    attention_weights_local = tl.load(attention_weights + pid_n * T + pid_t, mask=mask)
    V_local = tl.load(V + pid_n * T * d + pid_t * d, mask=mask)

    # Compute matrix multiplication
    output_val = attention_weights_local * V_local
    output_val = tl.sum(output_val, axis=0)

    tl.store(attention_output + pid_n * T * d + pid_t * d, output_val, mask=mask)


def make_lut(layout, block, device):
    _empty = torch.tensor([], dtype=torch.int64, device=layout.device)
    sizes = _empty.clone()
    # sizes along rows
    for h in range(layout.shape[0]):
        sizes = torch.cat((sizes, layout[h, :, :].sum(-1)))
    total_sizes = sizes * block
    # offsets in block format
    offsets = torch.zeros_like(sizes)
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    # block indices
    columns = layout.nonzero(as_tuple=False)[:, 2]
    header = torch.stack((sizes, offsets), dim=1).view(-1)
    lut = torch.cat((header, columns)).type(torch.int32).to(device)
    return lut, int(total_sizes.max())


# Function that calls the kernels
def self_attention_triton(Q, K, V, omega, M):
    N, T, d = Q.shape

    # Prepare an output tensor for A
    A = torch.zeros(1, dtype=torch.float32).to("cuda")

    # Prepare buffers for the sums
    sum_x2 = torch.zeros(N, dtype=torch.float32).to("cuda")
    sum_y2 = torch.zeros(N, dtype=torch.float32).to("cuda")
    print(f"N: {N}")
    print(f"M: {M}")
    print(f"T: {T}")
    print(f"Q shape: {Q.shape}")
    print(f"Q size: {Q.size}")
    print(f"K shape: {K.shape}")
    print(f"K size: {K.size}")
    # Set the grid
    grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)
    print(f"Grid size: {grid(None)}")
    print(f"Block size: {BLOCK_SIZE}")
    print(f"Number of blocks: {triton.cdiv(N, BLOCK_SIZE)}")
    print("d: ", d)
    print("d type: ", type(d))
    x_sample = torch.mean(Q, dim=(0, 1))
    y_sample = torch.mean(K, dim=(0, 1))
    print(f"x_sample shape: {x_sample.shape}")
    print(f"y_sample shape: {y_sample.shape}")
    print(f"x_sample type: {type(x_sample)}")
    print(f"y_sample type: {type(y_sample)}")
    print(f"x_sample: {x_sample}")
    print(f"y_sample: {y_sample}")
    empty_term1 = torch.zeros(N * T, dtype=torch.float32).to("cuda")
    empty_term2 = torch.zeros(N * T * M, dtype=torch.float32).to("cuda")
    empty_term3 = torch.zeros(N * T, dtype=torch.float32).to("cuda")
    # Call the kernel
    # calculate_A_kernel[grid](A,x_sample, y_sample, sum_x2, sum_y2, empty_term1, empty_term2, empty_term3, d, N)
    # Intermediate kernel call
    grid_intermediate = lambda meta: (triton.cdiv(d, BLOCK_SIZE),)
    A_intermediate = torch.zeros([d], dtype=torch.float32, device="cuda")
    calculate_A_intermediate_kernel[grid_intermediate](A_intermediate, x_sample, y_sample, sum_x2, sum_y2, empty_term1, empty_term2, empty_term3, d, N)
    print(f"A_intermediate shape: {A_intermediate.shape}")
    print(f"A_intermediate type: {type(A_intermediate)}")
    print(f"A_intermediate: {A_intermediate}")
    # Final kernel call
    grid_final = lambda meta: (1,)
    A_result = torch.zeros([1], dtype=torch.float32, device="cuda")
    calculate_A_final_kernel[grid_final](A_result, A_intermediate, N)
    print(f"A_result shape: {A_result.shape}")
    print(f"A_result type: {type(A_result)}")
    print(f"A_result: {A_result}")
    print(f"sum_x2 shape: {sum_x2.shape}")
    print(f"sum_y2 shape: {sum_y2.shape}")
    print(f"sum_x2 type: {type(sum_x2)}")
    print(f"sum_y2 type: {type(sum_y2)}")
    print(f"sum_x2: {sum_x2}")
    B = torch.sqrt((1 - 4 * A_result))
    C = -(1 + 1) / 2
    D = torch.sqrt((1 - 4 * A_result)) ** (d / 2)

    C = torch.tensor(C, device="cuda")
    # D = D.clone().detach()
    print(f"D type: {type(D)}")
    print(f"D: {D}")

    print(f"C type: {type(C)}")
    print(f"C: {C}")
    

    test_equivalence_A(A_intermediate, d, Q, K, x_sample, y_sample, sum_x2, sum_y2, empty_term1, empty_term2, empty_term3)
    # Prepare buffers for the terms
    term1 = torch.zeros([M], dtype=torch.float32, device="cuda")
    term2_Q = torch.zeros([N, T, M], dtype=torch.float32, device="cuda")
    term2_K = torch.zeros([N, T, M], dtype=torch.float32, device="cuda")
    term3_Q = torch.zeros([N, T], dtype=torch.float32, device="cuda")
    term3_K = torch.zeros([N, T], dtype=torch.float32, device="cuda")
    print(f"type of omega: {type(omega)}")
    print(f"omega.shape: {omega.shape}")
    print(f"omega: {omega}")
    print(f"type of A: {type(A)}")
    print(f"A.shape: {A.shape}")
    print(f"A: {A}")
    # print(f"A: {A}")
    # Calculate the terms
    # Set up the grid
    # grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE),)
    grid = lambda meta: (M,)
    block_size=d

    if d >= BLOCK_SIZE:
        block_size = BLOCK_SIZE


    print(f"Grid size: {grid(None)}")
    print(f"Block size: {block_size}")
    debug_pid_m = torch.zeros(M, dtype=torch.int32, device="cuda:0")
    debug_lane_id = torch.zeros((M, block_size), dtype=torch.int32, device="cuda:0")
    debug_before_acc = torch.zeros(
        (M, block_size), dtype=torch.float32, device="cuda:0"
    )
    debug_after_acc = torch.zeros((M, block_size), dtype=torch.float32, device="cuda:0")
    debug_omega_local = torch.zeros(
        (M, block_size), dtype=torch.float32, device="cuda:0"
    )
    debug_omega_squared = torch.zeros(
        (M, block_size), dtype=torch.float32, device="cuda:0"
    )
    debug_non_zero_pid_m = torch.zeros(M, dtype=torch.int32, device="cuda:0")
    debug_non_zero_omega_local = torch.zeros(
        (M, block_size), dtype=torch.float32, device="cuda:0"
    )
    debug_mask_d = torch.zeros((M, block_size), dtype=torch.bool, device="cuda:0")
    debug_omega_local_check = torch.zeros(
        (M, block_size), dtype=torch.float32, device="cuda:0"
    )
    debug_start_d = torch.zeros(M, dtype=torch.int32, device="cuda:0")
    debug_this_block = torch.zeros((M, block_size), dtype=torch.int32, device="cuda:0")
    debug_max_this_block = torch.zeros(M, dtype=torch.int32, device="cuda:0")
    A[0] = A_intermediate[0].to("cuda:0")
    print(f"A: {A}")
    print(f"A type: {type(A)}")
    print(f"A shape: {A.shape}")
    # Call the kernel
    calculate_term1[grid](
        term1,
        debug_pid_m,
        debug_lane_id,
        debug_before_acc,
        debug_after_acc,
        debug_omega_local,
        debug_omega_squared,
        debug_non_zero_pid_m,
        debug_non_zero_omega_local,
        debug_mask_d,
        debug_omega_local_check,
        debug_start_d,
        debug_this_block,
        debug_max_this_block,
        omega,
        A,
        M,
        d,
        block_size,
    ),

    #Accumulate term1
    final_term1 = torch.zeros([1], dtype=torch.float32, device="cuda")
    grid = lambda meta: (triton.cdiv(M, block_size),)
    accumulate_term1[grid](final_term1, term1, A, M, block_size)
    print(f"final_term1: {final_term1}")

    # print(f"debug_pid_m is {debug_pid_m.cpu()}")
    # print(f"debug_lane_id is {debug_lane_id.cpu()}")
    # print(f"debug_before_acc is {debug_before_acc.cpu()}")
    # print(f"debug_after_acc is {debug_after_acc.cpu()}")
    # print(f"debug_omega_local is {debug_omega_local.cpu()}")
    # print(f"debug_omega_squared is {debug_omega_squared.cpu()}")
    # print(f"debug_non_zero_pid_m is {debug_non_zero_pid_m.cpu()}")
    # print(f"debug_non_zero_omega_local is {debug_non_zero_omega_local.cpu()}")
    # print(f"debug_mask_d is {debug_mask_d.cpu()}")
    # print(f"debug_omega_local_check is {debug_omega_local_check.cpu()}")
    # print(f"debug_start_d is {debug_start_d.cpu()}")
    # print(f"debug_this_block is {debug_this_block.cpu()}")

    # print(f"debug_max_this_block is {debug_max_this_block.cpu()}")
    
    print(f"\n term1 calculated")
    print(f"\n term1: {term1}")
    print(f"\n term1 type: {type(term1)}")
    print(f"\n term1 shape: {term1.shape}")
    # Initialize final output tensor

    print(f"\n term1 calculated")
    
    test_equivalence_term1(term1, omega, A, M, d, BLOCK_SIZE, x_sample, y_sample)
    # Set up the grid
    # Define grid dimensions
    grid = lambda opt: (
        triton.cdiv(N, BLOCK_SIZE),
        triton.cdiv(T, BLOCK_SIZE),
        triton.cdiv(M, BLOCK_SIZE),
    )

    print(f"Grid size: {grid(None)}")
    print(f"N type: {type(N)}")
    print(f"T type: {type(T)}")
    print(f"M type: {type(M)}")
    print(f"BLOCK_SIZE type: {type(BLOCK_SIZE)}")
    # Initialize term2_Q and term2_K tensors
    # Call the kernels
    print(f"shape of term2_Q: {term2_Q.shape}")
    # calculate_term2_val[grid](
    #     term2_Q,
    #     Q,
    #     omega,
    #     N,
    #     T,
    #     d,
    #     M,
    #     BLOCK_SIZE,
    # )
    debug_pid_n = torch.zeros(N, dtype=torch.int32, device="cuda")
    debug_pid_t = torch.zeros(T, dtype=torch.int32, device="cuda")
    debug_pid_m = torch.zeros(M, dtype=torch.int32, device="cuda")
    debug_Q_local = torch.zeros(
        (N, T, BLOCK_SIZE), dtype=torch.float32, device="cuda:0"
    )
    debug_omega_local = torch.zeros(
        (M, BLOCK_SIZE), dtype=torch.float32, device="cuda:0"
    )

    grid = lambda meta: (N, T, M)
    calculate_term2[grid](
        term2_Q,
        Q,
        omega,
        N,
        T,
        d,
        M,
    )

    print(f"term2_Q calculated")
    calculate_term2[grid](
        term2_K,
        Q,
        omega,
        N,
        T,
        d,
        M,
    )
    test_equivalence_with_precomputed_result(term2_Q, Q, omega)
    exit(0)
    print(f"term2_K calculated")
    print(f"debug_pid_m2 is: {debug_pid_m2}")
    print(f"debug_lane_id2 is: {debug_lane_id2}")
    print(f"debug_omega_local2 is: {debug_omega_local2}")
    print(f"debug_mask_d2 is: {debug_mask_d2}")
    print(f"debug_start_d2 is: {debug_start_d2}")
    print(f"debug_this_block2 is: {debug_this_block2}")
    print(f"debug_max_this_block2 is: {debug_max_this_block2}")
    calculate_term3[grid](term3_Q, Q, N, T, d, BLOCK_SIZE)
    calculate_term3[grid](term3_K, K, N, T, d, BLOCK_SIZE)
    # calculate_terms[grid](
    #     term1,
    #     term2_Q,
    #     term2_K,
    #     term3_Q,
    #     term3_K,
    #     Q,
    #     K,
    #     omega,
    #     A,
    #     B,
    #     C,
    #     N,
    #     T,
    #     d,
    #     M,
    #     BLOCK_SIZE,
    # )

    print(f"terms calculated")
    print(f"term1: {term1}")
    print(f"term1 type: {type(term1)}")
    print(f"term1 shape: {term1.shape}")
    print(f"term1[0]: {term1[0]}")
    print(f"term1[0] type: {type(term1[0])}")
    print(f"term1[0] shape: {term1[0].shape}")
    print(f"term1[1]: {term1[1]}")
    print(f"term2_Q: {term2_Q}")
    print(f"term2_Q type: {type(term2_Q)}")
    print(f"term2_Q shape: {term2_Q.shape}")
    print(f"term2_Q[0]: {term2_Q[0]}")
    print(f"term3_Q: {term3_Q}")
    print(f"term3_Q type: {type(term3_Q)}")
    print(f"term3_Q shape: {term3_Q.shape}")
    print(f"term3_Q[0]: {term3_Q[0]}")
    print(f"term3_Q[0] type: {type(term3_Q[0])}")
    # Prepare buffers for f_ge_Q and f_ge_K
    f_ge_Q = torch.zeros((N, T, M), dtype=torch.float32, device="cuda")
    f_ge_K = torch.zeros((N, T, M), dtype=torch.float32, device="cuda")
    attention_output = torch.zeros((N, T, d), dtype=torch.float32, device="cuda")
    print(f"D type: {type(D)}")
    print(f"D: {D}")
    print(f"D shape: {D.shape}")

    # Step 2: Calculate f_ge for Q and K
    grid = lambda opt: (N, T, M)  # Grid size depends on the dimensions of Q and K
    calculate_f_ge[grid](
        f_ge_Q,
        term1,
        term2_Q,
        term3_Q,
        D,
        N,
        T,
        M,
        BLOCK_SIZE,
    )

    calculate_f_ge[grid](
        f_ge_K,
        term1,
        term2_K,
        term3_K,
        D,
        N,
        T,
        M,
        BLOCK_SIZE,
    )

    torch.cuda.synchronize()
    print(f"f_ge_Q calculated")
    print(f"f_ge_Q: {f_ge_Q}")
    print(f"f_ge_Q type: {type(f_ge_Q)}")
    print(f"f_ge_Q tensor type: {f_ge_Q.type()}")
    print(f"f_ge_Q shape: {f_ge_Q.shape}")
    print(f"f_ge_Q[0]: {f_ge_Q[0]}")
    print(f"f_ge_Q[0] type: {type(f_ge_Q[0])}")
    print(f"f_ge_K: {f_ge_K}")
    print(f"f_ge_K type: {type(f_ge_K)}")
    print(f"f_ge_K shape: {f_ge_K.shape}")
    # Step 3: Calculate Attention Output
    # Initialize attention_weights and attention_output tensors
    attention_weights = torch.zeros([N, T, T], dtype=torch.float32).cuda()
    attention_output = torch.zeros([N, T, d], dtype=torch.float32).cuda()

    # Define grid dimensions for attention_weights and attention_output
    grid_attention_weights = lambda opt: (
        triton.cdiv(N, BLOCK_SIZE),
        triton.cdiv(T, BLOCK_SIZE),
        triton.cdiv(T, BLOCK_SIZE),
    )

    grid_attention_output = lambda opt: (
        triton.cdiv(N, BLOCK_SIZE),
        triton.cdiv(T, BLOCK_SIZE),
    )

    # Call the kernels
    calculate_attention_weights[grid_attention_weights](
        attention_weights, f_ge_Q, f_ge_K, N, T, T, BLOCK_SIZE
    )
    print(f"attention weights calculated")
    print(f"attention_weights: {attention_weights}")
    print(f"attention_weights type: {type(attention_weights)}")
    print(f"attention_weights shape: {attention_weights.shape}")
    calculate_attention_output[grid_attention_output](
        attention_output, attention_weights, V, N, T, d, BLOCK_SIZE
    )
    print(f"attention output calculated")
    print(f"attention_output: {attention_output}")
    print(f"attention_output type: {type(attention_output)}")
    print(f"attention_output shape: {attention_output.shape}")
    # Now, attention_output contains the result.
    return attention_output


def calculate_A_torch(x, y, d):
    sum_x2 = torch.sum(x**2)
    sum_y2 = torch.sum(y**2)
    term1 = (2 * sum_x2 + 2 * sum_y2 + d) ** 2
    term2 = 8 * d * (sum_x2 + sum_y2)
    term3 = 2 * sum_x2 + 2 * sum_y2 - d
    rho_star = torch.sqrt(term1 + term2) - term3
    rho_star /= 4 * (sum_x2 + sum_y2)
    A = (1 - 1 / rho_star) / 8
    return A


def f_GE_torch(omega, x, A, B, C, D):
    return D * torch.exp(
        A * torch.sum(omega**2) + B * torch.matmul(omega, x) + C * torch.sum(x**2)
    )


def vectorized_self_attention_torch(Q, K, V, omega, M, debug=False, gpu=False):
    N, T, d = Q.shape  # Batch size, sequence length, feature dimension
    x_sample = torch.mean(Q, dim=(0, 1))
    y_sample = torch.mean(K, dim=(0, 1))
    A = calculate_A_torch(x_sample, y_sample, d)


    print(f"A: {A}")
    print(f"Type of A: {type(A)}")
    print(f"Shape of A: {A.shape}")
    B = torch.sqrt((1 - 4 * A))
    C = -(1 + 1) / 2
    D = torch.sqrt((1 - 4 * A)) ** (d / 2)
    print(f"D: {D}")
    print(f"omega: {omega}")
    print(f"omega shape: {omega.shape}")
    # Compute terms without storing the OPRFs tensor
    term1 = A * torch.sum(omega**2, dim=1)
    if debug:
        print(f"term1 in vectorized torch: {term1}")
        print(f"term1 type: {type(term1)}")
        print(f"term1 shape: {term1.shape}")
    term2_Q = torch.einsum("ijk,lk->ijl", Q, omega)
    if debug:
        print(f"term2_Q: {term2_Q}")
        print(f"term2_Q type: {type(term2_Q)}")
        print(f"term2_Q shape: {term2_Q.shape}")
    term2_K = torch.einsum("ijk,lk->ijl", K, omega)
    if debug:
        print(f"term2_K: {term2_K}")
        print(f"term2_K type: {type(term2_K)}")
        print(f"term2_K shape: {term2_K.shape}")
    term3_Q = C * torch.sum(Q**2, dim=2)[:, :, None]
    if debug:
        print(f"term3_Q: {term3_Q}")
        print(f"term3_Q type: {type(term3_Q)}")
        print(f"term3_Q shape: {term3_Q.shape}")
    term3_K = C * torch.sum(K**2, dim=2)[:, :, None]
    if debug:
        print(f"term3_K: {term3_K}")
        print(f"term3_K type: {type(term3_K)}")
        print(f"term3_K shape: {term3_K.shape}")

    f_ge_Q = D * torch.exp(term1 + term2_Q + term3_Q)
    f_ge_K = D * torch.exp(term1 + term2_K + term3_K)
    if debug:
        print(f"f_ge_Q: {f_ge_Q}")
        print(f"f_ge_Q type: {type(f_ge_Q)}")
        print(f"f_ge_Q shape: {f_ge_Q.shape}")
        print(f"f_ge_Q[0]: {f_ge_Q[0]}")
        print(f"f_ge_Q[0] type: {type(f_ge_Q[0])}")
        print(f"f_ge_K shape: {f_ge_K.shape}")
    attention_weights = torch.sum(f_ge_Q[:, :, None, :] * f_ge_K[:, None, :, :], dim=-1)
    attention_output = torch.matmul(attention_weights, V)
    return attention_output


def generate_random_orthogonal_matrix(d, M):
    # Generate an M x d orthogonal matrix
    return ortho_group.rvs(dim=d)[:M]


def calculate_A(x, y, d):
    term1 = (2 * np.sum(x**2) + 2 * np.sum(y**2) + d) ** 2
    term2 = 8 * d * (np.sum(x**2) + np.sum(y**2))
    term3 = 2 * np.sum(x**2) + 2 * np.sum(y**2) - d
    rho_star = np.sqrt(term1 + term2) - term3
    rho_star /= 4 * (np.sum(x**2) + np.sum(y**2))
    A = (1 - 1 / rho_star) / 8
    return A

once = False
def f_GE(omega, x, A, B, C, D):
    global once
    if not once:
        print(f"Inside f_GE from regular method")
        print(f"Term 1: {A * np.sum(omega**2)}")
        print(f"Term 2: {B * np.dot(omega, x)}")
        print(f"Term 3: {C * np.sum(x**2)}")
        print(f"Term 4: {D}")
        print(f"Term 5: {np.exp(A * np.sum(omega**2) + B * np.dot(omega, x) + C * np.sum(x**2))}")
        print(f"Term 6: {D * np.exp(A * np.sum(omega**2) + B * np.dot(omega, x) + C * np.sum(x**2))}")
        once = True
    return D * np.exp(
        A * np.sum(omega**2) + B * np.dot(omega, x) + C * np.sum(x**2)
    )


def self_attention(Q, K, V, omega, M, kwargs):
    if kwargs.get("triton"):
        return self_attention_triton(Q, K, V, omega, M)
    if kwargs.get("torch"):
        if kwargs.get("gpu"):
            gpu_enabled = kwargs["gpu"]
        else:
            gpu_enabled = False
        return vectorized_self_attention_torch(
            Q, K, V, omega, M, debug=kwargs["debug"], gpu=gpu_enabled
        )
    # if kwargs.get("gpu"):
    #     Q = cp.array(Q)
    #     K = cp.array(K)
    #     V = cp.array(V)
    #     omega = cp.array(omega)
    #     return gpu_self_attention(Q, K, V, omega, M, debug=kwargs["debug"])
    if kwargs.get("vectorized"):
        return vectorized_self_attention(Q, K, V, omega, M, debug=kwargs["debug"])
    if kwargs.get("parallel"):
        return batched_self_attention_parallel(
            Q, K, V, omega, M, n_jobs=kwargs["n_jobs"], debug=kwargs["debug"]
        )

    return batched_self_attention(Q, K, V, omega, M, debug=False)


def vectorized_self_attention(Q, K, V, omega, M, debug=False):

    N, T, d = Q.shape  # Batch size, sequence length, feature dimension
    # Calculate A for the whole batch
    x_sample = np.mean(Q, axis=(0, 1))
    y_sample = np.mean(K, axis=(0, 1))
    A = calculate_A(x_sample, y_sample, d)

    # Calculate B, C, D using A and s=+1
    B = np.sqrt((1 - 4 * A))
    C = -(1 + 1) / 2
    D = np.sqrt((1 - 4 * A)) ** (d / 2)

    OPRFs = np.zeros((N, T, T, M))

    # Using broadcasting and np.einsum for vectorized computation
    term1 = A * np.sum(omega**2, axis=1)
    print(f"term1 vectorized: {term1}")
    term2_Q = np.einsum("ijk,lk->ijl", Q, omega)
    term2_K = np.einsum("ijk,lk->ijl", K, omega)
    term3_Q = C * np.sum(Q**2, axis=2)[:, :, np.newaxis]
    term3_K = C * np.sum(K**2, axis=2)[:, :, np.newaxis]

    f_ge_Q = D * np.exp(term1 + term2_Q + term3_Q)
    f_ge_K = D * np.exp(term1 + term2_K + term3_K)
    # Here be the approximation of softmax

    OPRFs = f_ge_Q[:, :, np.newaxis, :] * f_ge_K[:, np.newaxis, :, :]
    if debug:
        mean_OPRFs = np.mean(OPRFs)
        var_OPRFs = np.var(OPRFs)
        print(
            f"For M = {M}, Mean of OPRFs: {mean_OPRFs}, Variance of OPRFs: {var_OPRFs}"
        )
    attention_weights = np.sum(OPRFs, axis=-1)
    attention_output = np.matmul(attention_weights, V)
    return attention_output


# def gpu_self_attention(Q, K, V, omega, M, debug=False):
#     # Just like the vectorized version, but using cupy instead of numpy
#     N, T, d = Q.shape  # Batch size, sequence length, feature dimension

#     # Calculate A for the whole batch
#     x_sample = cp.mean(Q, axis=(0, 1))
#     y_sample = cp.mean(K, axis=(0, 1))
#     A = calculate_A(x_sample, y_sample, d)

#     # Calculate B, C, D using A and s=+1
#     B = cp.sqrt((1 - 4 * A))
#     C = -(1 + 1) / 2
#     D = (cp.sqrt((1 - 4 * A)) ** (d / 2)).astype(cp.float32)

#     OPRFs = cp.zeros((N, T, T, M))

#     # Using broadcasting and np.einsum for vectorized computation
#     term1 = A * cp.sum(omega**2, axis=1)
#     term2_Q = cp.einsum("ijk,lk->ijl", Q, omega)
#     term2_K = cp.einsum("ijk,lk->ijl", K, omega)
#     term3_Q = C * cp.sum(Q**2, axis=2)[:, :, cp.newaxis]
#     term3_K = C * cp.sum(K**2, axis=2)[:, :, cp.newaxis]

#     f_ge_Q = D * cp.exp(term1 + term2_Q + term3_Q)
#     f_ge_K = D * cp.exp(term1 + term2_K + term3_K)
#     # Here be the approximation of softmax
#     OPRFs = f_ge_Q[:, :, cp.newaxis, :] * f_ge_K[:, cp.newaxis, :, :]
#     if debug:
#         mean_OPRFs = cp.mean(OPRFs)
#         var_OPRFs = cp.var(OPRFs)
#         print(
#             f"In cupy GPU function, for M = {M}, Mean of OPRFs: {mean_OPRFs}, Variance of OPRFs: {var_OPRFs}"
#         )
#     attention_weights = cp.sum(OPRFs, axis=-1)
#     attention_output = cp.matmul(attention_weights, V)
#     return attention_output


def batched_self_attention(Q, K, V, omega, M, debug=False):
    N, T, d = Q.shape  # Batch size, sequence length, feature dimension

    # Calculate A for the whole batch
    x_sample = np.mean(Q, axis=(0, 1))
    y_sample = np.mean(K, axis=(0, 1))
    A = calculate_A(x_sample, y_sample, d)

    # Calculate B, C, D using A and s=+1
    B = np.sqrt((1 - 4 * A))
    C = -(1 + 1) / 2
    D = np.sqrt((1 - 4 * A)) ** (d / 2)

    OPRFs = np.zeros((N, T, T, M))
    # Here be the approximation of softmax

    for i in range(N):
        for j in range(T):
            for k in range(T):
                for l in range(M):
                    x = Q[i, j]
                    y = K[i, k]
                    OPRFs[i, j, k, l] = f_GE(omega[l], x, A, B, C, D) * f_GE(
                        omega[l], y, A, B, C, D
                    )
    if debug:
        mean_OPRFs = np.mean(OPRFs)
        var_OPRFs = np.var(OPRFs)
        print(
            f"For M = {M}, Mean of OPRFs: {mean_OPRFs}, Variance of OPRFs: {var_OPRFs}"
        )
    attention_weights = np.sum(OPRFs, axis=-1)
    attention_output = np.matmul(attention_weights, V)

    return attention_output


def sub_batched_self_attention(sub_Q, sub_K, sub_V, omega, A, B, C, D, debug):
    # Compute terms just like in the vectorized version
    term1 = A * np.sum(omega**2, axis=1)
    term2_Q = np.einsum("ijk,lk->ijl", sub_Q, omega)
    term2_K = np.einsum("ijk,lk->ijl", sub_K, omega)
    term3_Q = C * np.sum(sub_Q**2, axis=2)[:, :, np.newaxis]
    term3_K = C * np.sum(sub_K**2, axis=2)[:, :, np.newaxis]

    f_ge_Q = D * np.exp(term1 + term2_Q + term3_Q)
    f_ge_K = D * np.exp(term1 + term2_K + term3_K)

    # Here be the approximation of softmax
    OPRFs = f_ge_Q[:, :, np.newaxis, :] * f_ge_K[:, np.newaxis, :, :]

    if debug:
        mean_OPRFs = np.mean(OPRFs)
        var_OPRFs = np.var(OPRFs)
        print(
            f"For parallelized batch, Mean of OPRFs: {mean_OPRFs}, Variance of OPRFs: {var_OPRFs}"
        )
    attention_weights = np.sum(OPRFs, axis=-1)
    attention_output = np.matmul(attention_weights, sub_V)
    return attention_output


def batched_self_attention_parallel(Q, K, V, omega, M, n_jobs=1, debug=False):
    N, T, d = Q.shape
    sub_batch_size = N // n_jobs
    # print(f"Sub-batch size: {sub_batch_size}")
    # print(f"Number of sub-batches: {n_jobs}")
    # print(f"Total batch size: {N}")
    # print(f"Total sequence length: {T}")
    # print(f"Feature dimension: {d}")
    # print(f"Number of random projections: {M}")
    # print(f"Debug mode: {debug}")
    # print("")

    # Calculate A, B, C, D for the whole batch
    x_sample = np.mean(Q, axis=(0, 1))
    y_sample = np.mean(K, axis=(0, 1))
    A = calculate_A(x_sample, y_sample, d)
    B = np.sqrt((1 - 4 * A))
    C = -(1 + 1) / 2
    D = np.sqrt((1 - 4 * A)) ** (d / 2)
    # Split the data into sub-batches and process them in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(sub_batched_self_attention)(
            Q[i * sub_batch_size : (i + 1) * sub_batch_size],
            K[i * sub_batch_size : (i + 1) * sub_batch_size],
            V[i * sub_batch_size : (i + 1) * sub_batch_size],
            omega,
            A,
            B,
            C,
            D,
            debug,
        )
        for i in range(n_jobs)
    )

    # Concatenate the results to get the final output
    return np.concatenate(results, axis=0)


def generate_block_orthogonal_matrix(d, M):
    # Initialize the block-orthogonal matrix
    W = np.zeros((M, d))

    # Calculate the number of full blocks
    num_full_blocks = M // d

    # Generate full orthogonal blocks
    for i in range(num_full_blocks):
        W[i * d : (i + 1) * d, :] = ortho_group.rvs(dim=d)

    # If there are remaining rows, fill them with a truncated orthogonal matrix
    remaining_rows = M % d
    if remaining_rows > 0:
        W[num_full_blocks * d :, :] = ortho_group.rvs(dim=d)[:remaining_rows, :]

    return W


# Update the self_attention function to use block-orthogonal matrices
def self_attention_block(Q, K, V, M, kwargs):
    N, T, d = Q.shape  # Batch size, sequence length, feature dimension

    # Generate block-orthogonal matrix
    omega = generate_block_orthogonal_matrix(d, M)

    # if kwargs.get("gpu"):
    #     Q = cp.array(Q)
    #     K = cp.array(K)
    #     V = cp.array(V)
    #     omega = cp.array(omega)
    #     return gpu_self_attention(Q, K, V, omega, M, debug=kwargs["debug"])
    if kwargs.get("vectorized"):
        return vectorized_self_attention(Q, K, V, omega, M, debug=kwargs["debug"])
    if kwargs.get("parallel"):
        return batched_self_attention_parallel(
            Q, K, V, omega, M, n_jobs=kwargs["n_jobs"], debug=kwargs["debug"]
        )

    return batched_self_attention(Q, K, V, omega, M, debug=False)


def test_samples():
    # Initialize some parameters
    d = 64  # Feature dimension
    M_values = [8, 128, 256]  # Number of random projections
    N = 32  # Batch size
    T = 10  # Sequence length

    # Initialize Q, K, V matrices
    Q = np.random.rand(N, T, d).astype(np.float32)
    K = np.random.rand(N, T, d).astype(np.float32)
    V = np.random.rand(N, T, d).astype(np.float32)
   
    # Initialize Q, K, V tensors so we can test the torch version
    Qtorch_gpu = torch.Tensor(Q).to("cuda")
    Ktorch_gpu = torch.Tensor(K).to("cuda")
    Vtorch_gpu = torch.Tensor(V).to("cuda")
    Qtorch_cpu = torch.Tensor(Q).to("cpu")
    Ktorch_cpu = torch.Tensor(K).to("cpu")
    Vtorch_cpu = torch.Tensor(V).to("cpu")
    omega = generate_block_orthogonal_matrix(d, max(M_values))
    omega_torch_gpu = torch.Tensor(omega).to("cuda")
    omega_torch_cpu = torch.Tensor(omega).to("cpu")
    # Initialize omega for each M value
    results = {}
    for M in M_values:
        print(f"Testing M={M} next... \n")
        omega_this_M = omega[:M]
        omega_torch_cpu_this_M = omega_torch_cpu[:M]
        omega_torch_gpu_this_M = omega_torch_gpu[:M]
        
        attention_output = self_attention(
            Q, K, V, omega_this_M, M, kwargs={"debug": True, "vectorized": True}
        )
        results[M] = attention_output.shape
        print(f"Attention output shape for M={M}: {attention_output.shape}")
        print("Testing parallelization next... \n")
        attention_output = self_attention(
            Q, K, V, omega_this_M, M, kwargs={"parallel": True, "n_jobs": 2, "debug": True}
        )
        print(
            f"Attention output shape for parallelized M={M}: {attention_output.shape}"
        )

        print("Testing gpu next... \n")
        attention_output = self_attention(
            Q, K, V, omega_this_M, M, kwargs={"gpu": True, "debug": True}
        )
        print(f"Attention output shape for gpu M={M}: {attention_output.shape}")

        print("Testing torch version without GPU next... \n")
        attention_output = self_attention(
            Qtorch_cpu,
            Ktorch_cpu,
            Vtorch_cpu,
            omega_torch_cpu_this_M,
            M,
            kwargs={"debug": True, "torch": True},
        )
        print(
            f"Attention output shape for torch without GPU for M={M}: {attention_output.shape}"
        )

        print("Testing torch version with GPU next... \n")
        attention_output = self_attention(
            Qtorch_gpu,
            Ktorch_gpu,
            Vtorch_gpu,
            omega_torch_gpu_this_M,
            M,
            kwargs={"torch": True, "debug": True, "gpu": True},
        )
        print(
            f"Attention output shape for torch with GPU for M={M}: {attention_output.shape}"
        )
        # convert omega to a torch tensor and send to gpu
        omega_torch_gpu = torch.tensor(omega).to("cuda", dtype=torch.float32)
        print(f"omega shape and size: {omega.shape}, {omega.size}")
        print(f"Omega torch gpu data type: {omega_torch_gpu.dtype}")
        print("Testing triton version next... \n")
        attention_output = self_attention(
            Qtorch_gpu,
            Ktorch_gpu,
            Vtorch_gpu,
            omega_torch_gpu_this_M,
            M,
            kwargs={"triton": True},
        )
        print(f"Attention output shape for triton for M={M}: {attention_output.shape}")

        num_samples = 10000  # Number of samples to estimate variance
        d = 64  # Feature dimension

        # Generate random vectors x and y
        x = np.random.rand(d)
        y = np.random.rand(d)

        # Generate random omega for M random projections
        omega = np.random.randn(M, d)

        # Calculate A, B, C, D using a sample x and y

        A = calculate_A(x, y, d)

        # Calculate B, C, D using A and s=+1
        B = np.sqrt((1 - 4 * A))
        C = -(1 + 1) / 2
        D = np.sqrt((1 - 4 * A)) ** (d / 2)
        OPRF_samples = np.array(
            [
                f_GE(omega[i], x, A, B, C, D) * f_GE(omega[i], y, A, B, C, D)
                for i in range(M)
                for _ in range(num_samples // M)
            ]
        )

        mean_OPRFs = np.mean(OPRF_samples)
        print(f"Mean OPRFs for M={M}: {mean_OPRFs}")
        var_OPRFs = np.var(OPRF_samples)
        print(f"Variance of OPRFs for M={M}: {var_OPRFs}")
        print("")

    # 2. Run some sample queries and keys through the attention mechanism
    M = 8  # Use the smallest M for quick testing
    omega = np.random.randn(M, d)
    attention_output = self_attention(
        Q, K, V, omega, M, kwargs={"debug": False, "vectorized": True}
    )
    print("\nSample Attention Output:", attention_output[0, 0, :])

    # 3. Benchmark the computational time for different values of M
    for M in M_values:
        start_time = time.time()
        attention_output = self_attention(
            Q, K, V, omega, M, kwargs={"debug": False, "vectorized": True}
        )
        end_time = time.time()
        print(f"Time for M={M}: {end_time - start_time}")
    print("Testing timing for parallelized functions next... \n")

    for M in M_values:
        start_time = time.time()
        attention_output = self_attention(
            Q, K, V, omega, M, kwargs={"parallel": True, "n_jobs": 2, "debug": False}
        )
        end_time = time.time()
        print(f"Time for parallelized M={M}: {end_time - start_time}")

    print("Testing timing for gpu functions next... \n")

    for M in M_values:
        start_time = time.time()
        attention_output = self_attention(
            Q, K, V, omega, M, kwargs={"gpu": True, "debug": False}
        )
        end_time = time.time()
        print(f"Time for gpu M={M}: {end_time - start_time}")

    print("Testing timing for torch functions without GPU next... \n")

    for M in M_values:
        start_time = time.time()
        attention_output = self_attention(
            Qtorch_cpu,
            Ktorch_cpu,
            Vtorch_cpu,
            omega_torch_cpu,
            M,
            kwargs={"debug": False, "torch": True},
        )
        end_time = time.time()
        print(f"Time for torch no gpu with M={M}: {end_time - start_time}")
    print("Testing timing for torch functions with GPU next... \n")

    for M in M_values:
        start_time = time.time()
        attention_output = self_attention(
            Qtorch_gpu,
            Ktorch_gpu,
            Vtorch_gpu,
            omega_torch_gpu,
            M,
            kwargs={"debug": False, "torch": True, "gpu": True},
        )
        end_time = time.time()
        print(f"Time for torch + gpu with M={M}: {end_time - start_time}")

    print("Testing timing for triton functions next... \n")

    for M in M_values:
        start_time = time.time()
        attention_output = self_attention(
            Qtriton, Ktriton, Vtriton, omega, M, kwargs={"triton": True}
        )
        end_time = time.time()
        print(f"Time for triton with M={M}: {end_time - start_time}")

    print(results)
    results


# Function to calculate the Gaussian kernel between two vectors x and y
def gaussian_kernel(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma**2))


# Function to calculate the variance of an estimator
def compute_variance(estimator_values):
    return np.var(estimator_values)


# Function to test the updated self_attention function
def test_samples_updated():
    # Initialize some parameters
    d = 64  # Feature dimension
    M_values = [8, 64, 65, 128, 256]  # Number of random projections
    N = 32  # Batch size
    T = 10  # Sequence length

    # Initialize Q, K, V matrices
    Q = np.random.rand(N, T, d).astype(np.float32)
    K = np.random.rand(N, T, d).astype(np.float32)
    V = np.random.rand(N, T, d).astype(np.float32)

    # Test for different M values
    results = {}
    for M in M_values:
        attention_output = self_attention_block(
            Q, K, V, M, kwargs={"debug": True, "vectorized": True}
        )
        results[M] = attention_output.shape
        print(f"Attention output shape for M={M}: {attention_output.shape}")
    for M in M_values:
        start_time = time.time()
        attention_output = self_attention_block(
            Q, K, V, M, kwargs={"vectorized": True, "debug": False}
        )
        end_time = time.time()
        print(
            f"Time using block-orthogonal projections: for M={M}: {end_time - start_time}"
        )

    print(results)


if __name__ == "__main__":
    test_samples()
    print("Testing updated self_attention function next... \n")

    test_samples_updated()
