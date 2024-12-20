import numpy as np

def tt_svd(matrix, max_rank=None, epsilon=1e-5):
    """
    Decomposes a 2D matrix into Tensor Train (TT) cores using the TT-SVD algorithm.
    Args:
        matrix (numpy.ndarray): The input matrix.
        max_rank (int): The maximum allowable rank for TT decomposition.
        epsilon (float): The tolerance for truncating singular values.

    Returns:
        tt_cores (list): List of TT cores.
    """
    rows, cols = matrix.shape
    a, b, c = 12, 18, 216  # Example split: (12 * 18 = 216)
    tensor = matrix.reshape((a, b, c))
    tt_cores = []
    ranks = [1]

    for mode in range(3):  # Number of modes in TT
        # Reshape the tensor for SVD
        unfolding = tensor.reshape(ranks[-1] * tensor.shape[mode], -1)
        U, S, Vt = np.linalg.svd(unfolding, full_matrices=False)

        # Truncate based on rank and epsilon
        if max_rank:
            rank = min(max_rank, len(S))
        else:
            rank = np.sum(S > epsilon)
        
        ranks.append(rank)

        # TT core
        core = U[:, :rank].reshape(ranks[-2], tensor.shape[mode], rank)
        tt_cores.append(core)

        # Update tensor for next iteration
        tensor = (np.diag(S[:rank]) @ Vt[:rank, :]).reshape(rank, *tensor.shape[mode + 1:])

    # Add the last core
    tt_cores.append(tensor)

    return tt_cores


def tt_multiply(tt_cores):
    """
    Multiplies TT cores to reconstruct the original tensor.
    Args:
        tt_cores (list): List of TT cores.

    Returns:
        matrix (numpy.ndarray): Reconstructed matrix.
    """
    core_1 = tt_cores[0]
    product = core_1

    for core in tt_cores[1:]:
        product = np.tensordot(product, core, axes=[-1, 0])

    return product.reshape(-1, product.shape[-1])


# Example random matrix
matrix = np.random.rand(216, 216)

# TT decomposition
tt_cores = tt_svd(matrix, max_rank=50)

# Reconstruct matrix from TT cores
reconstructed_matrix = tt_multiply(tt_cores)

# Compare accuracy with random tensor multiplication
difference = np.linalg.norm(matrix - reconstructed_matrix)
relative_error = difference / np.linalg.norm(matrix)

print("Reconstruction Error (Frobenius Norm):", difference)
print("Relative Error:", relative_error)
