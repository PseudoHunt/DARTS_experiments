import tensorly as tl
from tensorly.decomposition import tucker

# Initialize
ranks = [r1, r2, ..., rN]  # Initial low ranks
core, factors = tucker(original_tensor, ranks)

# Iteratively sparsify residual
for _ in range(max_iter):
    residual = original_tensor - tl.tucker_to_tensor(core, factors)
    residual = threshold(residual, tau)  # Hard/soft thresholding
    core, factors = tucker(original_tensor - residual, ranks)import tensorly as tl

def threshold(residual_dense, tau):
  residual_dense[residual_dense > tau] = 0
