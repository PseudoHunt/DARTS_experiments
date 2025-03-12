import torch
import torch.nn as nn
import torch.nn.functional as F

class SUPRAParallelFixedGamma(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = 1e-6
        self.gamma = nn.Parameter(torch.tensor(0.9))  # Learnable scalar decay

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def phi(self, x):
        return F.relu(x)  # Kernel function (could be replaced with learned MLP or others)

    def forward(self, x_seq):
        B, T, D = x_seq.shape

        # Linear projections
        q = self.phi(self.W_q(x_seq))  # [B, T, D]
        k = self.phi(self.W_k(x_seq))  # [B, T, D]
        v = self.W_v(x_seq)            # [B, T, D]

        # Outer product for k and v: [B, T, D, D]
        kv = torch.einsum('bti,btj->btij', k, v)

        # Prepare decay weights: gamma^(t - i) for each t and i
        gamma_exponents = torch.arange(T, device=x_seq.device).unsqueeze(0) - torch.arange(T, device=x_seq.device).unsqueeze(1)
        gamma_exponents = gamma_exponents.clamp(min=0).float()  # Ensures negative exponents are zeroed out
        gamma_powers = self.gamma ** gamma_exponents  # [T, T] matrix

        # Apply decay to k and kv
        decayed_k = torch.einsum('ij,bjd->bid', gamma_powers, k)  # [B, T, D]
        decayed_kv = torch.einsum('ij,bjdj->bidj', gamma_powers, kv)  # [B, T, D, D]

        # Cumulative sums (via matmul over decayed versions)
        S = decayed_kv.cumsum(dim=1)  # [B, T, D, D]
        Z = decayed_k.cumsum(dim=1)   # [B, T, D]

        # Compute output
        num = torch.einsum('bti,btij->btj', q, S)  # [B, T, D]
        denom = torch.einsum('bti,bti->bt', q, Z).unsqueeze(-1) + self.eps  # [B, T, 1]
        y = num / denom  # [B, T, D]

        return y

import torch
import torch.nn as nn
import torch.nn.functional as F

class SUPRARecurrentFixedGamma(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = 1e-6
        self.gamma = nn.Parameter(torch.tensor(0.9))  # Learnable scalar decay

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def phi(self, x):
        return F.relu(x)

    def forward(self, x_seq):
        batch_size, seq_len, embed_dim = x_seq.shape
        s = torch.zeros(batch_size, embed_dim, embed_dim, device=x_seq.device)
        z = torch.zeros(batch_size, embed_dim, device=x_seq.device)
        outputs = []

        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            q_t = self.phi(self.W_q(x_t))
            k_t = self.phi(self.W_k(x_t))
            v_t = self.W_v(x_t)

            # Update states with scalar gamma
            s = self.gamma * s + torch.einsum('bi,bj->bij', k_t, v_t)
            z = self.gamma * z + k_t

            # Compute attention output
            num = torch.einsum('bi,bij->bj', q_t, s)
            denom = torch.einsum('bi,bi->b', q_t, z).unsqueeze(-1) + self.eps
            y_t = num / denom
            outputs.append(y_t.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # [B, T, D]

def test_supra_equivalence_fixed_gamma():
    embed_dim = 16
    seq_len = 8
    batch_size = 2

    # Initialize models and sync weights
    recurrent_model = SUPRARecurrentFixedGamma(embed_dim)
    parallel_model = SUPRAParallelFixedGamma(embed_dim)
    parallel_model.load_state_dict(recurrent_model.state_dict())

    # Random input
    x_seq = torch.randn(batch_size, seq_len, embed_dim)

    # Set models to evaluation mode
    recurrent_model.eval()
    parallel_model.eval()

    # Forward pass
    with torch.no_grad():
        output_recurrent = recurrent_model(x_seq)
        output_parallel = parallel_model(x_seq)

    # Compare outputs
    difference = torch.abs(output_recurrent - output_parallel).max()
    print("Max difference between recurrent and parallel outputs:", difference.item())

    # Assertion to validate correctness
    assert difference < 1e-4, "Outputs from recurrent and parallel models do not match!"

    print("Test Passed! Recurrent and Parallel outputs match within tolerance.")


# Run test case
if __name__ == "__main__":
    test_supra_equivalence_fixed_gamma()
