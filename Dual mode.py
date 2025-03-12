class SUPRAParallelGatedTrue(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = 1e-6

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_a = nn.Linear(embed_dim, embed_dim)  # Adaptive forget gate

    def phi(self, x):
        return F.relu(x)

    def forward(self, x_seq):
        B, T, D = x_seq.shape
        q = self.phi(self.W_q(x_seq))  # [B, T, D]
        k = self.phi(self.W_k(x_seq))  # [B, T, D]
        v = self.W_v(x_seq)            # [B, T, D]
        A = torch.sigmoid(self.W_a(x_seq))  # [B, T, D]

        kv = torch.einsum('bti,btj->btij', k, v)  # [B, T, D, D]

        # Compute cumulative product of gates for decay factors
        log_A = torch.log(A + 1e-6)  # Stability
        log_cumprod_A = torch.cumsum(log_A.flip(dims=[1]), dim=1).flip(dims=[1])  # [B, T, D]
        cumprod_A = torch.exp(log_cumprod_A)

        # Multiply each kv and k with corresponding cumulative product
        weighted_kv = kv * cumprod_A.unsqueeze(-1)  # [B, T, D, D]
        weighted_k = k * cumprod_A  # [B, T, D]

        # Cumulative sums
        S = torch.cumsum(weighted_kv, dim=1)  # [B, T, D, D]
        Z = torch.cumsum(weighted_k, dim=1)   # [B, T, D]

        # Final output
        num = torch.einsum('bti,btij->btj', q, S)
        denom = torch.einsum('bti,bti->bt', q, Z).unsqueeze(-1) + self.eps
        y = num / denom  # [B, T, D]
        return y

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Recurrent SUPRA with Forget Gate
# -----------------------------

class SUPRARecurrentGated(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = 1e-6

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_a = nn.Linear(embed_dim, embed_dim)  # Adaptive forget gate

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
            A_t = torch.sigmoid(self.W_a(x_t))  # Adaptive gate

            s = A_t.unsqueeze(-1) * s + torch.einsum('bi,bj->bij', k_t, v_t)
            z = A_t * z + k_t

            num = torch.einsum('bi,bij->bj', q_t, s)
            denom = torch.einsum('bi,bi->b', q_t, z).unsqueeze(-1) + self.eps
            y_t = num / denom
            outputs.append(y_t.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # [B, T, D]


# -----------------------------
# Parallel SUPRA with Forget Gate
# -----------------------------

class SUPRAParallelGated(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = 1e-6

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_a = nn.Linear(embed_dim, embed_dim)  # Adaptive forget gate

    def phi(self, x):
        return F.relu(x)

    def forward(self, x_seq):
        B, T, D = x_seq.shape
        q = self.phi(self.W_q(x_seq))  # [B, T, D]
        k = self.phi(self.W_k(x_seq))  # [B, T, D]
        v = self.W_v(x_seq)            # [B, T, D]
        A = torch.sigmoid(self.W_a(x_seq))  # [B, T, D]

        # Initialize s and z
        s = torch.zeros(B, D, D, device=x_seq.device)
        z = torch.zeros(B, D, device=x_seq.device)
        S_list, Z_list = [], []

        for t in range(T):
            a_t = A[:, t, :].unsqueeze(-1)  # [B, D, 1]
            k_t = k[:, t, :]
            v_t = v[:, t, :]
            s = a_t * s + torch.einsum('bi,bj->bij', k_t, v_t)
            z = a_t.squeeze(-1) * z + k_t
            S_list.append(s.unsqueeze(1))
            Z_list.append(z.unsqueeze(1))

        S = torch.cat(S_list, dim=1)  # [B, T, D, D]
        Z = torch.cat(Z_list, dim=1)  # [B, T, D]

        num = torch.einsum('bti,btij->btj', q, S)
        denom = torch.einsum('bti,bti->bt', q, Z).unsqueeze(-1) + self.eps
        y = num / denom  # [B, T, D]
        return y


# -----------------------------
# Test Case to Compare Outputs
# -----------------------------

def test_supra_equivalence():
    embed_dim = 16
    seq_len = 8
    batch_size = 2

    # Initialize models and sync weights
    recurrent_model = SUPRARecurrentGated(embed_dim)
    parallel_model = SUPRAParallelGated(embed_dim)
    parallel_model.load_state_dict(recurrent_model.state_dict())

    # Random input
    x_seq = torch.randn(batch_size, seq_len, embed_dim)

    # Set models to eval mode
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


# -----------------------------
# Run Test
# -----------------------------

if __name__ == "__main__":
    test_supra_equivalence()
