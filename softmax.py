import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, secondary_mlp):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.secondary_mlp = secondary_mlp

    def forward(self, Q, K, V):
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        adjusted_scores = self.secondary_mlp(attention_scores)
        attention_weights = torch.softmax(adjusted_scores, dim=-1) #this softmax is not the original softmax but some approximation of the softmax 
        output = torch.matmul(attention_weights, V)
        return output

class SecondaryMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SecondaryMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
       
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage
d_k = 64
secondary_mlp = SecondaryMLP(input_size=d_k, hidden_size=128, output_size=d_k)
attention = ScaledDotProductAttention(d_k, secondary_mlp)

Q = torch.randn(10, 20, d_k)  # (batch_size, sequence_length, d_k)
K = torch.randn(10, 20, d_k)
V = torch.randn(10, 20, d_k)

output = attention(Q, K, V)
print(output.shape)  # Should be (batch_size, sequence_length, d_k)
