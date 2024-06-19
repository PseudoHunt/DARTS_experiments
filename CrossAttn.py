
import torch
import torch.nn as nn

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, query_dim, key_value_dim, embed_size, heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.queries = nn.Linear(query_dim, embed_size, bias=False)
        self.keys = nn.Linear(key_value_dim, embed_size, bias=False)
        self.values = nn.Linear(key_value_dim, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, queries, keys, values, mask):
        N = queries.shape[0]
        query_len = queries.shape[1]
        key_len = keys.shape[1]

        # Linear projections
        queries_proj = self.queries(queries)
        keys_proj = self.keys(keys)
        values_proj = self.values(values)

        # Reshape into multiple heads
        queries_proj = queries_proj.reshape(N, query_len, self.heads, self.head_dim)
        keys_proj = keys_proj.reshape(1, key_len, self.heads, self.head_dim).repeat(N, 1, 1, 1)
        values_proj = values_proj.reshape(1, key_len, self.heads, self.head_dim).repeat(N, 1, 1, 1)

        # Permute to bring heads dimension in front
        queries_proj = queries_proj.permute(0, 2, 1, 3)  # Shape: [N, heads, query_len, head_dim]
        keys_proj = keys_proj.permute(0, 2, 1, 3)        # Shape: [N, heads, key_len, head_dim]
        values_proj = values_proj.permute(0, 2, 1, 3)    # Shape: [N, heads, key_len, head_dim]

        # Step 1: Reshape queries and keys for batched matrix multiplication
        queries_proj = queries_proj.reshape(N * self.heads, query_len, self.head_dim)
        keys_proj = keys_proj.reshape(N * self.heads, key_len, self.head_dim)

        # Step 2: Matrix multiplication (batch matmul)
        energy = torch.matmul(queries_proj, keys_proj.transpose(-1, -2))  # Shape: [N * self.heads, query_len, key_len]

        # Step 3: Reshape back to original shape
        energy = energy.reshape(N, self.heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)

        # Apply attention weights to values
        values_proj = values_proj.reshape(N * self.heads, key_len, self.head_dim)
        out = torch.matmul(attention.reshape(N * self.heads, query_len, key_len), values_proj)

        # Reshape to (N, query_len, heads, head_dim) and combine heads
        out = out.reshape(N, self.heads, query_len, self.head_dim).permute(0, 2, 1, 3).reshape(N, query_len, self.embed_size)

        out = self.fc_out(out)
        return out

# Example usage
query_dim = 320
key_value_dim = 768
embed_size = 320
heads = 8
queries = torch.rand((64, 64, query_dim))
keys = torch.rand((1, 50, key_value_dim))
values = torch.rand((1, 50, key_value_dim))
mask = None

cross_attention_layer = MultiHeadCrossAttention(query_dim, key_value_dim, embed_size, heads)
out = cross_attention_layer(queries, keys, values, mask)
print(out.shape)  # Should print torch.Size([64, 64, 320])

import torch
import torch.nn as nn

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, query_dim, key_value_dim, embed_size, heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.queries = nn.Linear(query_dim, embed_size, bias=False)
        self.keys = nn.Linear(key_value_dim, embed_size, bias=False)
        self.values = nn.Linear(key_value_dim, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def preprocess(self,keys,queries,values):
        #iNIT
        N = queries.shape[0]
        query_len = queries.shape[1]
        print(self.keys.weight.shape)

        key_len = keys.shape[1]
        keys_proj = self.keys(keys)
        print(keys_proj.shape) # 1,50,320
        keys_proj = keys_proj.reshape(1, key_len, self.heads, self.head_dim).repeat(1, 1, 1, 1)
        keys_proj = keys_proj.permute(0, 2, 3, 1)  # Shape: [N, heads, head_dim,key_len]
        print(keys_proj.shape) # 1,8,40,50
        Wq = self.queries.weight
        #Wq = Wq.reshape(1, self.embed_size, self.heads, self.head_dim)
        Wq = Wq.reshape(1, self.heads, self.head_dim,self.embed_size)
        print(Wq.shape) # 320,8,40
        Wq = Wq.permute(0, 1, 3, 2)  # Shape: [N, heads, key_len, head_dim]
        print(Wq.shape) # 1,8,320,40
        self.qk = torch.matmul(Wq, keys_proj)
        print(self.qk.shape)
        #self.qk = self.queries(keys_proj)
        #self.qk = self.qk.reshape(1, key_len, self.heads, self.head_dim).repeat(1, 1, 1, 1)
        #print(self.qk.shape)

        #computation
        energy = torch.matmul(queries,self.qk)  # Shape: [N * self.heads, query_len, key_len]
        print(energy.shape)

        # Step 3: Reshape back to original shape
        energy = energy.reshape(N, self.heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)
        #until here it works

        #Fuse V and out
        values_proj = self.values(values)
        print(values_proj.shape) # 1,50,320
        values_proj = values_proj.reshape(1, key_len, self.heads, self.head_dim)
        print(values_proj.shape) # 1,50,8,40
        values_proj = values_proj.permute(0, 2, 1, 3)  # Shape: [N, heads, key_len, head_dim]
        print(values_proj.shape) # 1,8,50,40
        W_out = self.fc_out.weight.T
        #method1
        W_out = W_out.reshape(1, self.heads, self.head_dim,self.embed_size)
        print(W_out.shape) # 1,8,40,320
        #method2
        #W_out = W_out.reshape(1, self.embed_size, self.heads, self.head_dim)
        #print(W_out.shape) # 1,320,8,40
        #W_out = W_out.permute(0, 2, 3, 1)  # Shape: [N, heads, head_dim,key_len]
        #print(W_out.shape) # 1,8,40,320
        #multiply values_proj and W_out
        out = torch.matmul(values_proj,W_out)
        print(out.shape)
        print('check above')

        #compute
        out = torch.matmul(attention,out)
        print(out.shape)
        out = torch.sum(out, dim=1)
        out += self.fc_out.bias
        print(out.shape)
        return out


    def precompute(self):
        #iNIT
        N = queries.shape[0]
        query_len = queries.shape[1]
        print(self.keys.weight.shape)

        #Fuse Q and K
        key_len = keys.shape[1]
        keys_proj = self.keys(keys)
        keys_proj = keys_proj.reshape(1, key_len, self.heads, self.head_dim).repeat(1, 1, 1, 1)
        keys_proj = keys_proj.permute(0, 2, 3, 1)  # Shape: [N, heads, head_dim,key_len]
        Wq = self.queries.weight
        Wq = Wq.reshape(1, self.heads, self.head_dim,self.embed_size)
        Wq = Wq.permute(0, 1, 3, 2)  # Shape: [N, heads, key_len, head_dim]
        self.qk = torch.matmul(Wq, keys_proj)

        #Fuse V and out
        values_proj = self.values(values)
        values_proj = values_proj.reshape(1, key_len, self.heads, self.head_dim)
        values_proj = values_proj.permute(0, 2, 1, 3)  # Shape: [N, heads, key_len, head_dim]
        W_out = self.fc_out.weight.T
        W_out = W_out.reshape(1, self.heads, self.head_dim,self.embed_size)
        self.Vout = torch.matmul(values_proj,W_out)

    def fused_fwd(self,queries):
        #init
        N = queries.shape[0]
        query_len = queries.shape[1]
        key_len = keys.shape[1]

        #computation qk
        energy = torch.matmul(queries,self.qk)  # Shape: [N * self.heads, query_len, key_len]
        energy = energy.reshape(N, self.heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        #compute attention mask
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)

        #compute out
        out = torch.matmul(attention,self.Vout)
        out = torch.sum(out, dim=1)
        out += self.fc_out.bias

        return out


    def forward(self, queries, keys, values, mask):
        N = queries.shape[0]
        query_len = queries.shape[1]
        key_len = keys.shape[1]

        # Linear projections
        queries_proj = self.queries(queries)
        keys_proj = self.keys(keys)
        values_proj = self.values(values)

        # Reshape into multiple heads
        queries_proj = queries_proj.reshape(N, query_len, self.heads, self.head_dim)
        keys_proj = keys_proj.reshape(1, key_len, self.heads, self.head_dim).repeat(N, 1, 1, 1)
        values_proj = values_proj.reshape(1, key_len, self.heads, self.head_dim).repeat(N, 1, 1, 1)

        # Permute to bring heads dimension in front
        queries_proj = queries_proj.permute(0, 2, 1, 3)  # Shape: [N, heads, query_len, head_dim]
        keys_proj = keys_proj.permute(0, 2, 1, 3)        # Shape: [N, heads, key_len, head_dim]
        values_proj = values_proj.permute(0, 2, 1, 3)    # Shape: [N, heads, key_len, head_dim]

        # Step 1: Reshape queries and keys for batched matrix multiplication
        queries_proj = queries_proj.reshape(N * self.heads, query_len, self.head_dim)
        keys_proj = keys_proj.reshape(N * self.heads, key_len, self.head_dim)

        # Step 2: Matrix multiplication (batch matmul)
        energy = torch.matmul(queries_proj, keys_proj.transpose(-1, -2))  # Shape: [N * self.heads, query_len, key_len]

        # Step 3: Reshape back to original shape
        energy = energy.reshape(N, self.heads, query_len, key_len)
        #return energy
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)
        #return attention
        # Apply attention weights to values
        values_proj = values_proj.reshape(N * self.heads, key_len, self.head_dim)
        out = torch.matmul(attention.reshape(N * self.heads, query_len, key_len), values_proj)

        # Reshape to (N, query_len, heads, head_dim) and combine heads
        out = out.reshape(N, self.heads, query_len, self.head_dim).permute(0, 2, 1, 3).reshape(N, query_len, self.embed_size)

        out = self.fc_out(out)
        return out

# Example usage
query_dim = 320
key_value_dim = 768
embed_size = 320
heads = 8
queries = torch.rand((1, 4096, query_dim))
keys = torch.rand((1, 50, key_value_dim))
values = keys
#values = torch.rand((1, 50, key_value_dim))
mask = None


cross_attention_layer = MultiHeadCrossAttention(query_dim, key_value_dim, embed_size, heads)
test = cross_attention_layer.preprocess(keys,queries,values)
out = cross_attention_layer(queries, keys, values, mask)
print(out.shape)  # Should print torch.Size([64, 64, 320])

cross_attention_layer.precompute()
out = cross_attention_layer.fused_fwd(queries)
print(out.shape)  # Should print torch.Size([64, 64, 320])

# prompt: calculate the mse between test and out

import torch.nn as nn

mse_loss = nn.MSELoss()
mse = mse_loss(test, out)
print(mse)

test

out

print(cross_attention_layer.fc_out.bias.shape)

# prompt: how to add bias value to out

# Add bias to the output
out = cross_attention_layer.fc_out(out) + cross_attention_layer.fc_out.bias
