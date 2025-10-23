import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries #@ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
    
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Compute attention scores: queries @ keys.T
        # queries: [batch, seq_len, d_out], keys: [batch, seq_len, d_out]
        # Result should be [batch, seq_len, seq_len]
        attn_scores = queries @ keys.transpose(1, 2)  # [batch, seq_len, seq_len]

        # Create causal mask
        context_length = attn_scores.shape[1]  # sequence length
        mask_simple = torch.tril(torch.ones(context_length, context_length))
        print(f"mask_simple is \n{mask_simple}")

        # Apply scaled dot-product and softmax
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # Apply causal mask
        masked_simple = attn_weights * mask_simple
        print("mask_simple is")
        print(mask_simple)
        print("==========")
        print("attn_weights is")
        print(attn_weights)
        print("==========")

        # Compute context vectors: attn_weights @ values
        context_vec = masked_simple @ values

        return context_vec