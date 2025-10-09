import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5
        self.attn_dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        batch_size = x1.size(0)

        # 1. 线性变换并重新调整维度： (batch_size, seq_len, num_heads, head_dim)
        Q = self.W_q(x1).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len_1, head_dim)
        K = self.W_k(x2).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len_2, head_dim)
        V = self.W_v(x2).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len_2, head_dim)

        # 2. 调整维度以进行批量矩阵乘法 (batch_size * num_heads, seq_len, head_dim)
        Q = Q.contiguous().view(batch_size * self.num_heads, -1, self.head_dim)  # (batch_size * num_heads, seq_len_1, head_dim)
        K = K.contiguous().view(batch_size * self.num_heads, -1, self.head_dim)  # (batch_size * num_heads, seq_len_2, head_dim)
        V = V.contiguous().view(batch_size * self.num_heads, -1, self.head_dim)  # (batch_size * num_heads, seq_len_2, head_dim)

        # 3. 计算 Attention 权重： (batch_size * num_heads, seq_len_1, seq_len_2)
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (batch_size * num_heads, seq_len_1, seq_len_2)

        # 4. 对 K 维度应用 softmax：计算注意力分布
        attn = torch.softmax(attn, dim=-1)  # softmax 应用在 seq_len_2 上

        # 5. 应用 Dropout
        attn = self.attn_dropout(attn)

        # 6. 计算注意力输出： (batch_size * num_heads, seq_len_1, head_dim)
        output = torch.bmm(attn, V)  # (batch_size * num_heads, seq_len_1, head_dim)

        # 7. 恢复 batch_size 和 num_heads 维度： (batch_size, seq_len_1, dim)
        output = output.view(batch_size, self.num_heads, -1, self.head_dim)  # (batch_size, num_heads, seq_len_1, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)  # (batch_size, seq_len_1, dim)

        # 8. 最终线性映射
        output = self.out_proj(output)  # (batch_size, seq_len_1, dim)
        
        return output