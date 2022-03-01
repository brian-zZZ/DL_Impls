'''Define the Scaled Dot-Product Attention for multi-head Attention.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # Decoder的Masked Multi-Head Attention中的Mask(optional).
        # 对于时间t，Decoder输出只能获得1,...,t-1的output-embed，>=t的信息要被Mask掉
        # 将>=t的attn设为很小的值(如-e9)，那么经过Softmax后就映射成0，attn得分为0
        if mask is not None:
            # torch.Tensor.masked_fill(mask, value)将Tensor中mask矩阵值为1的位置的值填成value
            # Tensor和mask的shape要一致，一一对应
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1)) # attn weights
        output = torch.matmul(attn, v) # attn scores

        return output, attn
