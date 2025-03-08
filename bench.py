import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Load the CUDA kernel as a python module
minimal_attn = load(name='minimal_attn', sources=['main.cpp', 'flash_v2.cu'], extra_cuda_cflags=['-O2'])

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 16
n_q_head = 16
n_kv_head = 8
seq_len = 64
head_embd = 64

torch.manual_seed(45)
q = torch.randn(batch_size, n_q_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_kv_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_kv_head, seq_len, head_embd).cuda()
mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0).bool().cuda()
print('=== profiling manual attention ===')

def causal_softmax(x):
    mask = torch.tril(torch.ones_like(x), diagonal=-1).flip(dims=[-2, -1])
    y = x.clone()
    masked = torch.where(mask == 1, -torch.inf, y) 
    return F.softmax(masked, dim=-1)

# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_attn(q, k, v):
    att = (q.reshape(batch_size, n_kv_head, -1, head_embd) @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = causal_softmax(att.reshape(batch_size, n_q_head, seq_len, seq_len)).reshape(batch_size, n_kv_head, -1, seq_len)
    y = att @ v
    return y.reshape(batch_size, n_q_head, -1, head_embd)

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = manual_attn(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling minimal flash attention === ')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_result = minimal_attn.forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
print('attn values sanity check:', torch.allclose(minimal_result, manual_result, rtol=0, atol=1e-02))
