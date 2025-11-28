import torch
import time
from fla.layers.rwkv6 import RWKV6Attention

# Initialize layer
batch_size, num_heads, seq_len, hidden_size = 32, 4, 2048, 128
# check triton minimum size of 64
device, dtype = 'cuda:0', torch.bfloat16

retnet = RWKV6Attention(
    hidden_size=hidden_size, 
    num_heads=num_heads
).to(device=device, dtype=dtype)

# Forward pass
x = torch.randn(batch_size, seq_len, hidden_size).to(device=device, dtype=dtype)
times = []
for i in range(1000):
    start = time.time()
    y, *_ = retnet(x)
    torch.cuda.synchronize()
    end = time.time()
    times.append(end - start)
print(torch.mean(torch.tensor(times[100:])))
# y.shape: torch.Size([32, 2048, 1024])