import subprocess
subprocess.run(["nvidia-smi"])

import torch
x = torch.randn(4096, 4096, device="cuda")
y = x @ x.T
print(y.shape)

import torch
print(torch.version.cuda)                 # e.g., '12.1'
print(torch.__version__)                  # e.g., '2.4.1+cu121'
print(torch.cuda.get_device_capability()) # e.g., (8, 9) => sm_89

