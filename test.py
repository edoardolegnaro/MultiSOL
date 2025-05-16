import numpy as np
import tensorflow as tf
import timeit
import torch

from multisol.tf.multisol import SOL as SOL_tf
from multisol.torch.multisol import SOL as SOL_torch

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
torch.manual_seed(seed)

# Define dimensions
B = 32     # Batch size
m = 3      # Number of classes
N = 1000   # Number of tau samples
lam = 10.0

# Generate tau samples
tau_samples_np = np.random.rand(N, m).astype(np.float32)

# Create sample predictions and one-hot true labels.
# For y_pred, generate random logits then apply softmax.
logits_np = np.random.rand(B, m).astype(np.float32)
exp_logits = np.exp(logits_np)
y_pred_np = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Create random one-hot labels.
y_true_np = np.zeros((B, m), dtype=np.float32)
for i in range(B):
    idx = np.random.randint(0, m)
    y_true_np[i, idx] = 1.0

# --- TensorFlow Loss ---
sol_loss_tf = SOL_tf(score="accuracy", taus=tau_samples_np, lam=lam)
# Evaluate the loss once and print
tf_loss = sol_loss_tf(y_true_np, y_pred_np).numpy()
print("TensorFlow SOL loss: {:.6f}".format(tf_loss))

# --- PyTorch Loss ---
sol_loss_torch = SOL_torch(score="accuracy", taus=tau_samples_np, lam=lam)
# Convert numpy arrays to torch tensors.
y_pred_torch = torch.tensor(y_pred_np, dtype=torch.float32)
y_true_torch = torch.tensor(y_true_np, dtype=torch.float32)
# Evaluate the loss once and print
torch_loss = sol_loss_torch(y_pred_torch, y_true_torch).item()
print("PyTorch SOL loss:    {:.6f}".format(torch_loss))

# TensorFlow timing
print("\nTensorFlow timing:")
number = 100
timer = timeit.Timer(lambda: sol_loss_tf(y_true_np, y_pred_np).numpy())
tf_time = timer.timeit(number=number) / number
print(f"{tf_time:.6e} seconds per loop (mean of {number} runs)")

# PyTorch timing
print("\nPyTorch timing:")
timer = timeit.Timer(lambda: sol_loss_torch(y_pred_torch, y_true_torch).item())
torch_time = timer.timeit(number=number) / number
print(f"{torch_time:.6e} seconds per loop (mean of {number} runs)")

# Compare results and timing
print("\nResults comparison:")
print(f"Loss difference:    {abs(tf_loss - torch_loss):.6f}")
print(f"TF/PyTorch ratio:   {tf_time / torch_time:.2f}x")
