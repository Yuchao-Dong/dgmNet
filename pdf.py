import torch
import numpy as np
from sklearn.neighbors import KernelDensity

# Generate discrete data
data = torch.randn(100)  # Example of 100 data points

# Convert data to numpy array and reshape
data_np = data.numpy().reshape(-1, 1)

# Estimate the PDF using KernelDensity
kde = KernelDensity(kernel='gaussian', bandwidth=0.5)  # Choose kernel and bandwidth
kde.fit(data_np)  # Fit the KDE model to the data

# Generate random samples from a uniform distribution within the range of the data
n_samples = 100  # Number of samples to generate
samples = np.random.uniform(data.min(), data.max(), size=(n_samples, 1))  # Generate samples

# Compute the log-density at the generated samples
log_density = kde.score_samples(samples)  # Compute the log-density at the samples

# Exponentiate the log-density to obtain the PDF values at the samples
pdf = torch.exp(torch.from_numpy(log_density))

# Normalize PDF to sum to 1
pdf = pdf / pdf.sum()

# Compute the cumulative distribution function (CDF) from the PDF
cdf = torch.cumsum(pdf, dim=0)

# Use inverse CDF to obtain the sampled points
uniform_samples = torch.rand(n_samples)  # Generate uniform samples
sampled_points = torch.searchsorted(cdf, uniform_samples)  # Use inverse CDF to obtain sampled points

print("Sampled points:", sampled_points)