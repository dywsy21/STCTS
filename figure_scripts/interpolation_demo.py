import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 1. Generate Synthetic Ground Truth (Natural Speech Pitch Contour)
np.random.seed(42)
t = np.linspace(0, 10, 1000)  # 10 seconds, 100Hz resolution
# Base curve + some random variation to simulate natural speech jitter
ground_truth = 150 + 20 * np.sin(0.5 * t) + 10 * np.sin(1.2 * t) + 5 * np.cos(2.5 * t) 
# Add micro-jitter (which is lost in compression)
ground_truth += 2 * np.random.normal(0, 0.5, len(t))

# 2. Sparse Sampling (Keyframes)
# Sample every 2 seconds (0.5 Hz)
sample_indices = np.linspace(0, len(t)-1, 6, dtype=int)
t_sampled = t[sample_indices]
y_sampled = ground_truth[sample_indices]

# 3. Reconstruction (Cubic Spline Interpolation)
cs = CubicSpline(t_sampled, y_sampled)
reconstruction = cs(t)

# 4. Plotting
plt.figure(figsize=(10, 5), dpi=150)
plt.style.use('seaborn-v0_8-whitegrid')

# Plot Ground Truth
plt.plot(t, ground_truth, color='gray', linestyle='--', alpha=0.6, label='Original Pitch (Ground Truth)')

# Plot Reconstruction
plt.plot(t, reconstruction, color="#44B7FF", linewidth=3.0, alpha=0.9, label='Reconstructed (Interpolated)')

# Plot Keyframes
plt.scatter(t_sampled, y_sampled, color='#d62728', s=100, zorder=5, label='Transmitted Keyframes (0.5 Hz)')

# Annotations
# plt.title('Sparse Prosody Interpolation Principle', fontsize=14, pad=15)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Pitch / F0 (Hz)', fontsize=12)
plt.legend(loc='upper right', frameon=True)
plt.grid(True, alpha=0.3)

# Highlight the concept
plt.annotate('Micro-prosody details lost\n(Acceptable for intelligibility)', 
             xy=(3.75, 152), xytext=(3.75, 135),
             arrowprops=dict(facecolor='gray', shrink=0.05, alpha=0.7),
             fontsize=10, ha='center', alpha=0.8)

plt.annotate('Macro-prosody preserved\nvia Spline Interpolation', 
             xy=(7.25, 143), xytext=(7.25, 120),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=10, ha='center')

plt.tight_layout()
plt.savefig('prosody_interpolation_concept.png')
print('Plot generated: prosody_interpolation_concept.png')
