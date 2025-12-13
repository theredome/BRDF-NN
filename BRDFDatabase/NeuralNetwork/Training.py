import numpy as np

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
data_path = "alum.txt"
HIDDEN_SIZE = 32          # Number of neurons in hidden layer (keep small per project rules)
LR = 1e-2                 # Learning rate for SGD
EPOCHS = 100              # Maximum training epochs
PATIENCE = 8              # Early stopping: epochs without improvement before stopping
SEED = 42                 # For reproducibility

np.random.seed(SEED)

# =============================================================================
# PHASE 1: DATA LOADING
# =============================================================================
# Load MERL BRDF data: [theta_i, phi_i, theta_o, phi_o, red, green, blue]
# Using float32 to save memory (4 bytes vs 8 bytes for float64)

print("Loading data...")
data = np.loadtxt(data_path, dtype=np.float32)
print(f"Loaded {len(data):,} samples")

# Extract angle columns
theta_i = data[:, 0]
phi_i = data[:, 1]
theta_o = data[:, 2]
phi_o = data[:, 3]

# PROJECT REQUIREMENT: Only predict RED channel (single scalar output)
# Shape: (N, 1) - keeping 2D for consistent matrix operations
y = data[:, 4:5].astype(np.float32)

# =============================================================================
# INPUT ENCODING: Sin/Cos Transformation
# =============================================================================
# WHY sin/cos encoding?
# - Raw angles have discontinuities (e.g., 359° and 1° are "far" numerically but close angularly)
# - sin/cos creates a continuous representation on the unit circle
# - This helps the network learn smoother functions
#
# Input: 4 angles → Output: 8 features

sin_theta_i, cos_theta_i = np.sin(theta_i), np.cos(theta_i)
sin_phi_i, cos_phi_i = np.sin(phi_i), np.cos(phi_i)
sin_theta_o, cos_theta_o = np.sin(theta_o), np.cos(theta_o)
sin_phi_o, cos_phi_o = np.sin(phi_o), np.cos(phi_o)

# Stack into input matrix X with shape (N, 8)
X = np.stack([
    sin_theta_i, cos_theta_i,
    sin_phi_i, cos_phi_i,
    sin_theta_o, cos_theta_o,
    sin_phi_o, cos_phi_o
], axis=1).astype(np.float32)

N, D = X.shape  # N = number of samples, D = 8 features
print(f"Input shape: {X.shape}, Output shape: {y.shape}")

# =============================================================================
# PHASE 2: TRAIN/TEST SPLIT (50/50 as per project rules)
# =============================================================================
# Shuffle indices to ensure random split (not biased by data order)
indices = np.random.permutation(N)
split = N // 2

train_idx = indices[:split]
test_idx = indices[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"Train set: {len(X_train):,} samples")
print(f"Test set:  {len(X_test):,} samples")

# =============================================================================
# OUTPUT NORMALIZATION (Z-Score)
# =============================================================================
# WHY normalize outputs?
# - BRDF values can range from 0 to 1000+ (high dynamic range)
# - Large target values → large MSE → unstable gradients
# - Normalization centers data around 0 with unit variance
# - Network learns in a well-scaled space, then we denormalize for evaluation
#
# IMPORTANT: Compute statistics ONLY from training set (no data leakage)

y_mean = np.mean(y_train)
y_std = np.std(y_train)

# Normalize both sets using TRAINING statistics
y_train_norm = (y_train - y_mean) / y_std
y_test_norm = (y_test - y_mean) / y_std

print(f"\nOutput normalization: mean={y_mean:.4f}, std={y_std:.4f}")

# =============================================================================
# PHASE 3: NETWORK WEIGHT INITIALIZATION
# =============================================================================
# Architecture: Input(8) → Hidden(HIDDEN_SIZE) → Output(1)
#
# Xavier/Glorot initialization: scale weights by sqrt(2 / (fan_in + fan_out))
# This prevents vanishing/exploding gradients at initialization

def initialize_weights(input_dim, hidden_dim, output_dim):
    """
    Initialize network weights using Xavier initialization.
    Returns: W1, b1, W2, b2
    """
    # Hidden layer weights: (input_dim, hidden_dim)
    limit1 = np.sqrt(6.0 / (input_dim + hidden_dim))
    W1 = np.random.uniform(-limit1, limit1, (input_dim, hidden_dim)).astype(np.float32)
    b1 = np.zeros((1, hidden_dim), dtype=np.float32)
    
    # Output layer weights: (hidden_dim, output_dim)
    limit2 = np.sqrt(6.0 / (hidden_dim + output_dim))
    W2 = np.random.uniform(-limit2, limit2, (hidden_dim, output_dim)).astype(np.float32)
    b2 = np.zeros((1, output_dim), dtype=np.float32)
    
    return W1, b1, W2, b2

# Initialize weights
W1, b1, W2, b2 = initialize_weights(D, HIDDEN_SIZE, 1)
print(f"\nNetwork architecture: {D} → {HIDDEN_SIZE} → 1")
print(f"Total parameters: {W1.size + b1.size + W2.size + b2.size}")

# =============================================================================
# PHASE 4: FORWARD PASS
# =============================================================================
# Forward propagation computes: input → hidden → output
#
# Hidden layer: z1 = X @ W1 + b1, a1 = tanh(z1)
# Output layer: z2 = a1 @ W2 + b2, output = z2 (linear, no activation)

def forward(X, W1, b1, W2, b2):
    """
    Forward pass through the network.
    Returns: output prediction and cached values for backprop
    """
    # Hidden layer
    z1 = X @ W1 + b1           # Linear transformation
    a1 = np.tanh(z1)           # Activation (tanh squashes to [-1, 1])
    
    # Output layer (linear - no activation for regression)
    z2 = a1 @ W2 + b2
    
    # Cache values needed for backward pass
    cache = (X, z1, a1)
    return z2, cache

# =============================================================================
# PHASE 5: BACKWARD PASS (BACKPROPAGATION)
# =============================================================================
# Backprop computes gradients of loss w.r.t. each weight
#
# MSE Loss: L = (1/N) * sum((y_pred - y_true)^2)
# dL/dy_pred = (2/N) * (y_pred - y_true)
#
# Chain rule propagates gradients backward through the network

def backward(y_pred, y_true, cache, W2):
    """
    Backward pass: compute gradients for all weights.
    Returns: dW1, db1, dW2, db2
    """
    X, z1, a1 = cache
    m = y_true.shape[0]  # batch size
    
    # Output layer gradients
    # dL/dz2 = dL/dy_pred = (2/m) * (y_pred - y_true)
    dz2 = (2.0 / m) * (y_pred - y_true)
    
    # dL/dW2 = a1.T @ dz2
    dW2 = a1.T @ dz2
    
    # dL/db2 = sum of dz2 across batch
    db2 = np.sum(dz2, axis=0, keepdims=True)
    
    # Hidden layer gradients (chain rule through tanh)
    # dL/da1 = dz2 @ W2.T
    da1 = dz2 @ W2.T
    
    # tanh derivative: d/dx tanh(x) = 1 - tanh(x)^2 = 1 - a1^2
    dz1 = da1 * (1 - a1 ** 2)
    
    # dL/dW1 = X.T @ dz1
    dW1 = X.T @ dz1
    
    # dL/db1 = sum of dz1 across batch
    db1 = np.sum(dz1, axis=0, keepdims=True)
    
    return dW1, db1, dW2, db2

# =============================================================================
# LOSS FUNCTION
# =============================================================================
def mse_loss(y_pred, y_true):
    """Mean Squared Error loss."""
    return np.mean((y_pred - y_true) ** 2)

# =============================================================================
# PHASE 6: TRAINING LOOP WITH SGD AND EARLY STOPPING
# =============================================================================
print("\n" + "="*50)
print("TRAINING")
print("="*50)

# Track losses for plotting
train_losses = []
test_losses = []

# Early stopping variables
best_test_loss = float('inf')
best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    # Shuffle training data each epoch (prevents learning order patterns)
    perm = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[perm]
    y_train_shuffled = y_train_norm[perm]  # Use NORMALIZED targets
    
    # -------------------------------------------------------------------------
    # TRAINING PASS (full batch SGD for simplicity)
    # -------------------------------------------------------------------------
    # Forward pass
    y_pred_train, cache = forward(X_train_shuffled, W1, b1, W2, b2)
    
    # Compute training loss (in normalized space)
    train_loss = mse_loss(y_pred_train, y_train_shuffled)
    train_losses.append(train_loss)
    
    # Backward pass
    dW1, db1, dW2, db2 = backward(y_pred_train, y_train_shuffled, cache, W2)
    
    # SGD weight update: W = W - LR * gradient
    W1 -= LR * dW1
    b1 -= LR * db1
    W2 -= LR * dW2
    b2 -= LR * db2
    
    # -------------------------------------------------------------------------
    # TESTING PASS (evaluate on held-out data)
    # -------------------------------------------------------------------------
    y_pred_test, _ = forward(X_test, W1, b1, W2, b2)
    test_loss = mse_loss(y_pred_test, y_test_norm)  # Use NORMALIZED targets
    test_losses.append(test_loss)
    
    # -------------------------------------------------------------------------
    # EARLY STOPPING CHECK
    # -------------------------------------------------------------------------
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
        epochs_without_improvement = 0
        marker = " *"  # Mark improvement
    else:
        epochs_without_improvement += 1
        marker = ""
    
    # Print progress every 10 epochs or on improvement
    if epoch % 10 == 0 or marker:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}{marker}")
    
    # Stop if no improvement for PATIENCE epochs
    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
        break

# Restore best weights
W1, b1, W2, b2 = best_weights
print(f"\nRestored best weights (Test Loss: {best_test_loss:.6f})")

# =============================================================================
# PHASE 7: FINAL EVALUATION AND RESULTS
# =============================================================================
print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)

# Evaluate on both sets with best weights
# Network outputs are in normalized space, so we denormalize for true MSE
y_pred_train_norm, _ = forward(X_train, W1, b1, W2, b2)
y_pred_test_norm, _ = forward(X_test, W1, b1, W2, b2)

# Denormalize predictions: y_pred = y_pred_norm * std + mean
y_pred_train_final = y_pred_train_norm * y_std + y_mean
y_pred_test_final = y_pred_test_norm * y_std + y_mean

# Compute MSE in original scale (interpretable units)
final_train_loss = mse_loss(y_pred_train_final, y_train)
final_test_loss = mse_loss(y_pred_test_final, y_test)

# Also report normalized MSE (what the network actually optimized)
final_train_loss_norm = mse_loss(y_pred_train_norm, y_train_norm)
final_test_loss_norm = mse_loss(y_pred_test_norm, y_test_norm)

print(f"Normalized MSE  - Train: {final_train_loss_norm:.6f} | Test: {final_test_loss_norm:.6f}")
print(f"Original Scale  - Train: {final_train_loss:.2f} | Test: {final_test_loss:.2f}")

# Save weights AND normalization parameters for inference
np.savez('best_weights.npz', W1=W1, b1=b1, W2=W2, b2=b2, y_mean=y_mean, y_std=y_std)
print("\nWeights saved to 'best_weights.npz' (includes normalization params)")

# =============================================================================
# OPTIONAL: PLOT TRAINING CURVES (if matplotlib available)
# =============================================================================
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2, color='#2196F3')
    plt.plot(test_losses, label='Test Loss', linewidth=2, color='#FF5722')
    
    # Set y-axis to start from 0 with headroom above max loss
    # This makes train/test curves appear proportionally closer
    max_loss = max(max(train_losses), max(test_losses))
    plt.ylim(0, max_loss * 1.5)  # 50% headroom above max
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss (Normalized)', fontsize=12)
    plt.title('Training vs Test Loss (BRDF Red Channel)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add text annotation with final values
    plt.text(0.98, 0.95, f'Final Train: {train_losses[-1]:.4f}\nFinal Test: {test_losses[-1]:.4f}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=150)
    plt.show()
    print("Training curve saved to 'training_curve.png'")
except ImportError:
    print("\nNote: Install matplotlib to visualize training curves")
    print("  pip install matplotlib")
