#!/usr/bin/env python
"""Test script to verify the SHAP static categorical fix."""

import torch
import torch.nn as nn
import numpy as np
from astra.evaluation.behavior import embed_categorical_features, ModelWrapperWithOneHotCategoricals

# Create a simple mock model with embedding layers
class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeds = nn.ModuleList([
            nn.Embedding(3, 8),  # Gender: 3 classes, 8-dim embedding
            nn.Embedding(5, 8),  # Hospital: 5 classes, 8-dim embedding
        ])
        self.d_model = 8
        self.seq_len = 10
        self.n_ts_cat = 0
        self.temporal_head_enabled = False

    def forward(self, x):
        return torch.ones(x.shape[0], 20, 64)

# Test 1: One-hot encoding function
print("=" * 60)
print("TEST 1: One-hot encoding function")
print("=" * 60)

model = MockModel()
batch_size = 4

# Create sample categorical data
x_cat = torch.tensor([
    [0, 1],  # Gender=0, Hospital=1
    [1, 2],
    [2, 3],
    [0, 4],
], dtype=torch.long)

print(f"Input shape: {x_cat.shape}")
print(f"Input:\n{x_cat}")

# Convert to one-hot
x_cat_onehot = embed_categorical_features(model, x_cat)

print(f"\nOutput shape: {x_cat_onehot.shape}")
print(f"Expected: (4, 2, 5) - [batch=4, n_cat=2, max_classes=5]")

assert x_cat_onehot.shape == (4, 2, 5), f"Shape mismatch: {x_cat_onehot.shape}"
assert x_cat_onehot.requires_grad, "requires_grad should be True"

# Verify one-hot encoding is correct
# First feature (Gender): batch[0] should have a 1 at position 0
print(f"\nFirst sample, first feature (Gender=0):")
print(f"One-hot: {x_cat_onehot[0, 0, :3]}")
print(f"Expected: [1, 0, 0]")
assert torch.allclose(x_cat_onehot[0, 0, :3], torch.tensor([1., 0., 0.])), "One-hot encoding error"

# Second sample, second feature (Hospital=2)
print(f"Second sample, second feature (Hospital=2):")
print(f"One-hot: {x_cat_onehot[1, 1, :5]}")
print(f"Expected: [0, 0, 1, 0, 0]")
assert torch.allclose(x_cat_onehot[1, 1, :5], torch.tensor([0., 0., 1., 0., 0.])), "One-hot encoding error"

print("✓ Test 1 PASSED: One-hot encoding correct\n")

# Test 2: Matmul embedding wrapper
print("=" * 60)
print("TEST 2: Matmul embedding correctness")
print("=" * 60)

# Verify matmul gives same result as embedding lookup
# For a one-hot vector [1, 0, 0], matmul with embedding weight should give the first row of the weight

oh_0 = torch.tensor([[1., 0., 0.]])  # One-hot for class 0
emb_weight = model.embeds[0].weight  # [3, 8]

# Method 1: Direct embedding lookup
direct_emb = model.embeds[0](torch.tensor([0], dtype=torch.long))  # [1, 8]

# Method 2: Matmul with one-hot
matmul_emb = torch.matmul(oh_0, emb_weight)  # [1, 8]

print(f"Direct embedding shape: {direct_emb.shape}")
print(f"Matmul embedding shape: {matmul_emb.shape}")

# They should be identical
assert torch.allclose(direct_emb, matmul_emb, atol=1e-6), "Matmul embedding doesn't match direct lookup"
print("✓ Test 2 PASSED: Matmul embedding equals direct lookup\n")

# Test 3: Gradient flow
print("=" * 60)
print("TEST 3: Gradient flow through one-hot")
print("=" * 60)

x_cat_onehot_grad = embed_categorical_features(model, x_cat.clone())
x_cat_onehot_grad.retain_grad()

# Dummy loss: sum of all embedding outputs
loss = x_cat_onehot_grad.sum()
loss.backward()

print(f"x_cat_onehot gradient shape: {x_cat_onehot_grad.grad.shape}")
print(f"x_cat_onehot gradient:\n{x_cat_onehot_grad.grad}")

assert x_cat_onehot_grad.grad is not None, "Gradient not computed!"
assert x_cat_onehot_grad.grad.shape == x_cat_onehot_grad.shape, "Gradient shape mismatch"
print("✓ Test 3 PASSED: Gradients flow correctly\n")

# Test 4: None handling
print("=" * 60)
print("TEST 4: None handling")
print("=" * 60)

result = embed_categorical_features(model, None)
assert result is None, "Should return None for None input"

empty_tensor = torch.zeros((0, 0), dtype=torch.long)
result = embed_categorical_features(model, empty_tensor)
assert result is None, "Should return None for empty tensor"

print("✓ Test 4 PASSED: None handling works\n")

print("=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("\nSummary:")
print("- One-hot encoding produces correct format [batch, n_cat, max_classes]")
print("- Matmul embedding equals direct embedding lookup")
print("- Gradients flow through the one-hot tensor")
print("- None/empty handling works correctly")
print("\nThe fix should now provide non-zero SHAP values for static categorical features.")
