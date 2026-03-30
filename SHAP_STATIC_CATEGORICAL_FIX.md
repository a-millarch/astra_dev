# Fix: Static Categorical Features Show Near-Zero SHAP Values

## Problem

Static categorical features (SEX, FIRST_HOSPITAL, comorbidity flags, etc.) were showing near-zero SHAP values in cohort temporal SHAP heatmaps, making them appear irrelevant despite potentially being clinically important predictors.

## Root Cause

The previous implementation used a flawed pre-embedding strategy:

1. **Pre-embedding inside `torch.no_grad()`**: Integer category indices were embedded to dense vectors, then `requires_grad=True` was set on the result.
2. **SHAP sees embedding space, not categorical input**: `GradientExplainer` computed gradients w.r.t. embedding vectors, not category probabilities.
3. **Zero difference term for low-cardinality features**: For static categories like `SEX` (2-3 values), many patients share the same embedding vector. The difference `emb(test_cat) - mean(emb(bg_cat))` is ≈ 0 → SHAP ≈ 0.
4. **Signal dilution via mean**: Aggregation `mean(axis=2)` over d_model dimensions (~128) further diluted the gradient signal.

**Formula (broken):**
```
SHAP ≈ gradient(embedding) × (emb[test] - mean(emb[background]))
     ≈ 0  (when both test and background have same category value)
```

## Solution

Replace pre-embedding with **one-hot encoding + differentiable matmul embedding**:

### Key Changes

#### 1. New `embed_categorical_features()` Function
Converts integer category indices to one-hot encodings instead of embedding them:

```python
def embed_categorical_features(model, x_cat):
    # Returns: [batch, n_cat, max_classes] one-hot tensor with requires_grad=True
    # The model wrapper will perform: one_hot @ embedding.weight (differentiable)
```

**Advantages:**
- One-hot vectors vary meaningfully across background samples
- GradientExplainer sees real probability differences between foreground and background
- Gradients flow through the one-hot representation, not the pre-computed lookup

#### 2. New `ModelWrapperWithOneHotCategoricals` Class
Replaces `ModelWrapperWithEmbeddings` for static categoricals:

```python
class ModelWrapperWithOneHotCategoricals(nn.Module):
    def forward(self, x_ts, x_ts_cat_embedded, x_cat_onehot, x_cont):
        # Performs matmul embedding internally:
        for i, emb in enumerate(self.model.embeds):
            oh_i = x_cat_onehot[:, i, :emb.num_embeddings]  # [batch, num_classes]
            emb_w = emb.weight  # [num_classes, d_model]
            x_cat_i = torch.matmul(oh_i, emb_w)  # [batch, d_model] — DIFFERENTIABLE
```

This matmul is fully differentiable, so gradients flow correctly.

#### 3. Fixed Aggregation
Changed from `mean(axis=2)` (embedding dimensions) to `sum(axis=2)` (one-hot classes):

```python
# Old: cat_shap = np.abs(cat_shap_embedded).mean(axis=2)  # [n_samples, n_cat]
# New: cat_shap = np.abs(cat_shap_onehot).sum(axis=2)      # [n_samples, n_cat]
```

This sums the absolute SHAP contributions across all class probabilities, giving overall feature importance.

#### 4. Fixed Multi-Class Bug
In `_compute_shap_for_sample()`, fixed incorrect class selection:

```python
# Old: shap_values = shap_values[0]  # Always class 0 (alive)
# New: selected_class = min(1, n_classes - 1); shap_values = shap_values[selected_class]  # Class 1 (mortality)
```

### Updated Locations

- **`embed_categorical_features()`**: Line 1242-1274 in `behavior.py`
- **`ModelWrapperWithOneHotCategoricals`**: Lines 982-1097
- **`calculate_shap_from_dataloaders()`**: Lines 1520-1636
- **`_compute_shap_for_sample()`**: Lines 3173-3259

## Why This Works

**New formula (fixed):**
```
SHAP ≈ gradient(one_hot) × (one_hot[test] - mean(one_hot[background]))
     > 0  (when test and background categories differ)
```

For static categorical features:
- Background patients have diverse category values (different one-hot vectors)
- Gradient w.r.t. one-hot is non-zero (matmul is fully differentiable)
- Difference term is meaningful (distinct one-hot probability distributions)
- Result: **Meaningful, non-zero SHAP values**

For temporal categorical features:
- Already worked because they use raw multi-hot float tensors (which vary continuously)
- No changes needed

For static continuous features:
- Already worked because they're passed as raw floats through differentiable `Conv1d`
- No changes needed

## Testing

All functionality tested in `test_shap_fix.py`:
- ✓ One-hot encoding produces correct format `[batch, n_cat, max_classes]`
- ✓ Matmul embedding equals direct embedding lookup
- ✓ Gradients flow through the one-hot tensor
- ✓ None/empty handling works correctly

## Expected Outcomes

After this fix:
1. Static categorical features will show **meaningful, non-zero SHAP values**
2. Feature importance rankings will reflect actual clinical importance (e.g., AGE should have high importance)
3. Static continuous and temporal categorical features remain unaffected
4. SHAP explainability output becomes more trustworthy for clinical interpretation

## Backward Compatibility

- **Breaking change:** Cached SHAP results with old implementation will have different values
- **Data format:** Return dictionary now includes `cat_shap_onehot` (the intermediate one-hot SHAP values)
- **Visualizations:** Existing visualization code handles both 1D and 2D cat_shap arrays, so no changes needed there
