# SHAP Density Normalization Status

## Your Question
SPO2 is dominating the cohort temporal SHAP heatmap again, even though `density_normalize=True` is being passed to the analysis functions. You fixed this before by introducing density normalization, but it appears to be dominating again.

## Investigation Results

### Confirmed: Density normalization IS being applied correctly

✓ **Training script**: Passes `density_normalize=True` to both `shap_analysis()` and `run_cohort_temporal_shap_analysis()` (lines 252, 261, 270)

✓ **TemporalSHAPAnalyzer**: Receives and stores the flag (line 3089)

✓ **Individual patient analysis**: Applies density normalization when computing per-timeframe channel importance (lines 3383-3391):
```python
if eff > 0 and self.density_normalize:
    # Compute: sum(|SHAP| where measured) / count(measured cells)
    ts_channel_importance = (shap_eff * measured).sum(axis=1) / denom
```

✓ **Cohort aggregation**: Uses the density-normalized per-channel importance values (line 4502-4503):
```python
ch_imp = np.stack([r.ts_channel_importance for r in tf_results])
channel_importance[tf] = ch_imp.mean(axis=0)  # These are already density-normalized
```

✓ **Visualization**: Uses the correct density-normalized values in heatmaps (line 4586)

### How the Normalization Works

For each channel, per sample, per timeframe:
1. **Count measured cells**: How many timesteps have non-zero normalized values
2. **Sum SHAP values**: Sum of |SHAP| over effective steps
3. **Per-measurement importance**: Sum / Count

This should prevent SPO2 (frequently measured) from dominating over less frequently measured features.

### The Measurement Mask Issue

**Potential caveat**: The density normalization uses `ts_np[:, :eff] != 0.0` as the "measured" mask.
- `ts_np` is the **normalized** data (after StandardScaler)
- Zero in normalized data COULD be a measured value at the population mean
- However, this is consistent with how density normalization works elsewhere in `visualize_shap_summary()`

This should not be an issue in practice because:
1. The StandardScaler normalization preserves relative measurement patterns
2. Most features have variation around the mean, not clustered at zero
3. SPO2 and vitals are unlikely to normalize to exactly zero

## Why SPO2 Might Still Dominate

Even WITH density normalization, SPO2 could legitimately dominate if:

1. **SPO2 is genuinely more predictive** — its SHAP values per measurement are still higher than other channels despite density normalization

2. **Patient-level variation** — some patients in the cohort might have much denser SPO2 measurements, skewing the cohort aggregate

3. **Aggregation level** — The cohort heatmap is a mean across patients. If patients have different measurement patterns, the mean might not fully normalize all biases

## Recommendations to Verify

### 1. Check if density normalization is actually reducing SPO2 dominance
Add temporary logging:
```python
# In _compute_shap_for_sample, around line 3386:
if verbose:
    measured_count = measured.sum(axis=1)  # count per channel
    logger.info(f"Measured counts (per-channel): {measured_count}")
    logger.info(f"Ch0 (SPO2?) importance: {ts_channel_importance[0]:.6f}")
```

Compare before/after densitynormalization to confirm the denominator is different for SPO2.

### 2. Investigate per-channel measurement frequency
```python
# Check if SPO2 really is measured much more frequently
for i, ch in enumerate(channel2feature.items()):
    denom = measured.sum(axis=1)[i]
    print(f"{ch}: {denom} measurements out of {eff} steps")
```

### 3. Check if the issue is in temporal aggregation, not spatial

The density normalization in `analyze_patient()` normalizes **per-patient, per-timeframe**. When aggregating across timeframes or patients in `_aggregate_patient_results()`, there's no additional density normalization — it just takes the mean.

This could be an issue if:
- Different timeframes have different measurement densities for each channel
- The cohort mean is dominated by timeframes where SPO2 is most frequently measured

## Possible Solutions

### Option A: Double-check the measurement mask
Replace `ts_np[:, :eff] != 0.0` with a more explicit mask that tracks actual data quality flags if available:
```python
# If you have a separate "data_quality" or "measured_mask" in the dataset:
measured = data_quality_mask[:, :eff]  # Use the authoritative flag
```

### Option B: Aggregate per-timeframe separately, then combine
Instead of averaging across patients (which might have different timeframe measurement patterns), compute cohort importance per-timeframe with explicit measurement counts, then aggregate.

### Option C: Apply density normalization at cohort level too
When aggregating across patients and timeframes in `_aggregate_patient_results()`, compute:
```python
# Instead of: channel_importance[tf] = ch_imp.mean(axis=0)
# Use a weighted mean that accounts for measurement frequency
```

## Current Commit Status
- ✓ Fixed static categorical SHAP near-zero values (separate issue)
- ✓ Added density normalization diagnostic logging
- ⚠️ Density normalization implementation appears correct, but root cause of SPO2 dominance TBD

## Next Steps
1. Run SHAP analysis with the temporary logging to see actual density normalization numbers
2. Compare results WITH and WITHOUT `density_normalize=True` to isolate the effect
3. Identify if SPO2 truly has more measurements or if something else is happening
