#!/usr/bin/env python
"""
DIAGNOSTIC: Sinusoidal PE contamination analysis.

Computes PE vectors at various elapsed_hours values and shows that padding
positions (elapsed_hours=0.0) get a non-trivial PE nearly identical to
early-trajectory positions.

NO DATA NEEDED — runs locally with pure math.

Usage:
    python scripts/diagnostics/diag_sinusoidal_pe.py
"""
import math
import numpy as np
import torch


def compute_pe(elapsed_hours: float, d_model: int = 64, time_scale: float = 1.0):
    """Compute sinusoidal PE for a single elapsed_hours value."""
    half_d = d_model // 2
    freq = torch.exp(
        torch.arange(half_d, dtype=torch.float32) * -(math.log(10000.0) / half_d)
    )
    t = torch.tensor([elapsed_hours]).unsqueeze(-1) * time_scale
    pe = torch.cat([torch.sin(t * freq), torch.cos(t * freq)], dim=-1)[:, :d_model]
    return pe.squeeze().numpy()


def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return np.dot(a, b) / denom


def main():
    d_model = 64  # from configs/defaults.yaml

    print("=" * 72)
    print("SINUSOIDAL PE CONTAMINATION ANALYSIS")
    print(f"d_model = {d_model}, time_scale = 1.0 (initial)")
    print("=" * 72)

    # Key elapsed_hours values:
    # 0.0       = PADDING (also could look like the very start)
    # 0.083     = midpoint of first 10min bin
    # 0.167     = midpoint of second 10min bin
    # 0.5, 1.0  = early positions (30min, 1h)
    # 6.0       = transition to 20min bins
    # 24.0      = 1 day
    # 168.0     = 1 week
    # 720.0     = 1 month
    # 2160.0    = 3 months
    test_hours = [0.0, 0.083, 0.167, 0.5, 1.0, 3.0, 6.0, 12.0, 24.0, 72.0, 168.0, 720.0]
    labels = [f"{h}h" for h in test_hours]

    pes = {h: compute_pe(h, d_model) for h in test_hours}

    # --- Padding PE analysis ---
    padding_pe = pes[0.0]
    sin_part = padding_pe[:d_model // 2]
    cos_part = padding_pe[d_model // 2:]

    print("\n--- Padding PE (elapsed_hours = 0.0) ---")
    print(f"  sin components all zero: {np.allclose(sin_part, 0)}")
    print(f"  cos components all one:  {np.allclose(cos_part, 1)}")
    print(f"  L2 norm: {np.linalg.norm(padding_pe):.4f}")
    print(f"  Expected L2 norm (sqrt(d/2)): {math.sqrt(d_model / 2):.4f}")
    print(f"  This is NOT a zero vector — padding gets a STRONG PE signal!")

    # --- Zero PE comparison ---
    zero_pe = np.zeros(d_model)
    print(f"\n--- Comparison with zero PE (what padding SHOULD get) ---")
    print(f"  Zero PE L2 norm: {np.linalg.norm(zero_pe):.4f}")
    print(f"  Padding PE L2 norm: {np.linalg.norm(padding_pe):.4f}")
    print(f"  Ratio: {np.linalg.norm(padding_pe) / max(np.linalg.norm(zero_pe), 1e-12):.1f}x stronger")

    # --- Similarity matrix ---
    print(f"\n--- Cosine similarity matrix ---")
    print(f"  (padding=0.0h should be DISSIMILAR to real positions,")
    print(f"   but high similarity means contamination)")
    header = f"{'':>10s} " + " ".join(f"{l:>8s}" for l in labels)
    print(header)
    for i, h1 in enumerate(test_hours):
        row = f"{labels[i]:>10s} "
        for j, h2 in enumerate(test_hours):
            sim = cosine_sim(pes[h1], pes[h2])
            row += f"{sim:>8.4f} "
        print(row)

    # --- Critical comparisons ---
    sim_padding_first = cosine_sim(pes[0.0], pes[0.083])
    sim_padding_second = cosine_sim(pes[0.0], pes[0.167])
    sim_first_second = cosine_sim(pes[0.083], pes[0.167])

    print(f"\n--- Critical comparisons ---")
    print(f"  Padding (0.0h) vs first bin (0.083h):  cosine_sim = {sim_padding_first:.6f}")
    print(f"  Padding (0.0h) vs second bin (0.167h): cosine_sim = {sim_padding_second:.6f}")
    print(f"  First bin vs second bin:                cosine_sim = {sim_first_second:.6f}")

    if sim_padding_first > 0.99:
        print(f"\n  ** CRITICAL: Padding is NEARLY IDENTICAL to early positions!")
        print(f"     The model CANNOT distinguish padding from the start of trajectory.")
    elif sim_padding_first > 0.95:
        print(f"\n  ** WARNING: Padding is very similar to early positions.")
        print(f"     Attention will struggle to separate them.")
    else:
        print(f"\n  Padding is somewhat distinguishable from early positions.")

    # --- L2 distances ---
    print(f"\n--- L2 distances from padding PE ---")
    for h in test_hours[1:]:
        dist = np.linalg.norm(pes[h] - pes[0.0])
        print(f"  {h:>8.3f}h: L2 dist = {dist:.4f}")

    # --- Effect on attention scores ---
    print(f"\n--- Simulated attention impact ---")
    print(f"  Assume a query Q at position 6h attending to all positions.")
    print(f"  Score(Q, K) = Q^T K / sqrt(d_k).  K includes PE contribution.")
    print(f"  If padding PE is non-zero, padding keys are non-zero,")
    print(f"  and softmax distributes weight to padding positions.")

    q_pe = pes[6.0]  # Query at 6h
    print(f"\n  Raw attention scores from Q@6h to each position (PE only):")
    d_k = d_model // 8  # 8 heads
    for h in test_hours:
        score = np.dot(q_pe, pes[h]) / math.sqrt(d_k)
        print(f"    ->{h:>8.3f}h: score = {score:>8.4f}")

    print("\n" + "=" * 72)
    print("CONCLUSION: Padding positions receive cos(0)=1 PE, making them")
    print("look like valid early-trajectory positions to the attention mechanism.")
    print("FIX: Zero out PE at padding positions using trajectory_lengths.")
    print("=" * 72)


if __name__ == "__main__":
    main()
