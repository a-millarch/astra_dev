"""
Extended diagnostic: compare ALL model inputs (tabular, continuous TS,
categorical TS) between batch training pipeline and from-CSV inference pipeline.

Requires: data cache loaded, session loaded, PatientContext built.

Usage (in notebook after existing diagnostic):
    %run scripts/diagnose_features.py
    # or import and call:
    from scripts.diagnose_features import compare_all_features
    compare_all_features(data, session, ctx, cpr_hash, service_date)
"""
import numpy as np
import pandas as pd
import torch


def compare_all_features(data, session, ctx, cpr_hash, service_date):
    """Compare batch vs inference for tabular + categorical TS features.

    Args:
        data: Batch data dict from prepare_data_and_dls_cached().
        session: InferenceSession with loaded model.
        ctx: PatientContext built from CSV.
        cpr_hash: Full CPR hash string.
        service_date: ServiceDate to disambiguate multiple trajectories for the same CPR.
    """
    device = session.device

    # ---- 1. Find patient in holdout by batch PID ----
    batch_base = pd.read_pickle("data/interim/base_df.pkl")
    sd = pd.Timestamp(service_date)
    matches = batch_base[
        (batch_base["CPR_hash"] == cpr_hash)
        & (batch_base["ServiceDate"] == sd)
    ]
    if matches.empty:
        # Try prefix match on CPR + exact ServiceDate
        matches = batch_base[
            batch_base["CPR_hash"].str.startswith(cpr_hash[:16])
            & (batch_base["ServiceDate"] == sd)
        ]
    if matches.empty:
        print(f"ERROR: CPR_hash {cpr_hash[:8]}... + ServiceDate {sd} not found in batch base_df")
        return

    batch_pid = int(matches.iloc[0]["PID"])
    holdout_pids = data["holdout"].tab_df["PID"].tolist()
    if batch_pid not in holdout_pids:
        print(f"ERROR: Batch PID {batch_pid} not in holdout. Is this a holdout patient?")
        return

    sample_idx = holdout_pids.index(batch_pid)

    # Check if patient is deceased (mask_mortality applies in batch only)
    patient_row = matches.iloc[0]
    is_deceased = pd.notna(patient_row.get("DOD"))

    print(f"Patient: CPR={cpr_hash[:8]}... batch_PID={batch_pid} holdout_idx={sample_idx}")
    if is_deceased:
        print(f"  ** Deceased patient (DOD={patient_row['DOD']}) — trajectory length "
              f"diff expected (mask_mortality in batch only)")

    # ---- 2. Get batch tensors (normalized, what model sees in training) ----
    holdout_ds = data["holdout_mixed_dls"]._train_ds
    (x_ts_b, (x_cat_b, x_cont_b), x_ts_cat_b, traj_b), y_b = holdout_ds[sample_idx]

    # ---- 3. Get inference tensors via _prepare_tensors ----
    x_ts_i, x_cat_i, x_cont_i, x_ts_cat_i, traj_i = session._prepare_tensors(
        ctx.x_ts, ctx.x_ts_cat, ctx.tab_df
    )
    # Squeeze batch dim
    x_ts_i = x_ts_i.cpu().squeeze(0)
    x_cat_i = x_cat_i.cpu().squeeze(0)
    x_cont_i = x_cont_i.cpu().squeeze(0)
    x_ts_cat_i = x_ts_cat_i.cpu().squeeze(0)

    traj = min(int(traj_b), traj_i)
    traj_diff = traj_i - int(traj_b)
    traj_note = ""
    if traj_diff != 0 and is_deceased:
        traj_note = " (expected: mask_mortality in batch only)"
    elif traj_diff != 0:
        traj_note = " (UNEXPECTED for alive patient)"
    print(f"Trajectory lengths: batch={int(traj_b)} inference={traj_i}{traj_note}")

    # ---- 4. Compare tabular categorical ----
    print(f"\n{'='*50}")
    print("TABULAR CATEGORICAL")
    print(f"{'='*50}")
    cat_match = torch.equal(x_cat_b, x_cat_i)
    print(f"Match: {cat_match}")
    if not cat_match:
        classes = session.bundle['model_params']['classes']
        for i, col in enumerate(classes):
            if i >= len(x_cat_b) or i >= len(x_cat_i):
                break
            if x_cat_b[i] != x_cat_i[i]:
                cats = list(classes[col])
                b_val = cats[x_cat_b[i]] if x_cat_b[i] < len(cats) else f"?({x_cat_b[i]})"
                i_val = cats[x_cat_i[i]] if x_cat_i[i] < len(cats) else f"?({x_cat_i[i]})"
                print(f"  DIFF {col}: batch={b_val} (idx={x_cat_b[i]}), "
                      f"inf={i_val} (idx={x_cat_i[i]})")
    else:
        # Show values for context
        classes = session.bundle['model_params']['classes']
        for i, col in enumerate(classes):
            if i >= len(x_cat_b):
                break
            cats = list(classes[col])
            val = cats[x_cat_b[i]] if x_cat_b[i] < len(cats) else f"?({x_cat_b[i]})"
            print(f"  {col}: {val} (idx={x_cat_b[i]})")

    # ---- 5. Compare tabular continuous ----
    print(f"\n{'='*50}")
    print("TABULAR CONTINUOUS")
    print(f"{'='*50}")
    num_cols = session.bundle.get('tab_feature_names', [])
    if x_cont_b.numel() > 0:
        cont_diff = (x_cont_b - x_cont_i).abs()
        print(f"Max diff: {cont_diff.max():.6f}")
        for j, col in enumerate(num_cols):
            if j >= len(x_cont_b):
                break
            d = abs(float(x_cont_b[j]) - float(x_cont_i[j]))
            marker = " <-- DIFF" if d > 1e-4 else ""
            print(f"  {col}: batch={float(x_cont_b[j]):.6f} "
                  f"inf={float(x_cont_i[j]):.6f} diff={d:.6f}{marker}")
    else:
        print("  (no continuous tabular features)")

    # ---- 6. Compare categorical TS (multi-hot) ----
    print(f"\n{'='*50}")
    print("CATEGORICAL TS (multi-hot)")
    print(f"{'='*50}")
    print(f"Shape: batch={x_ts_cat_b.shape} inf={x_ts_cat_i.shape}")
    cat_ts_diff = (x_ts_cat_b[:, :traj].float() - x_ts_cat_i[:, :traj].float()).abs()
    n_diff = int((cat_ts_diff > 0).sum())
    print(f"Max diff: {cat_ts_diff.max():.0f}, Positions with diff: {n_diff}")
    if n_diff > 0:
        ts_cat_names = session.bundle.get('ts_cat_names', [])
        for dim in range(cat_ts_diff.shape[0]):
            d = cat_ts_diff[dim]
            if d.max() > 0:
                name = ts_cat_names[dim] if dim < len(ts_cat_names) else f"dim_{dim}"
                steps = torch.where(d > 0)[0].tolist()
                print(f"  {name}: {int(d.sum())} total diff across {len(steps)} steps")
                # Show first few differing steps
                for s in steps[:5]:
                    print(f"    step {s}: batch={int(x_ts_cat_b[dim, s])} "
                          f"inf={int(x_ts_cat_i[dim, s])}")

    # ---- 7. Compare continuous TS ----
    print(f"\n{'='*50}")
    print("CONTINUOUS TS")
    print(f"{'='*50}")
    ts_channel_names = session.bundle.get('ts_channel_names', [])
    print(f"Shape: batch={x_ts_b.shape} inf={x_ts_i.shape}")
    ts_diff = (x_ts_b[:, :traj].float() - x_ts_i[:, :traj].float()).abs()
    n_ts_diff = int((ts_diff > 1e-4).sum())
    print(f"Max diff: {ts_diff.max():.6f}, Positions with diff (>1e-4): {n_ts_diff}")

    # Per-channel summary
    n_channels = x_ts_b.shape[0]
    channels_ok = 0
    channels_diff = []
    for ch in range(n_channels):
        ch_name = ts_channel_names[ch] if ch < len(ts_channel_names) else f"ch_{ch}"
        ch_diff = ts_diff[ch]  # [traj]
        ch_max = float(ch_diff.max())
        ch_n_diff = int((ch_diff > 1e-4).sum())
        if ch_n_diff == 0:
            channels_ok += 1
        else:
            channels_diff.append((ch, ch_name, ch_max, ch_n_diff))

    print(f"Channels matching: {channels_ok}/{n_channels}")
    if channels_diff:
        print(f"Channels with differences:")
        for ch, ch_name, ch_max, ch_n_diff in channels_diff:
            # Count non-zero values in each to understand sparsity
            b_nonzero = int((x_ts_b[ch, :traj].abs() > 1e-6).sum())
            i_nonzero = int((x_ts_i[ch, :traj].abs() > 1e-6).sum())
            print(f"  {ch_name} (ch={ch}): max_diff={ch_max:.6f}, "
                  f"n_diff={ch_n_diff}/{traj}, "
                  f"nonzero batch={b_nonzero} inf={i_nonzero}")
            # Show first few differing timesteps
            diff_steps = torch.where(ch_diff > 1e-4)[0].tolist()
            for s in diff_steps[:5]:
                print(f"    step {s}: batch={float(x_ts_b[ch, s]):.6f} "
                      f"inf={float(x_ts_i[ch, s]):.6f} "
                      f"diff={float(ch_diff[s]):.6f}")
            if len(diff_steps) > 5:
                print(f"    ... and {len(diff_steps) - 5} more steps")

    # Check padding region (beyond shared trajectory)
    if traj < x_ts_b.shape[1] or traj < x_ts_i.shape[1]:
        b_pad_nonzero = int((x_ts_b[:, traj:].abs() > 1e-6).sum()) if traj < x_ts_b.shape[1] else 0
        i_pad_nonzero = int((x_ts_i[:, traj:].abs() > 1e-6).sum()) if traj < x_ts_i.shape[1] else 0
        if b_pad_nonzero > 0 or i_pad_nonzero > 0:
            print(f"  WARNING: non-zero values in padding region: "
                  f"batch={b_pad_nonzero} inf={i_pad_nonzero}")

    # ---- 8. Direct model forward pass comparison ----
    print(f"\n{'='*50}")
    print("MODEL PREDICTIONS (direct forward pass)")
    print(f"{'='*50}")
    with torch.no_grad():
        logits_b = session.model((
            x_ts_b.unsqueeze(0).float().to(device),
            (x_cat_b.unsqueeze(0).to(device),
             x_cont_b.unsqueeze(0).float().to(device)),
            x_ts_cat_b.unsqueeze(0).float().to(device),
            traj_b.unsqueeze(0).to(device),
        ))
        logits_i = session.model((
            x_ts_i.unsqueeze(0).float().to(device),
            (x_cat_i.unsqueeze(0).to(device),
             x_cont_i.unsqueeze(0).float().to(device)),
            x_ts_cat_i.unsqueeze(0).float().to(device),
            torch.tensor([traj_i], dtype=torch.long, device=device),
        ))
        if session.is_temporal:
            p_b = float(torch.sigmoid(logits_b)[0, traj - 1])
            p_i = float(torch.sigmoid(logits_i)[0, traj - 1])
        else:
            p_b = float(torch.softmax(logits_b, dim=1)[0, 1])
            p_i = float(torch.softmax(logits_i, dim=1)[0, 1])

    print(f"Batch tensors:     P={p_b:.6f} (traj_len={int(traj_b)})")
    print(f"Inference tensors: P={p_i:.6f} (traj_len={traj_i})")
    print(f"Difference:        {abs(p_b - p_i):.6f}")

    # ---- 9. Hybrid tests: swap individual components to isolate impact ----
    if not cat_match or (x_cont_b.numel() > 0 and cont_diff.max() > 1e-4):
        print(f"\n{'='*50}")
        print("HYBRID: batch TS + inference tabular")
        print(f"{'='*50}")
        with torch.no_grad():
            logits_h = session.model((
                x_ts_b.unsqueeze(0).float().to(device),
                (x_cat_i.unsqueeze(0).to(device),
                 x_cont_i.unsqueeze(0).float().to(device)),
                x_ts_cat_b.unsqueeze(0).float().to(device),
                traj_b.unsqueeze(0).to(device),
            ))
            if session.is_temporal:
                p_h = float(torch.sigmoid(logits_h)[0, traj - 1])
            else:
                p_h = float(torch.softmax(logits_h, dim=1)[0, 1])
        print(f"Hybrid prediction: P={p_h:.6f}")
        print(f"  vs batch P={p_b:.6f} (diff={abs(p_b - p_h):.6f})")

    if n_diff > 0:
        print(f"\n{'='*50}")
        print("HYBRID: batch tabular + inference categorical TS")
        print(f"{'='*50}")
        with torch.no_grad():
            logits_h2 = session.model((
                x_ts_b.unsqueeze(0).float().to(device),
                (x_cat_b.unsqueeze(0).to(device),
                 x_cont_b.unsqueeze(0).float().to(device)),
                x_ts_cat_i.unsqueeze(0).float().to(device),
                traj_b.unsqueeze(0).to(device),
            ))
            if session.is_temporal:
                p_h2 = float(torch.sigmoid(logits_h2)[0, traj - 1])
            else:
                p_h2 = float(torch.softmax(logits_h2, dim=1)[0, 1])
        print(f"Hybrid prediction: P={p_h2:.6f}")
        print(f"  vs batch P={p_b:.6f} (diff={abs(p_b - p_h2):.6f})")

    if n_ts_diff > 0:
        print(f"\n{'='*50}")
        print("HYBRID: inference continuous TS + batch everything else")
        print(f"{'='*50}")
        with torch.no_grad():
            logits_h3 = session.model((
                x_ts_i.unsqueeze(0).float().to(device),
                (x_cat_b.unsqueeze(0).to(device),
                 x_cont_b.unsqueeze(0).float().to(device)),
                x_ts_cat_b.unsqueeze(0).float().to(device),
                traj_b.unsqueeze(0).to(device),
            ))
            if session.is_temporal:
                p_h3 = float(torch.sigmoid(logits_h3)[0, traj - 1])
            else:
                p_h3 = float(torch.softmax(logits_h3, dim=1)[0, 1])
        print(f"Hybrid prediction: P={p_h3:.6f}")
        print(f"  vs batch P={p_b:.6f} (diff={abs(p_b - p_h3):.6f})")
