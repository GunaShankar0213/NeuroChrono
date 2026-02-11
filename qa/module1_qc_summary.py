"""
Module 1 – QC Summary Script
Computes:
- NCC between fixed and warped
- Max displacement from warp field
- Reads ANTs Jacobian metrics
- Writes final QC JSON summary

Usage:
python qa/module1_qc_summary.py \
    --fixed <fixed_n4.nii.gz> \
    --warped <warped_ants.nii.gz> \
    --warp <warp_ants.nii.gz> \
    --jacobian_metrics <ants_jacobian_metrics.json> \
    --out <module1_qc.json>
"""

import argparse
import json
import numpy as np
import nibabel as nib
import os


def compute_ncc(fixed: np.ndarray, warped: np.ndarray) -> float:
    """Compute masked NCC."""
    mask = (fixed > 0) & (warped > 0)
    f = fixed[mask]
    w = warped[mask]
    if f.size == 0:
        raise ValueError("Empty mask during NCC computation.")
    f = (f - f.mean()) / (f.std() + 1e-8)
    w = (w - w.mean()) / (w.std() + 1e-8)
    return float(np.mean(f * w))


def compute_max_displacement(warp_path: str) -> float:
    """Compute maximum displacement magnitude (mm)."""
    warp_img = nib.load(warp_path)
    warp = warp_img.get_fdata()

    if warp.shape[-1] != 3:
        raise ValueError("Warp field must have 3 displacement components.")

    disp_mag = np.sqrt(np.sum(np.square(warp), axis=-1))
    return float(np.max(disp_mag))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--warped", required=True)
    parser.add_argument("--warp", required=True)
    parser.add_argument("--jacobian_metrics", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    fixed = nib.load(args.fixed).get_fdata()
    warped = nib.load(args.warped).get_fdata()

    ncc_value = compute_ncc(fixed, warped)
    max_disp = compute_max_displacement(args.warp)

    with open(args.jacobian_metrics, "r") as f:
        jac_metrics = json.load(f)

    summary = {
        "mean_jacobian": jac_metrics["mean_jacobian"],
        "std_jacobian": jac_metrics["std_jacobian"],
        "negative_jacobian_pct": jac_metrics["negative_jacobian_pct"],
        "ncc": ncc_value,
        "max_displacement_mm": max_disp,
        "qa_pass": (
            jac_metrics["negative_jacobian_pct"] <= 0.2 and
            0.98 <= jac_metrics["mean_jacobian"] <= 1.02 and
            ncc_value >= 0.90 and
            max_disp <= 15.0
        )
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=4)

    print("QC Summary written to:", args.out)
    print("QA PASS:", summary["qa_pass"])


if __name__ == "__main__":
    main()
