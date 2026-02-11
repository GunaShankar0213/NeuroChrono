"""
Affine Registration Evaluation Module
Metrics: 
1. Normalized Cross Correlation (NCC): Checks if internal brain structures match.
2. Dice Coefficient (Mask Overlap): Checks if the brain shapes overlap correctly.
"""
# affine_eval.py

import argparse
import numpy as np
import nibabel as nib
import sys
from pathlib import Path

THRESHOLDS = {
    "NCC_MIN": 0.85,
    "DICE_MIN": 0.90
}

def load_data(path):
    try:
        img = nib.load(path)
        return img.get_fdata()
    except Exception as e:
        print(f"Error loading {path}: {e}")
        sys.exit(1)

def calculate_ncc(fixed, moving, mask):
    f_vals = fixed[mask]
    m_vals = moving[mask]

    f_norm = (f_vals - np.mean(f_vals)) / (np.std(f_vals) + 1e-8)
    m_norm = (m_vals - np.mean(m_vals)) / (np.std(m_vals) + 1e-8)

    return np.mean(f_norm * m_norm)

def calculate_dice(mask_f, mask_m):
    intersection = np.sum(mask_f & mask_m)
    size_f = np.sum(mask_f)
    size_m = np.sum(mask_m)

    if (size_f + size_m) == 0:
        return 0.0

    return 2.0 * intersection / (size_f + size_m)

def evaluate_affine(fixed_path, aligned_path):
    print("-" * 40)
    print("STEP 3 QC: AFFINE EVALUATION")
    print("-" * 40)

    fixed = load_data(fixed_path)
    aligned = load_data(aligned_path)

    brain_mask = (fixed > 0) & (aligned > 0)

    dice_score = calculate_dice(fixed > 0, aligned > 0)
    ncc_score = calculate_ncc(fixed, aligned, brain_mask)

    print("\n--- METRICS ---")
    print(f"Dice Overlap (Shape):    {dice_score:.4f} (Target > {THRESHOLDS['DICE_MIN']})")
    print(f"NCC (Structure):        {ncc_score:.4f} (Target > {THRESHOLDS['NCC_MIN']})")

    verdict = "PASS"
    reasons = []

    if dice_score < THRESHOLDS["DICE_MIN"]:
        verdict = "WARNING"
        reasons.append("Low Shape Overlap")

    if ncc_score < THRESHOLDS["NCC_MIN"]:
        verdict = "WARNING"
        reasons.append("Low Structural Correlation")

    if dice_score < 0.80:
        verdict = "FAIL"
        reasons.append("Critical Misalignment")

    print(f"\nFINAL VERDICT: {verdict}")
    for r in reasons:
        print(f"  [!] {r}")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", required=True, type=Path)
    parser.add_argument("--aligned", required=True, type=Path)
    args = parser.parse_args()

    if not args.fixed.exists() or not args.aligned.exists():
        print("Error: Input files not found.")
        sys.exit(1)

    evaluate_affine(args.fixed, args.aligned)
