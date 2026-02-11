"""
Bias Correction (N4) Validation Module
Metric: Coefficient of Variation (CV) improvement in White Matter (WM) estimate.
"""
# bias_eval.py

import numpy as np
import nibabel as nib
import argparse
from pathlib import Path

def get_wm_mask(data):
    valid_voxels = data[data > 0]
    threshold = np.percentile(valid_voxels, 85)
    return data > threshold

def calculate_cv(data, mask):
    vals = data[mask]
    mu = np.mean(vals)
    sigma = np.std(vals)
    if mu == 0:
        return 0.0
    return sigma / mu

def evaluate_n4(original_path, corrected_path):
    print(f"Loading Original:  {original_path.name}")
    print(f"Loading Corrected: {corrected_path.name}")

    orig = nib.load(original_path).get_fdata()
    corr = nib.load(corrected_path).get_fdata()

    wm_mask = get_wm_mask(corr)

    cv_orig = calculate_cv(orig, wm_mask)
    cv_corr = calculate_cv(corr, wm_mask)

    if cv_orig < 1e-6:
        print("Verdict: SKIPPED (Image already homogeneous)")
        return 0.0

    improvement = ((cv_orig - cv_corr) / cv_orig) * 100.0

    print("\n=== N4 CORRECTION QUALITY REPORT ===")
    print(f"Original White Matter CV:  {cv_orig:.4f}")
    print(f"Corrected White Matter CV: {cv_corr:.4f}")
    print(f"Homogeneity Improvement:   {improvement:.2f}%")

    if improvement > 5.0:
        print("Verdict: PASS")
    elif improvement > 0.0:
        print("Verdict: WARNING")
    else:
        print("Verdict: FAIL")

    return improvement

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    args = parser.parse_args()

    evaluate_n4(args.pre, args.post)
