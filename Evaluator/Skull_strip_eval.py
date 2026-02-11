import json
import logging
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_fill_holes, center_of_mass

# ============================================================
# CLINICAL THRESHOLDS (Based on Literature & HD-BET Benchmarks)
# ============================================================
# Sources: 
# - Isensee et al. (HD-BET Validation): Median Volume ~1200-1400cc
# - Shattuck et al. (Brain Volume): Range 950cc - 1800cc
# ============================================================

THRESHOLDS = {
    "volume_cc": {
        "min": 900.0,   # Flag if < 900cc (Possible over-stripping/child brain)
        "max": 1650.0,  # Flag if > 1650cc (Possible neck/eye inclusion)
    },
    "hole_ratio": {
        "max": 0.001    # Max 0.1% void volume allowed inside the mask
    },
    "lateral_asymmetry": {
        "max_shift_mm": 15.0 # Max left-right shift allowed
    }
}

def setup_logger():
    logger = logging.getLogger("ClinicalQC")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def get_volume_cc(mask, zoom):
    """Calculates physical volume in cubic centimeters."""
    voxel_vol_mm3 = np.prod(zoom)
    count = np.sum(mask)
    return (count * voxel_vol_mm3) / 1000.0

def check_topology(mask):
    """Checks for internal holes (topology errors)."""
    filled = binary_fill_holes(mask)
    holes = filled ^ mask
    hole_voxels = np.sum(holes)
    total_voxels = np.sum(filled)
    
    if total_voxels == 0:
        return 1.0 # Error state
        
    return hole_voxels / total_voxels

def check_centering(mask, affine):
    """Checks if the brain center aligns with the image center (approx)."""
    # Get center of mass in voxel coordinates
    com_vox = np.array(center_of_mass(mask))
    
    # Convert to real-world coordinates (mm)
    com_mm = nib.affines.apply_affine(affine, com_vox)
    
    # Image center in real-world coordinates (assuming standard NIfTI origin)
    # Ideally, we check distance from 0,0,0 (isocenter) or image center.
    # Here we check lateral symmetry (Left-Right deviation from 0).
    # In MNI space, X=0 is the midline.
    lateral_drift = abs(com_mm[0]) 
    return lateral_drift

def evaluate_clinical_validity(mask_path: Path, output_dir: Path):
    logger = setup_logger()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Evaluating: {mask_path}")
    
    try:
        img = nib.load(mask_path)
        mask = img.get_fdata() > 0
        zoom = img.header.get_zooms()[:3]
        affine = img.affine
    except Exception as e:
        logger.error(f"Failed to load file: {e}")
        return

    metrics = {}
    flags = []
    verdict = "PASS"

    # 1. Volume Check
    vol_cc = get_volume_cc(mask, zoom)
    metrics["volume_cc"] = round(vol_cc, 2)
    
    if not (THRESHOLDS["volume_cc"]["min"] <= vol_cc <= THRESHOLDS["volume_cc"]["max"]):
        flags.append(f"Volume {vol_cc}cc out of adult range ({THRESHOLDS['volume_cc']['min']}-{THRESHOLDS['volume_cc']['max']})")
        verdict = "WARNING"

    # 2. Topology Check
    hole_ratio = check_topology(mask)
    metrics["hole_ratio"] = round(hole_ratio, 6)
    
    if hole_ratio > THRESHOLDS["hole_ratio"]["max"]:
        flags.append(f"Topology Error: Mask contains {hole_ratio:.2%} holes")
        verdict = "BLOCK"

    # 3. Symmetry/Centering Check
    drift = check_centering(mask, affine)
    metrics["lateral_offset_mm"] = round(drift, 2)
    
    if drift > THRESHOLDS["lateral_asymmetry"]["max_shift_mm"]:
        flags.append(f"High lateral drift detected ({drift}mm). Check registration.")
        verdict = "WARNING"

    # 4. Empty Mask Check
    if vol_cc < 10:
        verdict = "BLOCK"
        flags.append("Mask is empty or nearly empty.")

    # Generate Report
    report = {
        "file": str(mask_path.name),
        "clinical_verdict": verdict,
        "metrics": metrics,
        "flags": flags,
        "reference_benchmarks": {
            "method": "HD-BET (Isensee et al., 2019)",
            "expected_dice": "> 0.97",
            "expected_surface_dist": "< 2.54mm"
        }
    }

    report_path = output_dir / f"{mask_path.stem}_clinical_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"QC Complete. Verdict: {verdict}")
    logger.info(f"Report saved to: {report_path}")
    
    # Print Summary for User
    print("\n=== CLINICAL QC REPORT ===")
    print(f"File:    {mask_path.name}")
    print(f"Volume:  {vol_cc:.2f} cc  (Ref: 900-1650)")
    print(f"Holes:   {hole_ratio:.4%} (Ref: <0.1%)")
    print(f"Verdict: {verdict}")
    if flags:
        print("Flags:")
        for f in flags:
            print(f"  [!] {f}")
    print("==========================\n")

if __name__ == "__main__":
    # Point this to your actual file
    # Note: Using your correct filename 'T0_bet_bet.nii.gz' or mask
    evaluate_clinical_validity(
        mask_path=Path("processed/T0_bet_bet.nii.gz"), 
        output_dir=Path("qc_reports")
    )