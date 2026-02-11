# app.py
# Standalone Module-1 Orchestrator (T0 → T1)
# Skull Strip → Skull QC → N4 → N4 QC → Affine → Affine QC → ANTs SyN → Jacobian

import time
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

# ------------------- IMPORT PIPELINES -------------------
from Modules.Preprocessing.wrapper_skull_strip import SkullStripper
from Modules.Preprocessing.bias_correction import run_bias_correction
from Modules.Preprocessing.affine_register import AffineRegistrationPipeline
from Modules.ants_syn import run_ants_syn

# ------------------- IMPORT EVALUATORS ------------------
from Evaluator.Skull_strip_eval import evaluate_clinical_validity
from Evaluator.Bias_eval import evaluate_n4
from Evaluator.Affine_eval import evaluate_affine

# ------------------- LOGGING SETUP ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MODULE1-APP")

# ------------------- APP LOGIC --------------------------
def run_module1(t0_path: Path, t1_path: Path, workdir: Path):
    workdir.mkdir(parents=True, exist_ok=True)

    paths = {
        "skull": workdir / "01_skull_strip",
        "bias": workdir / "02_bias_corrected",
        "affine": workdir / "03_affine",
        "ants": workdir / "04_ants_syn",
        "qc": workdir / "qc_reports",
    }

    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Module-1 Hybrid Morphometry Engine")
    t_start = time.time()

    with tqdm(total=7, desc="Module-1 Pipeline", unit="step") as pbar:

        # ---------------- STEP 1: SKULL STRIP ----------------
        pbar.set_description("Skull Stripping (HD-BET)")
        skull = SkullStripper(logger=logger)
        t0_brain = skull.run(t0_path, paths["skull"] / "T0")
        t1_brain = skull.run(t1_path, paths["skull"] / "T1")
        pbar.update(1)

        # ---------------- STEP 1 QC -------------------------
        pbar.set_description("Skull Strip QC")
        evaluate_clinical_validity(t0_brain, paths["qc"])
        evaluate_clinical_validity(t1_brain, paths["qc"])
        pbar.update(1)

        # ---------------- STEP 2: N4 -------------------------
        pbar.set_description("Bias Field Correction (N4)")
        run_bias_correction(t0_brain, paths["bias"] / "T0")
        run_bias_correction(t1_brain, paths["bias"] / "T1")

        t0_n4 = next((paths["bias"] / "T0").glob("*_n4.nii.gz"))
        t1_n4 = next((paths["bias"] / "T1").glob("*_n4.nii.gz"))
        pbar.update(1)

        # ---------------- STEP 2 QC -------------------------
        pbar.set_description("N4 QC")
        evaluate_n4(t0_brain, t0_n4)
        evaluate_n4(t1_brain, t1_n4)
        pbar.update(1)

        # ---------------- STEP 3: AFFINE --------------------
        pbar.set_description("Affine Registration")
        affine = AffineRegistrationPipeline(
            fixed_path=t0_n4,
            moving_path=t1_n4,
            output_dir=paths["affine"],
        )
        affine.run()
        t1_affine = paths["affine"] / "T1_affine_aligned.nii.gz"
        pbar.update(1)

        # ---------------- STEP 3 QC -------------------------
        pbar.set_description("Affine QC")
        evaluate_affine(t0_n4, t1_affine)
        pbar.update(1)

        # ---------------- STEP 4: ANTs SyN ------------------
        pbar.set_description("ANTs SyN + Jacobian")
        run_ants_syn(
            fixed_path=t0_n4,
            moving_path=t1_affine,
            output_dir=paths["ants"],
        )
        pbar.update(1)

    elapsed = time.time() - t_start
    logger.info(f"Module-1 completed successfully in {elapsed/60:.2f} minutes")
    logger.info(f"Final outputs available in: {workdir.resolve()}")

# ------------------- CLI -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module-1 Hybrid Morphometry Engine")
    parser.add_argument("--t0", required=True, type=Path, help="Baseline MRI (T0)")
    parser.add_argument("--t1", required=True, type=Path, help="Follow-up MRI (T1)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("module1_outputs"),
        help="Working/output directory",
    )

    args = parser.parse_args()

    run_module1(args.t0, args.t1, args.out)
