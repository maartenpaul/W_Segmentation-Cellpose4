#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cellpose training entry point for BIOMERO.

Reads training parameters from config.yaml (primary) or environment
variables (fallback). Prepares cellpose-compatible directory layouts,
runs training, and outputs the model + results to --outfolder.
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from glob import glob
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Config / argument helpers
# ---------------------------------------------------------------------------

def parse_args(argv):
    """Parse CLI arguments matching the biomero container interface."""
    parser = argparse.ArgumentParser(description="Cellpose training")
    parser.add_argument("--infolder", required=True,
                        help="Input data folder (data/in)")
    parser.add_argument("--outfolder", required=True,
                        help="Output folder (data/out)")
    parser.add_argument("--gtfolder", required=True,
                        help="Ground truth folder (data/gt)")
    # Accept and ignore extra BIAflows args so the container doesn't crash
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("-nmc", action="store_true", default=False)
    args, _unknown = parser.parse_known_args(argv)
    return args


def load_config(infolder):
    """Load config.yaml from the data root (one level above --infolder).

    Returns an empty dict if the file does not exist, so env-var-only
    operation is always possible.
    """
    config_path = os.path.join(os.path.dirname(infolder.rstrip("/")),
                               "config.yaml")
    if os.path.isfile(config_path):
        with open(config_path, "r") as fh:
            return yaml.safe_load(fh) or {}
    return {}


def get_param(config, yaml_section, yaml_key, env_name, default=None,
              cast=None):
    """Resolve a parameter: config.yaml > env var > default.

    Parameters
    ----------
    config : dict
        Parsed config.yaml contents.
    yaml_section : str
        Top-level key in config.yaml (e.g. ``"training"``).
    yaml_key : str
        Key inside *yaml_section*.
    env_name : str
        Environment variable name used as fallback.
    default
        Value returned when neither source provides one.
    cast : callable or None
        If given, the resolved value is passed through *cast* (e.g. ``int``).
    """
    section = config.get(yaml_section, {}) or {}
    value = section.get(yaml_key)
    if value is None:
        value = os.environ.get(env_name)
    if value is None:
        value = default
    if cast is not None and value is not None:
        value = cast(value)
    return value


# ---------------------------------------------------------------------------
# Directory preparation
# ---------------------------------------------------------------------------

def prepare_cellpose_dirs(infolder, gtfolder, split):
    """Create a cellpose-compatible directory for a given split.

    Cellpose expects images and ``<stem>_masks.tif`` files side-by-side in
    the same directory. This function symlinks images from
    ``infolder/<split>/`` and masks from ``gtfolder/<split>/`` into a
    temporary directory, renaming masks to match the cellpose convention.

    Returns the path to the prepared directory, or ``None`` if the split
    directory does not exist or is empty.
    """
    img_dir = os.path.join(infolder, split)
    mask_dir = os.path.join(gtfolder, split)

    if not os.path.isdir(img_dir):
        return None

    images = sorted(glob(os.path.join(img_dir, "*.tif")) +
                    glob(os.path.join(img_dir, "*.tiff")))
    if not images:
        return None

    out_dir = f"/tmp/cellpose_{split}"
    os.makedirs(out_dir, exist_ok=True)

    for img_path in images:
        stem = Path(img_path).stem
        # Symlink image
        dst_img = os.path.join(out_dir, os.path.basename(img_path))
        if not os.path.exists(dst_img):
            os.symlink(img_path, dst_img)

        # Find matching mask and symlink with _masks suffix
        mask_candidates = [
            os.path.join(mask_dir, f"{stem}.tif"),
            os.path.join(mask_dir, f"{stem}.tiff"),
            os.path.join(mask_dir, f"{stem}_masks.tif"),
            os.path.join(mask_dir, f"{stem}_masks.tiff"),
        ]
        found_mask = False
        for mask_path in mask_candidates:
            if os.path.isfile(mask_path):
                dst_mask = os.path.join(out_dir, f"{stem}_masks.tif")
                if not os.path.exists(dst_mask):
                    os.symlink(mask_path, dst_mask)
                found_mask = True
                break
        if not found_mask:
            print(f"WARNING: No mask found for {os.path.basename(img_path)} "
                  f"in {mask_dir}")

    return out_dir


# ---------------------------------------------------------------------------
# Model naming
# ---------------------------------------------------------------------------

def generate_model_id(config):
    """Generate a unique, filesystem-safe model ID.

    Uses ``<model_name>_<YYYYMMDD_HHMMSS>`` when a name is provided in the
    config, otherwise falls back to
    ``<workflow_name>_<user_id>_<YYYYMMDD_HHMMSS>``.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    training = config.get("training", {}) or {}
    metadata = config.get("metadata", {}) or {}

    model_name = training.get("model_name")
    if model_name:
        return f"{model_name}_{timestamp}", model_name

    workflow = metadata.get("workflow_name", "cellpose")
    user = metadata.get("trained_by", "unknown")
    model_id = f"{workflow}_{user}_{timestamp}"
    return model_id, model_id


# ---------------------------------------------------------------------------
# Training execution
# ---------------------------------------------------------------------------

def run_training(train_dir, val_dir, config):
    """Build and execute the cellpose training CLI command.

    Returns the subprocess.CompletedProcess result.
    """
    pretrained_model = get_param(config, "training", "pretrained_model",
                                 "PRETRAINED_MODEL", "cpsam")
    n_epochs = get_param(config, "training", "n_epochs",
                         "N_EPOCHS", 100, cast=int)
    learning_rate = get_param(config, "training", "learning_rate",
                              "LEARNING_RATE", 0.00001, cast=float)
    weight_decay = get_param(config, "training", "weight_decay",
                             "WEIGHT_DECAY", 0.1, cast=float)
    batch_size = get_param(config, "training", "batch_size",
                           "BATCH_SIZE", 1, cast=int)
    diameter = get_param(config, "training", "diameter",
                         "DIAMETER", 30, cast=int)
    channels = get_param(config, "training", "channels",
                         "CHANNELS", [0, 0])

    cmd = [
        "cellpose", "--train",
        "--dir", train_dir,
        "--pretrained_model", str(pretrained_model),
        "--n_epochs", str(n_epochs),
        "--learning_rate", str(learning_rate),
        "--weight_decay", str(weight_decay),
        "--train_batch_size", str(batch_size),
        "--diameter", str(diameter),
        "--chan", str(channels[0]) if channels else "0",
        "--chan2", str(channels[1]) if len(channels) > 1 else "0",
        "--mask_filter", "_masks",
        "--verbose",
    ]

    if val_dir:
        cmd.extend(["--test_dir", val_dir])

    # Enable GPU if available
    try:
        import torch
        if torch.cuda.is_available():
            cmd.append("--use_gpu")
    except ImportError:
        pass

    print(f"Running cellpose training: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, text=True)
    return result


# ---------------------------------------------------------------------------
# Model output
# ---------------------------------------------------------------------------

def find_trained_model(train_dir):
    """Locate the most recently trained cellpose model.

    Cellpose saves trained models to ``<train_dir>/models/`` by default.
    Falls back to ``$CELLPOSE_LOCAL_MODELS_PATH`` if not found there.
    The newest model file is returned.
    """
    search_dirs = [
        os.path.join(train_dir, "models"),
        os.environ.get("CELLPOSE_LOCAL_MODELS_PATH",
                        "/tmp/models/cellpose/"),
    ]

    for models_dir in search_dirs:
        if not os.path.isdir(models_dir):
            continue
        model_files = sorted(
            [f for f in glob(os.path.join(models_dir, "*"))
             if os.path.isfile(f)],
            key=os.path.getmtime,
            reverse=True,
        )
        if model_files:
            return model_files[0]

    return None


def save_model(model_file, model_id, outfolder):
    """Copy trained model to persistent storage and create a zip for upload.

    1. Copy to ``/tmp/models/<model_id>/`` (bound to ``$MODELS_PATH`` on
       the host, so the model persists on the Slurm cluster).
    2. Zip and place in ``outfolder`` for retrieval by biomero.
    """
    # Persistent copy
    persist_dir = f"/tmp/models/{model_id}"
    os.makedirs(persist_dir, exist_ok=True)
    shutil.copy2(model_file, persist_dir)
    print(f"Model saved to {persist_dir}")

    # Zip for upload
    os.makedirs(outfolder, exist_ok=True)
    zip_path = os.path.join(outfolder, f"{model_id}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(persist_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, os.path.dirname(persist_dir))
                zf.write(fpath, arcname)
    print(f"Model zipped to {zip_path}")
    return zip_path


def write_results(outfolder, model_id, model_name, config):
    """Write training_results.yaml to outfolder.

    This file is picked up by biomero and used to create OMERO map
    annotations with training provenance.
    """
    training = config.get("training", {}) or {}
    metadata = config.get("metadata", {}) or {}

    results = {
        "model_id": model_id,
        "model_name": model_name,
        "source_datasets": metadata.get("source_datasets", []),
        "trained_by": metadata.get("trained_by", "unknown"),
        "pretrained_model": training.get("pretrained_model", "cpsam"),
        "n_epochs": training.get("n_epochs", 100),
        "learning_rate": training.get("learning_rate", 0.00001),
        "weight_decay": training.get("weight_decay", 0.1),
        "batch_size": training.get("batch_size", 1),
        "diameter": training.get("diameter", 30),
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(outfolder, exist_ok=True)
    results_path = os.path.join(outfolder, "training_results.yaml")
    with open(results_path, "w") as fh:
        yaml.dump(results, fh, default_flow_style=False)
    print(f"Training results written to {results_path}")
    return results


# ---------------------------------------------------------------------------
# Optional test-set evaluation
# ---------------------------------------------------------------------------

def evaluate_test_set(test_dir, model_file, config, outfolder):
    """Run inference on the held-out test set and compute basic metrics.

    Writes per-image mask outputs to ``outfolder/test_predictions/`` and
    returns a metrics dict that can be merged into training_results.yaml.
    """
    if not test_dir or not os.path.isdir(test_dir):
        return None

    images = sorted(glob(os.path.join(test_dir, "*.tif")) +
                    glob(os.path.join(test_dir, "*.tiff")))
    # Filter out mask files
    images = [f for f in images if "_masks" not in os.path.basename(f)]
    if not images:
        return None

    print(f"Evaluating on {len(images)} test images...")

    pred_dir = os.path.join(outfolder, "test_predictions")
    os.makedirs(pred_dir, exist_ok=True)

    training = config.get("training", {}) or {}
    diameter = training.get("diameter", 30)

    cmd = [
        "cellpose",
        "--dir", test_dir,
        "--savedir", pred_dir,
        "--pretrained_model", str(model_file),
        "--diameter", str(diameter),
        "--save_tif",
        "--no_npy",
        "--verbose",
    ]

    try:
        import torch
        if torch.cuda.is_available():
            cmd.append("--use_gpu")
    except ImportError:
        pass

    print(f"Running test evaluation: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Test evaluation failed: {e}")
        return {"error": str(e)}

    return {
        "test_images": len(images),
        "predictions_dir": pred_dir,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv):
    print("=" * 60)
    print("CELLPOSE TRAINING — BIOMERO")
    print("=" * 60)

    # 1. Parse arguments
    args = parse_args(argv)
    print(f"Infolder:  {args.infolder}")
    print(f"Outfolder: {args.outfolder}")
    print(f"GTfolder:  {args.gtfolder}")

    # 2. Load config
    config = load_config(args.infolder)
    if config:
        print(f"Loaded config.yaml with keys: {list(config.keys())}")
    else:
        print("No config.yaml found, using env vars / defaults")

    # 3. Prepare cellpose training directories
    print("\nPreparing training directories...")
    train_dir = prepare_cellpose_dirs(args.infolder, args.gtfolder, "train")
    if not train_dir:
        print("ERROR: No training images found in infolder/train/")
        sys.exit(1)
    print(f"  train_dir: {train_dir} "
          f"({len(glob(os.path.join(train_dir, '*.tif')) + glob(os.path.join(train_dir, '*.tiff')))} files)")

    val_dir = prepare_cellpose_dirs(args.infolder, args.gtfolder, "validation")
    if val_dir:
        print(f"  val_dir:   {val_dir} "
              f"({len(glob(os.path.join(val_dir, '*.tif')))} files)")
    else:
        print("  val_dir:   (none)")

    test_dir = prepare_cellpose_dirs(args.infolder, args.gtfolder, "test")
    if test_dir:
        print(f"  test_dir:  {test_dir} "
              f"({len(glob(os.path.join(test_dir, '*.tif')))} files)")
    else:
        print("  test_dir:  (none)")

    # 4. Generate model ID
    model_id, model_name = generate_model_id(config)
    print(f"\nModel ID:   {model_id}")
    print(f"Model name: {model_name}")

    # 5. Run training
    print("\nStarting cellpose training...")
    try:
        run_training(train_dir, val_dir, config)
        print("Training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Training failed with exit code {e.returncode}")
        sys.exit(1)

    # 6. Find and save trained model
    model_file = find_trained_model(train_dir)
    if not model_file:
        print("ERROR: Could not find trained model file")
        sys.exit(1)
    print(f"Found trained model: {model_file}")

    save_model(model_file, model_id, args.outfolder)

    # 7. Write results
    results = write_results(args.outfolder, model_id, model_name, config)

    # 8. Optional: evaluate on test set
    if test_dir:
        print("\nRunning test set evaluation...")
        test_metrics = evaluate_test_set(test_dir, model_file, config,
                                         args.outfolder)
        if test_metrics:
            results["test_metrics"] = test_metrics
            # Re-write results with test metrics included
            results_path = os.path.join(args.outfolder,
                                        "training_results.yaml")
            with open(results_path, "w") as fh:
                yaml.dump(results, fh, default_flow_style=False)
            print(f"Updated results with test metrics: {test_metrics}")

    # 9. Cleanup temp dirs
    for d in [train_dir, val_dir, test_dir]:
        if d and os.path.isdir(d) and d.startswith("/tmp/cellpose_"):
            shutil.rmtree(d, ignore_errors=True)

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE — model: {model_id}")
    print("=" * 60)


if __name__ == "__main__":
    main(sys.argv[1:])
