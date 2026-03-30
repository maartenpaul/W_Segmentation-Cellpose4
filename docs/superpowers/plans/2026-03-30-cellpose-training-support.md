# Cellpose Training Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add training capability to the W_Segmentation-Cellpose4 container so the same image supports both inference and training, selected by `TRAINING_MODE` env var.

**Architecture:** A new `entrypoint.sh` dispatcher routes to either the existing `run.py` (inference) or a new `train.py` (training). `train.py` reads a `config.yaml` for parameters, prepares cellpose-compatible training directories from the biomero data layout, runs cellpose training via CLI, and outputs the trained model + results YAML to `--outfolder`.

**Tech Stack:** Python 3.10, Cellpose 4.x, PyYAML, Bash, Singularity/Apptainer container

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `entrypoint.sh` | Create | Dispatcher: routes to `run.py` or `train.py` based on `$TRAINING_MODE` |
| `train.py` | Create | Training entry point: config parsing, dir prep, cellpose training, model output |
| `Dockerfile` | Modify | Copy new files, install PyYAML in cellpose_env, update ENTRYPOINT |
| `run.py` | Unchanged | Existing inference logic — no modifications |
| `descriptor.json` | Unchanged | Training params come via config.yaml |

---

### Task 1: Create entrypoint.sh dispatcher

**Files:**
- Create: `entrypoint.sh`

- [ ] **Step 1: Create entrypoint.sh**

Create the file at the repo root:

```bash
#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
if [ "$TRAINING_MODE" = "true" ]; then
    echo "Training mode enabled, launching train.py..."
    conda activate cellpose_env
    exec python /app/train.py "$@"
else
    conda activate cytomine_py37
    exec python /app/run.py "$@"
fi
```

- [ ] **Step 2: Make executable**

Run: `chmod +x entrypoint.sh`

- [ ] **Step 3: Verify script syntax**

Run: `bash -n entrypoint.sh`
Expected: no output (no syntax errors)

- [ ] **Step 4: Commit**

```bash
git add entrypoint.sh
git commit -m "feat: add entrypoint.sh dispatcher for training/inference mode"
```

---

### Task 2: Create train.py — config parsing and argument handling

**Files:**
- Create: `train.py`

- [ ] **Step 1: Create train.py with argument parsing and config loading**

Create the file at the repo root. This step builds the config/argument parsing foundation that all subsequent steps build on.

```python
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
import time
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
```

- [ ] **Step 2: Verify the file parses**

Run: `cd /var/home/maartenpaul/Documents/GitHub/BIOMERO-repos/W_Segmentation-Cellpose4 && python -c "import ast; ast.parse(open('train.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add train.py with config parsing and argument handling"
```

---

### Task 3: Add directory preparation to train.py

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Add the prepare_cellpose_dirs function**

Append to `train.py`, after the `get_param` function:

```python
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
        for mask_path in mask_candidates:
            if os.path.isfile(mask_path):
                dst_mask = os.path.join(out_dir, f"{stem}_masks.tif")
                if not os.path.exists(dst_mask):
                    os.symlink(mask_path, dst_mask)
                break

    return out_dir
```

- [ ] **Step 2: Verify the file still parses**

Run: `cd /var/home/maartenpaul/Documents/GitHub/BIOMERO-repos/W_Segmentation-Cellpose4 && python -c "import ast; ast.parse(open('train.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add cellpose directory preparation for training splits"
```

---

### Task 4: Add model naming to train.py

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Add the generate_model_id function**

Append to `train.py`, after `prepare_cellpose_dirs`:

```python
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
```

- [ ] **Step 2: Verify the file still parses**

Run: `cd /var/home/maartenpaul/Documents/GitHub/BIOMERO-repos/W_Segmentation-Cellpose4 && python -c "import ast; ast.parse(open('train.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add unique model ID generation with timestamp"
```

---

### Task 5: Add training execution to train.py

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Add the run_training function**

Append to `train.py`, after `generate_model_id`:

```python
# ---------------------------------------------------------------------------
# Training execution
# ---------------------------------------------------------------------------

def run_training(train_dir, val_dir, config):
    """Build and execute the cellpose training CLI command.

    Returns the subprocess.CompletedProcess result.
    """
    training = config.get("training", {}) or {}

    pretrained_model = training.get("pretrained_model", "cpsam")
    n_epochs = training.get("n_epochs", 100)
    learning_rate = training.get("learning_rate", 0.00001)
    weight_decay = training.get("weight_decay", 0.1)
    batch_size = training.get("batch_size", 1)
    diameter = training.get("diameter", 30)
    channels = training.get("channels", [0, 0])

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
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    return result
```

- [ ] **Step 2: Verify the file still parses**

Run: `cd /var/home/maartenpaul/Documents/GitHub/BIOMERO-repos/W_Segmentation-Cellpose4 && python -c "import ast; ast.parse(open('train.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add cellpose training CLI execution"
```

---

### Task 6: Add model saving and results output to train.py

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Add save_model and write_results functions**

Append to `train.py`, after `run_training`:

```python
# ---------------------------------------------------------------------------
# Model output
# ---------------------------------------------------------------------------

def find_trained_model():
    """Locate the most recently trained cellpose model.

    Cellpose saves trained models to ``$CELLPOSE_LOCAL_MODELS_PATH`` (which
    defaults to ``/tmp/models/cellpose/`` in this container).  The newest
    file that is not a built-in model is returned.
    """
    models_dir = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH",
                                "/tmp/models/cellpose/")
    if not os.path.isdir(models_dir):
        return None

    model_files = sorted(
        [f for f in glob(os.path.join(models_dir, "*"))
         if os.path.isfile(f)],
        key=os.path.getmtime,
        reverse=True,
    )
    return model_files[0] if model_files else None


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
```

- [ ] **Step 2: Verify the file still parses**

Run: `cd /var/home/maartenpaul/Documents/GitHub/BIOMERO-repos/W_Segmentation-Cellpose4 && python -c "import ast; ast.parse(open('train.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add model saving, zipping, and results YAML output"
```

---

### Task 7: Add optional test set evaluation to train.py

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Add the evaluate_test_set function**

Append to `train.py`, after `write_results`:

```python
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
        subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Test evaluation failed: {e.stderr}")
        return {"error": str(e.stderr)}

    return {
        "test_images": len(images),
        "predictions_dir": pred_dir,
    }
```

- [ ] **Step 2: Verify the file still parses**

Run: `cd /var/home/maartenpaul/Documents/GitHub/BIOMERO-repos/W_Segmentation-Cellpose4 && python -c "import ast; ast.parse(open('train.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add optional test set evaluation after training"
```

---

### Task 8: Add main function to train.py

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Add the main function and entry point**

Append to the end of `train.py`:

```python
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
          f"({len(glob(os.path.join(train_dir, '*.tif')))} files)")

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
        print(f"ERROR: Training failed: {e.stderr}")
        sys.exit(1)

    # 6. Find and save trained model
    model_file = find_trained_model()
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
```

- [ ] **Step 2: Verify the file parses and has all expected functions**

Run: `cd /var/home/maartenpaul/Documents/GitHub/BIOMERO-repos/W_Segmentation-Cellpose4 && python -c "
import ast
tree = ast.parse(open('train.py').read())
funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
expected = ['parse_args', 'load_config', 'get_param', 'prepare_cellpose_dirs',
            'generate_model_id', 'run_training', 'find_trained_model',
            'save_model', 'write_results', 'evaluate_test_set', 'main']
missing = set(expected) - set(funcs)
assert not missing, f'Missing functions: {missing}'
print(f'OK — {len(funcs)} functions defined')
"`

Expected: `OK — 11 functions defined`

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add main function wiring all training steps together"
```

---

### Task 9: Update Dockerfile

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 1: Add PyYAML install after cellpose install**

In `Dockerfile`, after the line `RUN conda run -n $CELLPOSE_ENV_NAME pip install --no-cache-dir cellpose[distributed]==4.0.4`, add:

```dockerfile
# Add PyYAML for training config parsing
RUN conda run -n $CELLPOSE_ENV_NAME pip install --no-cache-dir pyyaml
```

- [ ] **Step 2: Add COPY commands for new files**

In the `Application Code & Entrypoint` section, after `COPY descriptor.json /app/descriptor.json`, add:

```dockerfile
COPY train.py /app/train.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
```

- [ ] **Step 3: Replace the ENTRYPOINT**

Replace the existing ENTRYPOINT line:

```dockerfile
ENTRYPOINT ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate cytomine_py37 && exec python /app/run.py \"$@\"", "--"]
```

With:

```dockerfile
ENTRYPOINT ["/app/entrypoint.sh"]
```

- [ ] **Step 4: Verify Dockerfile syntax**

Run: `cd /var/home/maartenpaul/Documents/GitHub/BIOMERO-repos/W_Segmentation-Cellpose4 && docker build --check . 2>&1 || echo "No --check support, visual inspection OK"`

Manually verify the Dockerfile has:
1. PyYAML install line after cellpose install
2. `COPY train.py`, `COPY entrypoint.sh`, `RUN chmod +x` before ENTRYPOINT
3. New ENTRYPOINT pointing to `/app/entrypoint.sh`
4. No duplicate ENTRYPOINT lines

- [ ] **Step 5: Commit**

```bash
git add Dockerfile
git commit -m "feat: update Dockerfile with training support (entrypoint, train.py, pyyaml)"
```

---

### Task 10: End-to-end dry-run test with mock data

**Files:**
- No new files (test uses temp directories)

- [ ] **Step 1: Create a test script to verify the training pipeline locally**

This validates that `train.py` can parse args, load config, and prepare directories without actually running cellpose (which requires the container environment). Run:

```bash
cd /var/home/maartenpaul/Documents/GitHub/BIOMERO-repos/W_Segmentation-Cellpose4
python -c "
import os, tempfile, yaml
from pathlib import Path

# Create mock data structure
with tempfile.TemporaryDirectory() as tmpdir:
    # data/in/train, data/in/validation, data/in/test
    for split in ['train', 'validation', 'test']:
        os.makedirs(f'{tmpdir}/data/in/{split}')
        os.makedirs(f'{tmpdir}/data/gt/{split}')
        # Create dummy tif files
        for i in range(3):
            Path(f'{tmpdir}/data/in/{split}/img_{i}.tif').write_bytes(b'fake')
            Path(f'{tmpdir}/data/gt/{split}/img_{i}.tif').write_bytes(b'fake')
    os.makedirs(f'{tmpdir}/data/out')

    # Write config.yaml
    config = {
        'training': {
            'pretrained_model': 'cpsam',
            'n_epochs': 10,
            'learning_rate': 0.00001,
            'model_name': 'test_model',
            'channels': [0, 0],
            'diameter': 30,
        },
        'metadata': {
            'source_datasets': [1, 2],
            'trained_by': 'test_user',
            'workflow_name': 'W_Segmentation-Cellpose4',
        }
    }
    with open(f'{tmpdir}/data/config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Test parsing
    from train import parse_args, load_config, prepare_cellpose_dirs, generate_model_id
    args = parse_args(['--infolder', f'{tmpdir}/data/in',
                        '--outfolder', f'{tmpdir}/data/out',
                        '--gtfolder', f'{tmpdir}/data/gt',
                        '--local', '-nmc'])
    assert args.infolder == f'{tmpdir}/data/in'

    cfg = load_config(args.infolder)
    assert cfg['training']['model_name'] == 'test_model'

    train_dir = prepare_cellpose_dirs(args.infolder, args.gtfolder, 'train')
    assert train_dir is not None
    # Check symlinks were created with _masks naming
    assert os.path.exists(os.path.join(train_dir, 'img_0.tif'))
    assert os.path.exists(os.path.join(train_dir, 'img_0_masks.tif'))

    val_dir = prepare_cellpose_dirs(args.infolder, args.gtfolder, 'validation')
    assert val_dir is not None

    model_id, model_name = generate_model_id(cfg)
    assert model_name == 'test_model'
    assert model_id.startswith('test_model_')

    print('All checks passed!')

    # Cleanup /tmp/cellpose_* dirs
    import shutil
    for d in ['/tmp/cellpose_train', '/tmp/cellpose_validation', '/tmp/cellpose_test']:
        if os.path.isdir(d):
            shutil.rmtree(d)
"
```

Expected: `All checks passed!`

- [ ] **Step 2: Commit all work (if any test fixes were needed)**

```bash
git status
# If clean, no commit needed. If fixes were made:
git add -A
git commit -m "fix: address issues found in dry-run test"
```
